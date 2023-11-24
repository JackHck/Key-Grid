# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 16:50:30 2020

@author: eliphat
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from.pointnetpp.decoder import Decoder
from .pointnetpp.pointnet2_sem_seg_msg import get_model as PointNetPP


torch.square = lambda x: x ** 2


def gen_grid2d(grid_size: int, left_end: float=-1, right_end: float=1) -> torch.Tensor:
    x = torch.linspace(left_end, right_end, 16)
    z = torch.linspace(left_end, right_end, 8)
    x, y, z = torch.meshgrid([x, x, z])
    grid = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), dim=1).reshape(1, 2048, 3)
    return grid


class PBlock(nn.Module):  # MLP Block
    def __init__(self, iu, *units, should_perm):
        super().__init__()
        self.sublayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.should_perm = should_perm
        ux = iu
        for uy in units:
            self.sublayers.append(nn.Linear(ux, uy))
            self.batch_norms.append(nn.BatchNorm1d(uy))
            ux = uy

    def forward(self, input_x):
        x = input_x
        for sublayer, batch_norm in zip(self.sublayers, self.batch_norms):
            x = sublayer(x)
            if self.should_perm:
                x = x.permute(0, 2, 1)
            x = batch_norm(x)
            if self.should_perm:
                x = x.permute(0, 2, 1)
            x = F.relu(x)
        return x


class Head(nn.Module):  # Decoder unit, one per line segment
    def __init__(self):
        super().__init__()
        self.emb = nn.Parameter(torch.randn((200, 3)) * 0.002)

    def forward(self, KPA, KPB):
        dist = torch.mean(torch.sqrt(1e-3 + (torch.sum(torch.square(KPA - KPB), dim=-1))))
        count = min(200, max(15, int((dist / 0.01).item())))
        device = dist.device
        self.f_interp = torch.linspace(0.0, 1.0, count).unsqueeze(0).unsqueeze(-1).to(device)
        self.b_interp = 1.0 - self.f_interp
        # KPA: N x 3, KPB: N x 3
        # Interpolated: N x count x 3
        x = KPA.unsqueeze(-2) * self.f_interp + KPB.unsqueeze(-2) * self.b_interp
        R = self.emb[:count, :].unsqueeze(0) + x  # N x count x 3
        return R.reshape((-1, count, 3)), self.emb

def mask_feature (input, mask_rate):
        if mask_rate>0:
            B,N,_ =  input.shape
            mask_N, mask_long= int(mask_rate*N),int((1-mask_rate)*N)
            #random_N = torch.randint(0,mask_N-1,(1,))
            random_N =1
            #damage_mask = torch.zeros(input.shape[0], input.shape[1], 1, device=input.device).uniform_() > mask_rate
            #damage_mask = F.interpolate(damage_mask.to(input), size=1, mode='nearest')
            damage_mask = torch.zeros(input.shape[0], input.shape[1], 1, device=input.device)
            damage_mask [:,random_N:random_N+mask_long] =1
            damage_mask = F.interpolate(damage_mask.to(input), size=1, mode='nearest')
            mask_input = input * damage_mask
            return mask_input
        else:
            return input
        

class Net(nn.Module):  # Skeleton Merger structure
    def __init__(self, npt, k):
        super().__init__()
        self.npt = npt
        self.k = k
        self.rate = rate
        self.PTW = PointNetPP(k)
        self.PT_L = nn.Linear(k, k)
        self.skeleton_idx = torch.triu_indices(k, k, offset=1)
        self.MA_EMB = nn.Parameter(torch.randn([len(self.skeleton_idx[0])]))
        self.MA = PBlock(1024, 512, 256, should_perm=False)
        self.MA_L = nn.Linear(256, len(self.skeleton_idx[0]))
        self.DEC = nn.ModuleList()
        self.decoder = Decoder()
        for i in range(k):
            DECN = nn.ModuleList()
            for j in range(i):
                DECN.append(Head())
            self.DEC.append(DECN)

    def draw_lines(normal_point,paired_joints):
        bs, n_points, _, _ = paired_joints.shape
        start = paired_joints[:, :, 0, :]   # (batch_size, n_points, 3)
        end = paired_joints[:, :, 1, :]     # (batch_size, n_points, 3)
        paired_diff = end - start           # (batch_size, n_points, 3)
        grid = normal_point.reshape(1, 1, -1, 3)
        diff_to_start = grid - start.unsqueeze(-2)  # (batch_size, n_points,2048, 3)
        t = (diff_to_start @ paired_diff.unsqueeze(-1)).squeeze(-1) / (1e-8+paired_diff.square().sum(dim=-1, keepdim=True))

        diff_to_end = grid - end.unsqueeze(-2)  # (batch_size, n_points, 2048, 3)

        before_start = (t <= 0).float() * diff_to_start.square().sum(dim=-1)
        after_end = (t >= 1).float() * diff_to_end.square().sum(dim=-1)
        between_start_end = (0 < t).float() * (t < 1).float() * (grid - (start.unsqueeze(-2) + t.unsqueeze(-1) * paired_diff.unsqueeze(-2))).square().sum(dim=-1)

        squared_dist = (before_start + after_end + between_start_end).reshape(bs, n_points,2048)
        heatmaps = torch.exp(- squared_dist / 2.5e-3)
        return heatmaps

    def forward(self, input_x,train):
        B,C,W  = input_x.shape
        normal_point = gen_grid2d(C,-1,1).cuda()
        point_cloud = torch.cat([input_x, input_x, input_x], -1)
        kp_x, l3_feats,re_points,re_feature = self.PTW(point_cloud.permute(0, 2, 1))
        kp_x = self.PT_L(kp_x)
        if train == 'True':
            kp_heatmaps = F.softmax(kp_x.permute(0, 2, 1), -1)  # [n, k, npt]
        else:
            kp_heatmaps = F.softmax(kp_x.permute(0, 2, 1), -1) 
            kp_heatmaps = (kp_heatmaps == kp_heatmaps.max(dim=2, keepdim=True)[0]).to(input_x)
        kpcd = kp_heatmaps.bmm(input_x)  # KeyPoint ClouD [n, k, 3]
       
        paired_joints = torch.stack([kpcd[:,self.skeleton_idx[0],:], kpcd[:,self.skeleton_idx[1],:]], dim=2)
        global_feats = F.max_pool1d(l3_feats, 16).squeeze()
        strengths = F.sigmoid(self.MA_L(self.MA(global_feats)))
        feature_map = draw_lines(normal_point, paired_joints)
        feature_map =  feature_map * strengths.reshape(B, len(self.skeleton_idx[0]), 1)
        feature_map = feature_map.max(dim=1, keepdim=True)[0].transpose(1, 2)
        feature_map = torch.cat([feature_map,normal_point.repeat(B,1,1)], dim=-1)
        reconstruct = self.decoder(feature_map,re_points,re_feature)
        
        return kpcd, reconstruct
       
      
   
        
        
        
