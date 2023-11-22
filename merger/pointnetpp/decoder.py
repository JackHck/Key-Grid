import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet_util import PointNetSetAbstractionMsg,PointNetFeaturePropagation



def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

class PointNetReconstruct(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetReconstruct, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2, feature):
        
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
            
            dists = square_distance(xyz1, feature[:,:,1:])
            dists, idx = dists.sort(dim=-1)
            idx = idx[:, :, 0]  # [B, N, 3]
            new_feature = index_points(feature, idx)
            
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points,new_feature], dim=-1)
            #new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

class MLPre(nn.Module):
    def __init__(self):
        super(MLPre, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 16, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 3, 1),
            
        )
    def forward(self, input):
        x = self.mlp1(input)
        x = self.mlp2(x)
        x = x.permute(0, 2, 1)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fp4 = PointNetReconstruct(512+512+256+256+4, [256, 256])
        self.fp3 = PointNetReconstruct(128+128+256+4, [256, 256])
        self.fp2 = PointNetReconstruct(32+64+256+4, [256, 128])
        #self.fp4 = PointNetReconstruct(512+512+256+256, [256, 256])
        #self.fp3 = PointNetReconstruct(128+128+256, [256, 256])
        #self.fp2 = PointNetReconstruct(32+64+256, [256, 128])
        self.fp1 = PointNetReconstruct(128, [128, 128, 128]) 
        self.decoder = MLPre()
        
    def forward(self, feature, encoder_point, encoder_feature):
        
        l3_points_decoder = self.fp4(encoder_point[3], encoder_point[4],encoder_feature[2],encoder_feature[3],feature)
        l2_points_decoder = self.fp3(encoder_point[2], encoder_point[3], encoder_feature[1], l3_points_decoder,feature)
        l1_points_decoder = self.fp2(encoder_point[1], encoder_point[2], encoder_feature[0], l2_points_decoder,feature)
        l0_points_decoder = self.fp1(encoder_point[0], encoder_point[1], None, l1_points_decoder,feature)
        
        x = self.decoder(l0_points_decoder)
        return x
