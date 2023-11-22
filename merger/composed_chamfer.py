# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 16:33:57 2021

@author: eliphat
"""
import torch
import torch.nn as nn
import pytorch3d.loss
import pytorch3d.utils
from einops import rearrange
from einops import repeat
from simpleicp import PointCloud, SimpleICP

torch.square = lambda x: x ** 2
torch.minimum = lambda x, y: torch.min(torch.stack((x, y)), dim=0)[0]

def sample_farthest_points(points, num_samples, return_index=False):
    points =  points.transpose(1, 2)
    b, c, n = points.shape
    sampled = torch.zeros((b, 3, num_samples), device=points.device, dtype=points.dtype)
    indexes = torch.zeros((b, num_samples), device=points.device, dtype=torch.int64)

    index = torch.randint(n, [b], device=points.device)

    gather_index = repeat(index, 'b -> b c 1', c=c)
    sampled[:, :, 0] = torch.gather(points, 2, gather_index)[:, :, 0]
    indexes[:, 0] = index
    dists = torch.norm(sampled[:, :, 0][:, :, None] - points, dim=1)

    # iteratively sample farthest points
    for i in range(1, num_samples):
        _, index = torch.max(dists, dim=1)
        gather_index = repeat(index, 'b -> b c 1', c=c)
        sampled[:, :, i] = torch.gather(points, 2, gather_index)[:, :, 0]
        indexes[:, i] = index
        dists = torch.min(dists, torch.norm(sampled[:, :, i][:, :, None] - points, dim=1))

    if return_index:
        return sampled, indexes
    else:
        return sampled

def composed_sqrt_chamfer(y_true, y_preds, activations):
    L = 0.0
    # activations: N x P where P: # sub-clouds
    # y_true: N x ? x 3
    # y_pred: P x N x ? x 3 
    part_backs = []
    for i, y_pred in enumerate(y_preds):
        # y_true: k1 x 3
        # y_pred: k2 x 3
        y_true_rep = torch.unsqueeze(y_true, axis=-2)  # k1 x 1 x 3
        y_pred_rep = torch.unsqueeze(y_pred, axis=-3)  # 1 x k2 x 3
        # k1 x k2 x 3
        y_delta = torch.sqrt(1e-4 + torch.sum(torch.square(y_pred_rep - y_true_rep), -1))
        # k1 x k2
        y_nearest = torch.min(y_delta, -2)[0]
        # k2
        part_backs.append(torch.min(y_delta, -1)[0])
        L = L + torch.mean(torch.mean(y_nearest, -1) * activations[:, i]) / len(y_preds)
    part_back_stacked = torch.stack(part_backs)  # P x N x k1
    sorted_parts, indices = torch.sort(part_back_stacked, dim=0)
    weights = torch.ones_like(sorted_parts[0])  # N x k1
    for i in range(len(y_preds)):
        w = torch.minimum(weights, torch.gather(activations, -1, indices[i]))
        L = L + torch.mean(sorted_parts[i] * w)
        weights = weights - w
    L = L + torch.mean(weights * 20.0)
    return L


def loss_all(batch_x, keypoint, reconstruct, epoch, ns, label):
    losses = {}
    sample_point = sample_farthest_points(batch_x, ns.keynumber)
    init_points_loss = pytorch3d.loss.chamfer_distance(keypoint, rearrange(sample_point, 'b d n -> b n d'))[0]
    losses['init_points'] = ns.lambda_init_points * init_points_loss

    if epoch> ns.chamfer:   
        B,C,W  = batch_x.shape
        #normal_1, normal_2 = torch.split(batch_x, int(B/2), dim = 0)
        #re1_chamfer_loss = pytorch3d.loss.chamfer_distance(reconstruct_1,  normal_1)[0]
        #re2_chamfer_loss = pytorch3d.loss.chamfer_distance(reconstruct_2,  normal_2)[0]
        chamfer_loss = pytorch3d.loss.chamfer_distance(reconstruct,batch_x)[0]
        losses['chamfer'] = ns.lambda_chamfer * chamfer_loss

    if epoch>ns.keypoint:
        losses['keypoint'] = 0
        B,C,W  = batch_x.shape
        keypoint_1, keypoint_2 = torch.split(keypoint,int(B/2), dim = 0)
        normal_1, normal_2 = torch.split(batch_x, int(B/2), dim = 0)
        label_1, label_2 = torch.split(label,int(B/2), dim = 0)
        pdist = nn.PairwiseDistance(p=2)
        for j in range(int(B/2)):
            for i in range(ns.n_keypoint):
                distance = pdist(keypoint_1[j,i,:].unsqueeze(0), normal_1[j])
                index = torch.argmin(distance)
                if label_1[j, index] in label_2[j]:
                    for k in range(2048):
                        if label_1[j, index] == label_2[j,k]:
                            losses['keypoint'] = losses['keypoint']+pytorch3d.loss.chamfer_distance(keypoint_2[j].reshape(1,ns.n_keypoint,3),normal_2[j,k].reshape(1,1,3))[0]

        losses['keypoint'] = ns.lambda_chamfer*losses['keypoint']/(int(B/2)*ns.n_keypoint)  
        '''
        pdist = nn.PairwiseDistance(p=2)
        for j in range(int(B/2)):
            for i in range(ns.n_keypoint):
                distance = pdist(keypoint_1[j,i,:].unsqueeze(0), normal_1[j])
                index = torch.argmin(distance)
                batch_distance = pdist(normal_1[j,index].unsqueeze(0),normal_2[j])
                New_index = torch.argmin(batch_distance)
                losses['keypoint'] = losses['keypoint']+pytorch3d.loss.chamfer_distance(keypoint_2[j].reshape(1,ns.n_keypoint,3),normal_2[j,New_index].reshape(1,1,3))[0]
        losses['keypoint'] = ns.lambda_chamfer*losses['keypoint']/(int(B/2)*ns.n_keypoint) 



        losses['keypoint'] = 0
        keypoint_1, keypoint_2 = torch.split(keypoint,int(B/2), dim = 0)
        normal_1 =  normal_1.detach().cpu().numpy()
        normal_2 =  normal_2.detach().cpu().numpy()
        for i in range(int(B/2)):
            X_fix =  normal_1[i].reshape(2048,3)
            X_mov =  normal_2[i].reshape(2048,3)
            pc_fix = PointCloud(X_fix, columns=["x", "y", "z"])
            pc_mov = PointCloud(X_mov, columns=["x", "y", "z"])
            icp = SimpleICP()
            icp.add_point_clouds(pc_fix, pc_mov)
            H, X_mov_transformed, rigid_body_transformation_params, distance_residuals = icp.run(max_overlap_distance=2)
            H =  torch.from_numpy(H).to(keypoint).cuda()
            keypoint_mov= torch.cat((keypoint_2[i].reshape(ns.n_keypoint,3), torch.ones(ns.n_keypoint,1)), dim=1).T.cuda()
            keypoint_2_mov = torch.mm(H, keypoint_mov)[:3].T.unsqueeze(0)
            keypoint_1_fix = keypoint_1[i].reshape(ns.n_keypoint,3).unsqueeze(0).cuda()
            losses['keypoint'] = losses['keypoint']+pytorch3d.loss.chamfer_distance(keypoint_1_fix,  keypoint_2_mov)[0]
        '''
    return losses
