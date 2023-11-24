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



def loss_all(batch_x, keypoint, reconstruct, epoch, ns):
    losses = {}
    sample_point = sample_farthest_points(batch_x, ns.keynumber)
    init_points_loss = pytorch3d.loss.chamfer_distance(keypoint, rearrange(sample_point, 'b d n -> b n d'))[0]
    losses['init_points'] = ns.lambda_init_points * init_points_loss
    
    if epoch> ns.chamfer:   
        B,C,W  = batch_x.shape
        chamfer_loss = pytorch3d.loss.chamfer_distance(reconstruct,batch_x)[0]
        losses['chamfer'] = ns.lambda_chamfer * chamfer_loss
        
    return losses
