# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 14:48:13 2020

@author: eliphat
"""
import torch
import merger.merger_net as merger_net
import json
import tqdm
import numpy as np
import argparse
import open3d as o3d
import os

arg_parser = argparse.ArgumentParser(description="Predictor for Keypoint on KeypointNet dataset.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-m', '--checkpoint-path', '--model-path', type=str, default='./shapenet/model/chair_10.pt',
                        help='Model checkpoint file path to load.')
arg_parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='Pytorch device for predicting.')
arg_parser.add_argument('-k', '--n-keypoint', type=int, default=10,
                        help='Requested number of keypoints to detect.')
arg_parser.add_argument('-b', '--batch', type=int, default=8,
                        help='Batch size.')
arg_parser.add_argument('--max-points', type=int, default=2048,
                        help='Indicates maximum points in each input point cloud.')

ns = arg_parser.parse_args()
net = merger_net.Net(ns.max_points, ns.n_keypoint).to(ns.device)
net.load_state_dict(torch.load(ns.checkpoint_path, map_location=torch.device(ns.device))['model_state_dict'])
net.eval()

out_kpcd = []

x = np.loadtxt("./shapenet/dataset/testdata14.txt")
x = x.reshape(int(x.shape[0]/2048), 2048, 3)

with torch.no_grad():
     key_points, r1= net(torch.FloatTensor(x).to(ns.device),'True')      
for kp in key_points:
    out_kpcd.append(kp)
for i in range(len(out_kpcd)):
    out_kpcd[i] = out_kpcd[i].cpu().numpy()
np.savetxt('./shapenet/keypoint/chair_keypoint.txt', np.array(out_kpcd).reshape(-1,np.array(out_kpcd).shape[-1]))    
