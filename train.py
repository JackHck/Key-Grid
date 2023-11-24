# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:56:16 2021

@author: eliphat
"""
import random
import math
import argparse
import contextlib
import torch
import torch.optim as optim
import numpy
import numpy as np
from merger.data_flower import all_h5
from merger.merger_net import Net
from merger.composed_chamfer import loss_all


arg_parser = argparse.ArgumentParser(description="Training Key_Grid")
arg_parser.add_argument('-m', '--checkpoint-path', '--model-path', type=str, default='./shapenet/model/chair_10.pt',
                        help='Model checkpoint file path for saving.')
arg_parser.add_argument('-k', '--n-keypoint', type=int, default=10,
                        help='Requested number of keypoints to detect.')
arg_parser.add_argument('-d', '--device', type=str, default='cpu',
                        help='Pytorch device for training.')
arg_parser.add_argument('-b', '--batch', type=int, default=8,
                        help='Batch size.')
arg_parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='Number of epochs to train.')
arg_parser.add_argument('--max-points', type=int, default=2048,
                        help='Indicates maximum points in each input point cloud.')
arg_parser.add_argument("--keynumber", type=int, help="", default=14)
arg_parser.add_argument("--chamfer", type=int, help="", default=20)
arg_parser.add_argument("--lambda_init_points", type=float, help="", default=1.0)
arg_parser.add_argument("--lambda_chamfer", type=float, help="", default=1.0)


def L2(embed):
    return 0.01 * (torch.sum(embed ** 2))


def feed(net, optimizer, x_set, train, shuffle, batch, epoch, ns):
    running_init_points = 0.0
    running_chamfer = 0.0
    running_keypoint = 0.0
    net.train(train)
    if shuffle:
        x_set = list(x_set)
        random.shuffle(x_set)
    with contextlib.suppress() if train else torch.no_grad():
        for i in range(len(x_set) // batch):
            idx = slice(i * batch, (i + 1) * batch)
            refp = next(net.parameters())
            batch_x = torch.FloatTensor(x_set[idx]).cuda()
            if train:
                optimizer.zero_grad()
            keypoint, reconstruct = net(batch_x, 'True')
            loss = loss_all(batch_x, keypoint, reconstruct,epoch,ns)
            running_init_points += loss['init_points']
            if epoch> ns.chamfer:
                running_chamfer += loss['chamfer']
            loss = sum(loss.values())
        
            if train:
                loss.backward()
                optimizer.step()
    
      
            print('[%s%d, %4d] init_point: %.4f chamfer: %.4f keypoint: %.4f' %
                  ('VT'[train], epoch, i, running_init_points / (i + 1), running_chamfer / (i + 1)))

            
if __name__ == '__main__':
    ns = arg_parser.parse_args()
    batch = ns.batch
    
    x = np.loadtxt("./shapenet/dataset/traindata14.txt")
    x = x.reshape(int(x.shape[0]/2048), 2048, 3)
    net = Net(ns.max_points, ns.n_keypoint).cuda()
    optimizer = optim.Adadelta(net.parameters(),lr=0.1,eps=1e-2)
    
    for epoch in range(ns.epochs):
        feed(net, optimizer, x, True, True, batch, epoch, ns)
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
        }, ns.checkpoint_path)
       
        
        
            
