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
from merger.composed_chamfer import composed_sqrt_chamfer,loss_all


arg_parser = argparse.ArgumentParser(description="Training Skeleton Merger. Valid .h5 files must contain a 'data' array of shape (N, n, 3) and a 'label' array of shape (N, 1).", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-t', '--train-data-dir', type=str, default='../point_cloud/train',
                        help='Directory that contains training .h5 files.')
arg_parser.add_argument('-v', '--val-data-dir', type=str, default='../point_cloud/val',
                        help='Directory that contains validation .h5 files.')
arg_parser.add_argument('-c', '--subclass', type=int, default=14,
                        help='Subclass label ID to train on.')  # 14 is `chair` class.
arg_parser.add_argument('-m', '--checkpoint-path', '--model-path', type=str, default='./fold/model/pant_new_keypoint_12.pt',
                        help='Model checkpoint file path for saving.')
arg_parser.add_argument('-k', '--n-keypoint', type=int, default=12,
                        help='Requested number of keypoints to detect.')
arg_parser.add_argument('-d', '--device', type=str, default='cpu',
                        help='Pytorch device for training.')
arg_parser.add_argument('-b', '--batch', type=int, default=8,
                        help='Batch size.')
arg_parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='Number of epochs to train.')
arg_parser.add_argument('--max-points', type=int, default=2048,
                        help='Indicates maximum points in each input point cloud.')
arg_parser.add_argument("--keynumber", type=int, help="", default=16)
arg_parser.add_argument("--keypoint", type=int, help="", default=100)
arg_parser.add_argument("--chamfer", type=int, help="", default=20)
arg_parser.add_argument("--lambda_init_points", type=float, help="", default=1.0)
arg_parser.add_argument("--lambda_chamfer", type=float, help="", default=1.0)


def L2(embed):
    return 0.01 * (torch.sum(embed ** 2))


def feed(net, optimizer, x_set, y_set, train, shuffle, batch, epoch, ns):
    running_init_points = 0.0
    running_chamfer = 0.0
    running_keypoint = 0.0
    net.train(train)
    if shuffle:
        x_set = list(x_set)
        y_set = list(y_set)
        random.shuffle(x_set)
        random.shuffle(y_set)
    with contextlib.suppress() if train else torch.no_grad():
        for i in range(len(x_set) // batch):
            idx = slice(i * batch, (i + 1) * batch)
            refp = next(net.parameters())
            batch_x = torch.FloatTensor(x_set[idx]).cuda()
            label_y = torch.FloatTensor(y_set[idx])
            if train:
                optimizer.zero_grad()
            keypoint, reconstruct = net(batch_x,'True')
            loss = loss_all(batch_x, keypoint, reconstruct,epoch,ns,label_y)
            running_init_points += loss['init_points']
            if epoch> ns.chamfer:
                running_chamfer += loss['chamfer']
            if epoch>ns.keypoint:
                running_keypoint += loss['keypoint']
            loss = sum(loss.values())
        
            if train:
                loss.backward()
                optimizer.step()
    
      
            print('[%s%d, %4d] init_point: %.4f chamfer: %.4f keypoint: %.4f' %
                  ('VT'[train], epoch, i, running_init_points / (i + 1), running_chamfer / (i + 1), running_keypoint/ (i + 1)))

            
if __name__ == '__main__':
    ns = arg_parser.parse_args()
    batch = ns.batch
    
    x = np.loadtxt("./fold/for_chengkai_pants/seqdata.txt")
    x = x.reshape(int(x.shape[0]/2048), 2048, 3)
    y = np.loadtxt('./fold/for_chengkai_top_1/label.txt')   
    net = Net(ns.max_points, ns.n_keypoint, 0.5).cuda()
    optimizer = optim.Adadelta(net.parameters(),lr=0.1,eps=1e-2)
    
    for epoch in range(ns.epochs):
        feed(net, optimizer, x, y, True, True, batch, epoch, ns)
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
        }, ns.checkpoint_path)
       
        
        
            