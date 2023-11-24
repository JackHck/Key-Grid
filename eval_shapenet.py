# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:35:47 2020

@author: eliphat
"""
import json
import numpy as np
import collections
import argparse


arg_parser = argparse.ArgumentParser(description="Evaluation for detected keypoints.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-a', '--annotation-json', type=str, default='./shapenet/annotations/chair.json',
                        help='Annotation JSON file path from KeypointNet dataset.')
arg_parser.add_argument('-i', '--pcd-path', type=str, default='./shapenet/pcd',
                        help='Point cloud file folder path from KeypointNet dataset.')
arg_parser.add_argument('-p', '--prediction', type=str, default='./shapenet/keypoint/chair.npz',
                        help='Prediction file from predictor output.')
arg_parser.add_argument('--miou', default=0.1, 
                        help='Show miou value')


def naive_read_pcd(path):
    lines = open(path, 'r').readlines()
    idx = -1
    for i, line in enumerate(lines):
        if line.startswith('DATA ascii'):
            idx = i + 1
            break
    lines = lines[idx:]
    lines = [line.rstrip().split(' ') for line in lines]
    data = np.asarray(lines)
    pc = np.array(data[:, :3], dtype=np.float)
    return pc

def mIoU(threshold):
    kps = []
    gts = []
    for entry, kpcd, nfact in zip(kpn_ds, predicted['kpcd'], predicted['nfact']):
        cid = entry['class_id']
        mid = entry['model_id']
        pc = naive_read_pcd(r'{}/{}/{}.pcd'.format(ns.pcd_path, cid, mid))
        dmax = nfact[0]
        dmin = nfact[1]
        ground_truths = []
        gtkp = entry['keypoints']
        for kp in gtkp:
            ground_truths.append(pc[kp['pcd_info']['point_index']])
        gts.append(ground_truths)
        npc = (pc - dmin) / (dmax - dmin)
        npc = 2.0 * (npc - 0.5)
        kpcd_e = np.expand_dims(kpcd, 1)  # k1 x 1 x 3
        npc_e = np.expand_dims(npc, 0)  # 1 x k2 x 3
        dist = np.sqrt(np.sum(np.square(kpcd_e - npc_e), -1))  # k1 x k2
        argminfwd = np.argmin(dist, -1)  # k1
        kps.append(pc[argminfwd])
    npos = 0
    fp_sum = 0
    fn_sum = 0
    for ground_truths, kpcd in zip(gts, kps):
        kpcd_e = np.expand_dims(kpcd, 1)  # k1 x 1 x 3
        gt_e = np.expand_dims(ground_truths, 0)  # 1 x k2 x 3
        dist = np.sqrt(np.sum(np.square(kpcd_e - gt_e), -1))  # k1 x k2
        npos += len(np.min(dist, -2))
        fp_sum += np.count_nonzero(np.min(dist, -1) > threshold)
        fn_sum += np.count_nonzero(np.min(dist, -2) > threshold)
    print('miou metric')  
    print((npos - fn_sum) / (npos + fp_sum))




if __name__ == '__main__':
    ns = arg_parser.parse_args()
    with open(ns.annotation_json) as data_file:
        kpn_ds = json.load(data_file)
    predicted = np.load(ns.prediction)
    mIoU(ns.miou)
