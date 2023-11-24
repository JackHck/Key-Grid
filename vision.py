
import os
import visualizations as visualizer
import random
import numpy as np



pointcloud = np.loadtxt(r"../shapenet/dataset/testdata14.txt") # chairdataset.txt
pointcloud = pointcloud.reshape(int(pointcloud.shape[0]/2048), 2048, 3)
             
keypoint = np.loadtxt(r"./shapenet/keypoint/chair_keypoint.txt") #   keypoint.txt 
keypoint = keypoint.reshape(int(keypoint.shape[0]/10), 10, 3)

for i in range(keypoint.shape[0]):
   visualizer.save_kp_and_pc_in_pcd(pointcloud[i],keypoint[i], '{}_visualizations'.format(r'./shapenet/visual/chair.txt'), save=True,name="{}".format(i))
