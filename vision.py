
import os
import visualizations as visualizer
import random
import numpy as np



pointcloud = np.loadtxt(r"./testdata") # chairdataset.txt
pointcloud = pointcloud.reshape(int(pointcloud.shape[0]/2048), 2048, 3)
             
keypoint = np.loadtxt(r"") #   keypoint.txt 
keypoint = keypoint.reshape(int(keypoint.shape[0]/10), 10, 3)

for i in range(keypoint.shape[0]):
   visualizer.save_kp_and_pc_in_pcd(pointcloud[i],keypoint[i], '{}_visualizations'.format(r''), save=True,name="{}".format(i))
