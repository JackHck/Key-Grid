
import os
import visualizations as visualizer
import random
import numpy as np





pointcloud = np.loadtxt(r"C:\Users\RF5TG0OP-HOUCHENGKAI\Desktop\pointcloud\shapenet\testdata38.txt")
pointcloud = pointcloud.reshape(int(pointcloud.shape[0]/2048), 2048, 3)[:30]
             
keypoint = np.loadtxt(r"C:\Users\RF5TG0OP-HOUCHENGKAI\Desktop\pointcloud\shapenet_keypoint\_keypoint.txt")    
keypoint = keypoint.reshape(int(keypoint.shape[0]/10), 10, 3)
#keypoint1 = np.loadtxt(r"C:\Users\RF5TG0OP-HOUCHENGKAI\Desktop\pointcloud\foldpant\pant_new_nogridheatmap_point_keypoint65.txt")    
#keypoint1 = keypoint1.reshape(int(keypoint1.shape[0]/8), 8, 3)

#keypoint = np.append(keypoint,keypoint1,axis=0)

for i in range(keypoint.shape[0]):
   #keypoint[i] = keypoint[i]+np.array([random.uniform(-0.015,0.015),random.uniform(-0.015,0.015),random.uniform(-0.015,0.015)])
   #for j in range(3):
  #    keypoint[i][j] = keypoint[i][j]+np.array([random.uniform(-0.15,0.15),random.uniform(-0.15,0.15),random.uniform(-0.15,0.15)])
   
   #if i in [69]:
   #  for j in range(3):
   #     keypoint[i][j] = keypoint[i][j]+np.array([random.uniform(-0.15,0.15),random.uniform(-0.15,0.15),random.uniform(-0.15,0.15)])
   visualizer.save_kp_and_pc_in_pcd(pointcloud[i],keypoint[i], '{}_visualizations'.format(r'D:\keypoint\shapenet\motorbike'), save=True,name="{}_original".format(i))
