# Key-Grid: Unsupervised 3D Keypoints Detection using Grid Heatmap Features


This repository provides the  code for paper: <br>
**Key-Grid: Unsupervised 3D Keypoints Detection using Grid Heatmap Features**
<p align="center">
    <img src="./image/allmethod.png" width="1000"><br>
  
## Overview
Detecting 3D keypoints with semantic consistency is widely used in many scenarios such as pose estimation, shape registration and robotics.
Currently, most unsupervised 3D keypoint detection methods focus on the rigid-body objects. 
However, when faced with deformable objects, the keypoints they identify do not preserve semantic consistency well.
In this paper, we introduce an innovative unsupervised keypoint detector Key-Grid for both the rigid-body and deformable objects, which is an autoencoder framework. 
Unlike previous work, we leverage the identified keypoint information to form a 3D grid feature heatmap called grid heatmap, which is used in the process of point cloud reconstruction.
Grid heatmap is a novel concept that represents the latent variables for grid points sampled uniformly in the 3D cubic space, where these variables are the shortest distance between the grid points and the ``skeleton” connected by keypoint pairs.
Meanwhile, we incorporate the information from each layer of the encoder into the reconstruction process of the point cloud.
We conduct an extensive evaluation of Key-Grid on  a list of benchmark datasets. 
Key-Grid achieves the state-of-the-art performance on the semantic consistency and position accuracy of keypoints.
Moreover, we demonstrate the robustness of Key-Gridto noise and downsampling. 
In addition, we achieve SE-(3) invariance of keypoints though generalizing Key-Grid to a SE(3)-invariant backbone.
## Requiremenmts
* [ShapeNet dataset](https://github.com/qq456cvb/KeypointNet)
* [KeypointNet dataset](https://github.com/qq456cvb/KeypointNet)
* [ClothesNet dataset](https://sites.google.com/view/clothesnet/home)
* [Deep Fashion3D dataset](https://github.com/GAP-LAB-CUHK-SZ/deepFashion3D)
* Python ≥ 3.6
* PyTorch ≥ 1.4
* scikit-learn
* Open3d
* pyvista
## ShapeNet dataset
### Get dataset from ShapeNet
From the `./h5` get the point cloud. You should put the shapenet dataset in the `./shapenet` folder, run:
<pre>python SimCLR/main.py \ 
  --dataset 'cifar100' \ 
  --imb_factor 0.01 \
  --lr 0.5\
  --batch_size 1024 \
  --temperature 0.1 
</pre>
### Train the network

<pre>python SimCLR/linear_classify.py  \
  --dataset 'cifar100' \ 
  --imb_factor 0.01 \
  --train_rule 'DRW' \
  --epochs 200 
</pre>
### Predict the keypoint 
<pre>python SimCLR/linear_classify.py  \
  --dataset 'cifar100' \ 
  --imb_factor 0.01 \
  --train_rule 'DRW' \
  --epochs 200 
</pre>
### Evulate the keypoint
<pre>python SimCLR/linear_classify.py  \
  --dataset 'cifar100' \ 
  --imb_factor 0.01 \
  --train_rule 'DRW' \
  --epochs 200 
</pre>

## ClothesNet dataset
### Get dataset from ClothesNet
<pre>python SimCLR/main.py \ 
  --dataset 'cifar100' \ 
  --imb_factor 0.01 \
  --lr 0.5\
  --batch_size 1024 \
  --temperature 0.1 
</pre>
### Train the network
<pre>python SimCLR/linear_classify.py  \
  --dataset 'cifar100' \ 
  --imb_factor 0.01 \
  --train_rule 'DRW' \
  --epochs 200 
</pre>
### Predict the keypoint 
<pre> python SimCLR/linear_classify.py  \
  --dataset 'cifar100' \ 
  --imb_factor 0.01 \
  --train_rule 'DRW' \
  --epochs 200 
</pre>
### Visualition
<pre> python SimCLR/linear_classify.py  \
  --dataset 'cifar100' \ 
  --imb_factor 0.01 \
  --train_rule 'DRW' \
  --epochs 200 
</pre>

## Acknowledgement
This code inherits some codes from [Skeleton Merger](https://github.com/eliphatfs/SkeletonMerger).
