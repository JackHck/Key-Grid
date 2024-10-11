# [NeurIPS 2024] Key-Grid: Unsupervised 3D Keypoints Detection using Grid Heatmap Features
[![Arxiv](https://img.shields.io/badge/ArXiv-2109.11377-orange.svg)](https://arxiv.org/abs/2410.02237) 

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
* [ClothesNet dataset](https://sites.google.com/view/clothesnet/home)
* PyTorch ≥ 1.4
* Scikit-learn
* Open3d
* Pyvista
* Pytorch3d
## ClothesNet dataset
### Train the network
**Train the pointnet++ on the train dataset.** Here, we take the chair category as an example. Note, if you want to train other categories, you should change the dataset root.
<pre>
python train.py 
</pre>
### Predict the keypoint
**Predict the keypoint on the test dataset.** We get the keypoint predicted by the pointnet++ on the ClothesNet dataset.
<pre>
python predict_keypoint.py  
</pre>
### Visualition
In this section, we provide code to visualize keypoints and point cloud. **1.Predict the keypoints on the test dataset.** 
You should run  `predict_keypoint.py`; **2. Visualization results of keypoints and point cloud.** You should run  `vision.py`.
## Video
<video width="320" height="240" controls>
  <source src="https://example.com/path/to/example_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Acknowledgement
This code inherits some codes from [Skeleton Merger](https://github.com/eliphatfs/SkeletonMerger), [SC3K](https://github.com/IIT-PAVIS/SC3K).
