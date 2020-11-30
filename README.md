# TransPoseNet 
This repository implements the TransPoseNet architecture described in our paper: "Paying Attention to Activation Maps in Camera Pose Regression".

In addition, it allows training and testing basekne camera pose regressor architecturse (PoseNet) as well as TransPoseNet-based variants. 

TransPoseNet takes activation maps from a convolutional backbone and processes them with a dual Transformer Encoder head. Each encoder outputs a global latent descriptor which is used for regressing the position and orientation of a camera. 

