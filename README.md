# TransPoseNet 
This repository implements the TransPoseNet architecture described in our paper: "Paying Attention to Activation Maps in Camera Pose Regression".

TransPoseNet takes activation maps from a convolutional backbone and processes them with a dual Transformer Encoder head. 

Each encoder outputs a global latent descriptor which is used for regressing the position and orientation of a camera. 

