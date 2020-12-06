# TransPoseNet 
This repository implements the TransPoseNet architecture described in our paper: "Paying Attention to Activation Maps in Camera Pose Regression".

TransPoseNet introduces an attention-based camera pose regression scheme (Fig. 1). 

[Figure 1 TransPoseNet: an attention-based camera pose regression](yolish.github.com/transposenet/img/transposenet.png)

The input image is
first encoded by a convolutional backbone. Two activation maps, at different resolutions, are transformed into sequential representations. The two activation sequences are analyzed by dual Transformer encoders, one per regression task. We depict the attention weights via
heatmaps. Position is best estimated by corner-like image features, while orientation is estimated by edge-like features. Each Transformer encoder output is  used to regress the respective camera pose component (position or orientation).


