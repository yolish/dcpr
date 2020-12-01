"""
The TransPoseNet model
"""

import torch
import torch.nn.functional as F
from torch import nn
from .transformer import Transformer
from .pencoder import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone
from .TransPoseNet import PoseRegressor


class TransPoseNet(nn.Module):

    def __init__(self, config, pretrained_path):
        """ Initializes the model.
        """
        super().__init__()

        config["backbone"] = pretrained_path
        config["learn_embedding_with_pose_token"] = False
        num_scenes = config.get("num_scenes")
        self.backbone = build_backbone(config)

        self.input_proj_t = nn.Conv2d(self.backbone.num_channels[0], decoder_dim, kernel_size=1)
        self.input_proj_rot = nn.Conv2d(self.backbone.num_channels[1], decoder_dim, kernel_size=1)

        self.transformer_t = Transformer(config)
        self.transformer_rot = Transformer(config)

        decoder_dim = self.transformer_t.d_model

        self.query_embed_t = nn.Embedding(num_scenes, hidden_dim)
        self.query_embed_rot = nn.Embedding(num_scenes, hidden_dim)

        self.log_softmax = nn.LogSoftmax(dim=1)

        self.scene_embed = nn.Linear(hidden_dim, num_scenes)
        self.regressor_head_t = [PoseRegressor(decoder_dim, 3) for _ in range(num_scenes)]
        self.regressor_head_rot = [PoseRegressor(decoder_dim, 4) for _ in range(num_scenes)]

    def forward_encoder(self, samples: NestedTensor):
        """
        The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels NOT USED
        """
        batch_size = samples.shape[0]

        # Handle data structures
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # Extract the features and the position embedding from the visual backbone
        features, pos = self.backbone(samples)

        src_t, mask_t = features[0].decompose()
        src_rot, mask_rot = features[1].decompose()

        # Run through the transformer to translate to "camera-pose" language
        assert mask_t is not None
        assert mask_rot is not None
        local_descs_t = self.transformer_t(self.input_proj_t(src), mask_t, self.query_embed_t.weight, pos[0])[0]
        local_descs_rot = self.transformer_rot(self.input_proj_rot(src), mask_rot, self.query_embed_rot.weight, pos[1])[0]

        scene_log_distr = self.log_softmax(self.scene_embed(torch.cat((local_descs_t, local_descs_rot), dim=1)))
        max_vals = scene_log_distr.max(dim=1)
        for i, v in enumerate(max_vals.unsqueeze(1)):
            scene_log_distr[i] = (scene_log_distr[i] >= v).to(dtype=scene_log_distr.dtype)
        global_desc_t = torch.sum(scene_log_distr*local_descs_t, dim=1)
        global_desc_rot = torch.sum(scene_log_distr * local_descs_rot, dim=1)

        return global_desc_t, global_desc_rot, scene_log_distr


    def forward_heads(self, global_desc_t, global_desc_rot, scene_log_distr):
        # We can only use argmax to index the weights
        expected_pose = global_desc_rot[:, :7]*0
        scene_indices = torch.argmax(scene_log_distr, dim=1)
        for i, scene_index in enumerate(scene_indices):
            chosen_head_t = self.regressor_head_t[scene_index]
            chosen_head_rot = self.regressor_head_rot[scene_index]
            x_t = chosen_head_t(global_desc_t)
            if self.use_prior:
                global_desc_rot = torch.cat((global_desc_t, global_desc_rot), dim=1)
            x_rot = chosen_head_rot(global_desc_rot)
            expected_pose[i, :] = torch.cat((x_t, x_rot), dim=1)
        return expected_pose, scene_log_distr

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels NOT USED

            returns an expected pose and (log) probability distribution over scenes
        """
        global_desc_t, global_desc_rot, scene_log_distr = self.forward_encoder(samples)
        # Regress the pose from the image descriptors
        expected_pose, scene_log_distr = self.forward_heads(global_desc_t, global_desc_rot)
        return expected_pose, scene_log_distr

