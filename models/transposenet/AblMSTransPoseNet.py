"""
The TransPoseNet model with Multi-Scene support (Ablation for MSTransPoseNet which uses full Transformers)
"""

import torch
import torch.nn.functional as F
from torch import nn
from .transformer_encoder import Transformer
from .pencoder import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone
from .TransPoseNet import PoseRegressor


class AblMSTransPoseNet(nn.Module):

    def __init__(self, config, pretrained_path):
        """
        config: (dict) configuration of the model
        pretrained_path: (str) path to the pretrained backbone
        """
        super().__init__()

        config["backbone"] = pretrained_path
        config["learn_embedding_with_pose_token"] = True
        num_scenes = config.get("num_scenes")

        # CNN backbone
        self.backbone = build_backbone(config)

        # Position (t) and orientation (rot) encoders
        self.transformer_t = Transformer(config)
        self.transformer_rot = Transformer(config)

        decoder_dim = self.transformer_t.d_model

        # The learned pose token for position (t) and orientation (rot)
        self.pose_token_embed_t = nn.Parameter(torch.zeros((1, decoder_dim)), requires_grad=True)
        self.pose_token_embed_rot = nn.Parameter(torch.zeros((1, decoder_dim)), requires_grad=True)

        # The projection of the activation map before going into the Transformer's encoder
        self.input_proj_t = nn.Conv2d(self.backbone.num_channels[0], decoder_dim, kernel_size=1)
        self.input_proj_rot = nn.Conv2d(self.backbone.num_channels[1], decoder_dim, kernel_size=1)

        # Scene selection
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.scene_embed = nn.Linear(decoder_dim * 2, num_scenes)

        # Whether to use prior from the position for the orientation
        self.use_prior = config.get("use_prior_t_for_rot")

        # Regressors for position (t) and orientation (rot)
        self.regressor_head_t = PoseRegressor(decoder_dim, 3)
        self.regressor_head_rot = PoseRegressor(decoder_dim, 4, self.use_prior)

    def forward_transformers(self, data):
        """
        The forward pass expects a dictionary with key-value 'img' -- NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels NOT USED
        return a dictionary with the following keys--values:
            global_desc_t: latent representation from the position encoder
            global_dec_rot: latent representation from the orientation encoder
            scene_log_distr: the log softmax over the scenes
            max_indices: the index of the max value in the scene distribution

        """
        samples = data.get('img')
        scene_indices = data.get('scene')

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
        local_descs_t = self.transformer_t(self.input_proj_t(src_t), mask_t, pos[0], self.pose_token_embed_t)
        local_descs_rot = self.transformer_rot(self.input_proj_rot(src_rot), mask_rot, pos[1],
                                               self.pose_token_embed_rot)

        # Take the global desc from the pose token
        global_desc_t = local_descs_t[:, 0, :]
        global_desc_rot = local_descs_rot[:, 0, :]

        scene_log_distr = self.log_softmax(
            self.scene_embed(torch.cat((local_descs_t, local_descs_rot), dim=2))).squeeze(2)
        _, max_indices = scene_log_distr.max(dim=1)
        if scene_indices is not None:
            max_indices = scene_indices

        return {'global_desc_t': global_desc_t,
                'global_desc_rot': global_desc_rot,
                'scene_log_distr': scene_log_distr,
                'max_indices': max_indices}

    def forward_heads(self, transformers_res):
        """
        The forward pass execpts a dictionary with two keys-values:
        global_desc_t: latent representation from the position encoder
        global_dec_rot: latent representation from the orientation encoder
        scene_log_distr: the log softmax over the scenes
        max_indices: the index of the max value in the scene distribution
        returns: dictionary with key-value 'pose'--expected pose (NX7) and scene_log_distr
        """
        global_desc_t = transformers_res.get('global_desc_t')
        global_desc_rot = transformers_res.get('global_desc_rot')
        max_indices = transformers_res.get('max_indices')  # We can only use the max index for weights selection

        batch_size = global_desc_t.shape[0]
        expected_pose = torch.zeros((batch_size, 7)).to(global_desc_t.device).to(global_desc_t.dtype)
        for i in range(batch_size):
            x_t = self.regressor_head_t[max_indices[i]](global_desc_t[i].unsqueeze(0))
            x_rot = self.regressor_head_rot[max_indices[i]](global_desc_rot[i].unsqueeze(0))
            expected_pose[i, :] = torch.cat((x_t, x_rot), dim=1)
        return {'pose': expected_pose, 'scene_log_distr': transformers_res.get('scene_log_distr')}

    def forward(self, data):
        """ The forward pass expects a dictionary with key-value 'img' -- NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels NOT USED

            returns dictionary with key-value 'pose'--expected pose (NX7)
        """
        transformers_encoders_res = self.forward_transformers(data)
        # Regress the pose from the image descriptors
        heads_res = self.forward_heads(transformers_encoders_res)
        return heads_res

