"""
The TransPoseNet model
"""

import torch
import torch.nn.functional as F
from torch import nn
from .transformer import Transformer
from .pencoder import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone


class TransPoseNet(nn.Module):

    def __init__(self, config, pretrained_path):
        """ Initializes the model.
        """
        super().__init__()

        config["backbone"] = pretrained_path
        self.backbone = build_backbone(config)
        self.transformer_t = Transformer(config)
        self.transformer_rot = Transformer(config)

        decoder_dim = self.transformer_t.d_model
        self.pose_token_embed_t = nn.Parameter(torch.zeros((1, decoder_dim)), requires_grad=True)
        self.pose_token_embed_rot = nn.Parameter(torch.zeros((1, decoder_dim)), requires_grad=True)
        self.input_proj_t = nn.Conv2d(self.backbone.num_channels[0], decoder_dim, kernel_size=1)
        self.input_proj_rot = nn.Conv2d(self.backbone.num_channels[1], decoder_dim, kernel_size=1)
        self.use_prior = config.get("use_prior_t_for_rot")
        self.regressor_head_t = PoseRegressor(decoder_dim, 3)
        self.regressor_head_rot = PoseRegressor(decoder_dim, 4, self.use_prior)

    def forward_encoder(self, samples: NestedTensor):
        """
        The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels NOT USED
        """
        multiple_crops = samples.shape[1] > 3
        batch_size = samples.shape[0]

        # Handle multiple crops if needed (flatten)
        num_crops_per_img = 1

        if multiple_crops:
            num_crops_per_img = int(samples.shape[1] / 3)
            samples = samples.contiguous().view(batch_size, num_crops_per_img, 3, samples.shape[2], samples.shape[3])
            samples = samples.contiguous().view(batch_size * num_crops_per_img, 3, samples.shape[3],
                                                samples.shape[4])  # N*NUM_CROPS X C X H X W

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

        # Handle multiple crops
        if num_crops_per_img > 1:
            j = 0
            for i in range(batch_size):
                global_desc_t[i] = global_desc_t[j:(j + num_crops_per_img), :].mean(dim=0)
                global_desc_rot[i] = global_desc_rot[j:(j + num_crops_per_img), :].mean(dim=0)
                j = j + num_crops_per_img
            global_desc_t = global_desc_t[:batch_size, :]
            global_desc_rot = global_desc_rot[:batch_size, :]

        return global_desc_t, global_desc_rot

    def forward_heads(self, global_desc_t, global_desc_rot):
        x_t = self.regressor_head_t(global_desc_t)
        if self.use_prior:
            global_desc_rot = torch.cat((global_desc_t, global_desc_rot), dim=1)
        x_rot = self.regressor_head_rot(global_desc_rot)
        expected_pose = torch.cat((x_t, x_rot), dim=1)
        return expected_pose

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels NOT USED

            returns an expected pose
        """
        global_desc_t, global_desc_rot = self.forward_encoder(samples)
        # Regress the pose from the image descriptors
        expected_pose = self.forward_heads(global_desc_t, global_desc_rot)
        return expected_pose


class PoseRegressor(nn.Module):
    """ A simple pose regressor"""

    def __init__(self, decoder_dim, output_dim, use_prior=False):
        super().__init__()
        ch = 1024
        self.fc_h = nn.Linear(decoder_dim, ch)
        self.use_prior = use_prior
        if self.use_prior:
            self.fc_h_prior = nn.Linear(decoder_dim * 2, ch)
        self.fc_o = nn.Linear(ch, output_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        if self.use_prior:
            x = F.gelu(self.fc_h_prior(x))
        else:
            x = F.gelu(self.fc_h(x))

        return self.fc_o(x)
