import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import copy


class PoseNet(nn.Module):
    """
    A class to represent a classic pose regressor (PoseNet) with resnet-152 backbone
    PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization,
    Kendall et al., 2015
	 """

    def __init__(self, backbone_path):
        """
        Constructor
        :param backbone_path: backbone path to a resnet backbone
        :param use_elu: (bool) indicates whether to use ELU or RELU activation
        """
        super(PoseNet, self).__init__()

        # Load resnet 152
        backbone = torchvision.models.resnet152(pretrained=False)
        backbone.load_state_dict(torch.load(backbone_path))

        # Remove the classifier heads and pooling
        self.backbone = copy_modules(backbone, 0, -2)

        # Regressor layers
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 3)
        self.fc3 = nn.Linear(1024, 4)

        self.dropout = nn.Dropout(p=0.2)
        self.avg_pooling_2d = nn.AdaptiveAvgPool2d(1)

        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        """
        Forward pass
        :param x: (torch.Tensor) query input image (N X C X H X W)
        :return: (torch.Tensor) 7-dimensional absolute pose for (N X 7)
        """
        x = self.avg_pooling_2d(self.backnone(x))  # N X 3 X 224 X 224 -> Nx2048x7x7 -> Nx2048x1
        x = x.view(x.size(0), -1)  # output shape Nx2048

        x = self.dropout(F.relu(self.fc1_gp(x)))
        p_x = self.fc2(x)
        p_q = self.fc3(x)
        return torch.cat((p_x, p_q), dim=1)


def copy_modules(model, start_idx, end_idx):
    """
    Copy modules from a model
    :param net: (nn.Module) the network to copy
    :param start_idx: (int) index of the module where the copy should start
    :param end_idx: (int) index of the module where the copy should end (exclusive)
    :return: deep copy of submodel
    """
    modules = list(model.children())[start_idx:end_idx]

    # Copy the modules
    sub_model = nn.Sequential(*modules)
    params_orig = model.state_dict()
    params_truncated = sub_model.state_dict()

    # Copy the parameters
    for name, param in params_orig.items():
        if name in params_truncated:
            params_truncated[name].data.copy_(param.data)

    # Load parameters into the architecture
    sub_model.load_state_dict(params_truncated)
    return copy.deepcopy(sub_model)

