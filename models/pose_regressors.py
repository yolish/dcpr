from .posenet.PoseNet import PoseNet
from .transposenet.TransPoseNet import TransPoseNet
from .transposenet.MSTransPoseNet import MSTransPoseNet
from .transposenet.AblMSTransPoseNet import AblMSTransPoseNet

def get_model(model_name, backbone_path, config):
    """
    Get the instance of the request model
    :param model_name: (str) model name
    :param backbone_path: (str) path to a .pth backbone
    :param config: (dict) config file
    :return: instance of the model (nn.Module)
    """
    if model_name == 'posenet':
        return PoseNet(backbone_path)
    elif model_name == 'transposenet':
        return TransPoseNet(config, backbone_path)
    elif model_name == 'ms-transposenet':
        return MSTransPoseNet(config, backbone_path)
    elif model_name == 'abl-ms-transposenet':
        return AblMSTransPoseNet(config, backbone_path)
    else:
        raise "{} not supported".format(model_name)