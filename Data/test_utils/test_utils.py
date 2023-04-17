import torch

from models.models import BasicXrayNet

from options import utils_option as option

def load_by_opt(opt='options/training/Bilateral.json', pth=""):

    opt = option.parse(opt, is_train=True)
    opt_model = opt["model"]

    backbone_name = opt_model["backbone_name"]
    model_name = opt_model["model_name"]

    in_chans = opt_model["in_chans"]
    drop_out = opt_model["drop_out"]
    backbone_drop_out = opt_model["backbone_drop_out"]
    embed_dim = opt_model["embed_dim"]
    classes = opt_model["classes"]

    net = BasicXrayNet(model_name=backbone_name,
                       pretrained=True, separate=opt["separate_model"],
                       drop_out=drop_out, backbone_drop_out=backbone_drop_out, in_chans=in_chans)

    net = load_by_path(net, pth)
    net.eval()

    return net

def load_by_path(net, pth, device="cuda"):
    checkpoints = torch.load(pth,
                             map_location=torch.device(device))
    net.load_state_dict(checkpoints)
    return net
