import os
import gc
import collections
import copy
import time
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from utils.utils import moveTo
from utils.pytorchtools import EarlyStopping
from Data.test_utils.tta import tta_inference
import copy
from torch.nn.parallel import DataParallel, DistributedDataParallel
import os
import gc
import collections
import copy
import time
import torch
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
from sklearn.metrics import roc_auc_score

# from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.utils import moveTo
from utils.pytorchtools import EarlyStopping
from Data.test_utils.tta import tta_inference
import copy
from torch.nn.parallel import DataParallel, DistributedDataParallel

tta_list = [0, 1, 5, 15, 11]

from functions import *


def fix_name(state_dict):
    for name in list(state_dict.keys()):

        if "regressor.4." in name:
            state_dict[name.replace("regressor.4.", "regressor.3.")] = state_dict.pop(name)

    return state_dict


def replace_key_name_for_feature_only(state_dict):
    for name in list(state_dict.keys()):

        if "stages." in name:
            state_dict[name.replace("stages.", "stages_")] = state_dict.pop(name)
        if "stem." in name:
            state_dict[name.replace("stem.", "stem_")] = state_dict.pop(name)
        if "head." in name:
            state_dict.pop(name)

    return state_dict


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


class kaggleTester():
    def __init__(self, opt, cls_model, lmk_model,
                 checkpoints_dir="./checkpoints", get_layer_feature=False
                 ) -> None:

        self.opt = opt
        self.model = cls_model['model']
        self.get_layer_feature = get_layer_feature
        self.checkpoints_dir = checkpoints_dir

    def load(self, best_model_wts: str = None,
             device="cpu"):

        print("Load Model...")
        if best_model_wts:
            checkpoints = torch.load(best_model_wts,
                                     map_location=torch.device(device))
            if self.get_layer_feature:
                self.model.load_state_dict(replace_key_name_for_feature_only(
                    fix_name(checkpoints)))  # replace_key_name_for_feature_only(checkpoints))
            else:
                self.model.load_state_dict(fix_name(checkpoints))

        self.model.eval()
        self.model.to(device)

    def test(self, test_data, score_funcs: dict,
             is_bilateral=False, cam=False, tta=tta_list):

        results = collections.defaultdict(list)

        with torch.no_grad():
            test_loss = _run_test_epoch(
                opt=self.opt,
                cls_models=[self.model],
                lmk_models=[None] * 5,
                data_loader=test_data,
                results=results,
                score_funcs=score_funcs,
                is_bilateral=is_bilateral,
                prefix="test", tta=tta, cam=cam)

        results = pd.DataFrame.from_dict(results)
        print(results)


from models.base_model import BaseModel
from models.base_model import *
import torch.nn as nn

import cv2
import numpy as np


class BasicXrayNet_test(BaseModel):
    def __init__(self, model_name, pretrained=True, cross=False,
                 in_chans=1, drop_out=0.5, backbone_drop_out=0.1, num_classes=1,
                 embed=256, separate=True, get_layer_feature=False) -> None:

        super(BasicXrayNet_test, self).__init__()

        self.model_name = model_name
        self.pretrained = pretrained
        self.in_chans = in_chans
        self.drop_out = drop_out

        self.create_backbone(model_name, pretrained, in_chans, backbone_drop_out, get_layer_feature=get_layer_feature)
        self.separate = separate

        self.create_attention(self.backbone_embed, embed)

        final_embed = 3 * embed

        self.create_sl_attention(final_embed)  # final
        self.create_head(final_embed, self.backbone_embed, drop_out)
        self.embed = final_embed

        self.get_layer_feature = get_layer_feature

    def forward(self, x, points, pointwise=False, return_features=False):

        x_left_bk, x_right_bk = self.backbone_forward(x)

        if not self.get_layer_feature:
            x_left_bk = [x_left_bk]
            x_right_bk = [x_right_bk]

        x_left, x_right = self.res_attn_forward(x_left_bk[-1], x_right_bk[-1])

        self.points = points

        # Pooling Features, [B//2, embed]
        out_left_ = self.global_pool(x_left)[:, :, 0, 0]
        out_right_ = self.global_pool(x_right)[:, :, 0, 0]

        # Squeezing to embed
        out_left = self.squeezer(out_left_)  # Out: embed   #self.layernorm_left_after_pool(out_left))
        out_right = self.squeezer(out_right_)  # self.layernorm_right_after_pool(out_right))

        # Calulate attention

        self.channel_attn_forward(out_left, out_right)  # In embed Out 3*embed
        self.ln_forward()  # In 3*embed Out 3*embed

        self.self_attn_forward(self.out_left, self.out_right)  # In 3*embed  Out 1.5*embed

        out_left = torch.cat([self.out_left, self.left_sl_score], dim=1)  # , right_sl_score
        out_right = torch.cat([self.out_right, self.right_sl_score], dim=1)  # , left_sl_score
        self.out_feature = torch.cat([out_left, out_right], dim=0)  # -> B

        # if not self.separate:
        self.point_header_forward(out_left_, out_right_)
        self.points = self.points.detach()

        self.header_forward()
        x_left_bk.append(x_left)
        x_right_bk.append(x_right)

        if return_features == True:
            return (self.logit, self.grade, self.points), ((out_left_, out_right_), (x_left_bk, x_right_bk))

        return self.logit, self.grade, self.points


from sklearn.metrics import roc_auc_score, average_precision_score

import albumentations as A
from albumentations.pytorch import ToTensorV2

from datasets import NewDataset, NewLoader, new_split_data, read_img, get_datasets
from models.models import BasicXrayNet, SBOnet, SEnet, PHYSBOnet, BilaterPHResnet50, MyModel2

import torch


def get_test_model(opt, get_layer_feature=False):
    opt_model = opt["model"]

    backbone_name = opt_model["backbone_name"]
    model_name = opt_model["model_name"]

    in_chans = opt_model["in_chans"]
    drop_out = opt_model["drop_out"]
    backbone_drop_out = opt_model["backbone_drop_out"]
    embed_dim = opt_model["embed_dim"]
    classes = opt_model["classes"]

    model = BasicXrayNet_test(model_name=backbone_name,
                              pretrained=True, separate=opt["separate_model"],
                              drop_out=drop_out, backbone_drop_out=backbone_drop_out, in_chans=in_chans,
                              get_layer_feature=get_layer_feature)

    model.cuda()

    cls_model = {"model": model}

    trainer = kaggleTester(opt=opt, cls_model=cls_model, lmk_model=None, get_layer_feature=get_layer_feature)
    return trainer


def get_tester(opt, get_layer_feature=False):
    seed_everything(seed=opt["seed"])

    trainer = get_test_model(opt, get_layer_feature=get_layer_feature)

    score_funcs = {'auc': roc_auc_score,
                   'ap': average_precision_score}

    return trainer, score_funcs


import matplotlib.pyplot as plt
import cv2


def process_cam(img):
    size_upsample = (224, 224)
    img = img - np.min(img)
    img = img / np.max(img)

    img = np.uint8(255 * img)

    return np.asarray(cv2.resize(img, size_upsample))  # 224, 224


def returnCAM(feature_conv, weight_softmax=None, class_idx=None):
    # generate the class activation maps upsample to 256x256

    output_cam = {}

    for feature_layer in feature_conv:
        bz, nc, h, w = feature_layer.shape  # feature_layer: (32, 192, 56, 56)
        feature_layer = feature_layer.cpu().numpy()

        for idx in range(bz):

            cam = feature_layer[idx]

            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)

            cams = process_cam(cam_img[0])  # np.mean(cam_img, 0)
            #             cams = process_cam(cam_img[1])
            cams2 = process_cam(cam_img[2])
            cams3 = process_cam(cam_img[3])
            cams4 = process_cam(cam_img[4])
            cams5 = process_cam(cam_img[5])
            cams6 = process_cam(cam_img[6])
            cams7 = process_cam(cam_img[7])
            cams8 = process_cam(cam_img[8])
            cams9 = process_cam(cam_img[9])
            cams1 = process_cam(np.mean(cam_img, 0))

            if idx not in output_cam:
                output_cam[idx] = [cams, cams1, cams2, cams3, cams4, cams5, cams6, cams7, cams8, cams9]
            else:
                output_cam[idx].extend([cams, cams1, cams2, cams3, cams4, cams5, cams6, cams7, cams8, cams9])

    return output_cam


def show(left_title, right_title, left_image, right_image, left_cams, right_cams, fontsize=10):
    fig = plt.figure(figsize=(12, 16))
    grid = plt.GridSpec(4, 6, hspace=0.2, wspace=0.2)
    #     main_ax =
    #     y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
    #     x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

    ax1 = fig.add_subplot(grid[:3, :3])
    ax2 = fig.add_subplot(grid[:3, 3:6])
    ax1.axis('off')
    ax2.axis('off')

    ax1.imshow(left_image)

    ax1.set_title("left view: " + left_title, fontsize=fontsize)  # , fontsize=fontsize

    ax2.imshow(right_image)
    #     ax2.locator_params(nbins=3)
    ax2.set_title("right view: " + right_title, fontsize=fontsize)

    for i in range(3):
        ax = fig.add_subplot(grid[2, i])
        ax.axis('off')
        ax.imshow(left_cams[i])

    for i in range(3):
        ax = fig.add_subplot(grid[2, 3 + i])
        ax.axis('off')
        ax.imshow(right_cams[i])

    #     fig.tight_layout()
    plt.show()


from datasets.dataloaders import *

TRANSFORM_IMG = A.Compose([
    A.Resize(310, 310, always_apply=True),
    A.CenterCrop(224, 224, always_apply=True),
    ToTensorV2(),
])

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _run_test_epoch(opt, cls_models, lmk_models, data_loader,
                    results, score_funcs,
                    is_bilateral=False,
                    prefix="",
                    desc=None, tta=[0], cam=False
                    ):
    [model] = cls_models
    model.eval()

    running_loss = []
    y_true = []
    y_pred = []

    tta_inferencer = tta_inference()

    os.makedirs("cam_results", exist_ok=True)
    total_n = 0

    inputs_dict = data_loader

    labels_left, labels_right = torch.split(inputs_dict["labels"], split_size_or_sections=1, dim=1)

    labels = torch.cat([labels_left, labels_right], dim=0)

    if not is_bilateral:
        inputs_left, inputs_right = inputs_dict["images"]
        inputs = torch.cat([inputs_left, inputs_right], dim=0)
    else:
        inputs = inputs_dict["images"]

    if len(labels.shape) == 1:
        labels = labels.view(labels.shape[0], -1)  #
    labels = labels.float()

    device = 'cuda'
    inputs = moveTo(inputs, device)
    labels = moveTo(labels, device)

    with torch.no_grad():
        outputs_list = []
        for i in tta:
            inputs_tta = tta_inferencer(inputs, i)

            if cam:
                outputs, features = model(inputs_tta, None, return_features=cam)
                # features : ((out_left_, out_right_), (x_left, x_right), (x_left_bk, x_right_bk))

                Left_CAMs = returnCAM(features[1][0])  # x_left
                Right_CAMs = returnCAM(features[1][1])  # x_rightX

                # n in B
                for n, m in enumerate(Left_CAMs.keys()):  # Patients

                    left_origin = np.repeat(np.expand_dims(inputs_tta[0][n, 0].cpu().numpy(), -1), 3, axis=2)
                    right_origin = np.repeat(np.expand_dims(inputs_tta[1][n, 0].cpu().numpy(), -1), 3, axis=2)

                    left_label = labels[n][0].cpu().numpy()

                    right_label = labels[inputs[0].shape[0] + n][0].cpu().numpy()

                    pred_logit, pred_grade = outputs[0], outputs[1]  # logit, self.grade, self.point
                    left_pred_logit = pred_logit[n][0].cpu().numpy()
                    left_pred_grade = pred_grade[n][0].cpu().numpy()

                    right_pred_logit = pred_logit[inputs[0].shape[0] + n][0].cpu().numpy()
                    right_pred_grade = pred_grade[inputs[0].shape[0] + n][0].cpu().numpy()

                    left_title = f'label{left_label}_pred{np.round(sigmoid(left_pred_logit), 3)}'
                    right_title = f'label{right_label}_pred{np.round(sigmoid(right_pred_logit), 3)}'

                    left_results, right_results = [], []

                    for camid, _ in enumerate(Left_CAMs[n]):
                        h1 = cv2.applyColorMap(Left_CAMs[n][camid], cv2.COLORMAP_JET)

                        left_results.append(np.uint8(h1 * 0.5) + np.uint8(255 * 0.5 * left_origin))

                    for camid, _ in enumerate(Right_CAMs[n]):
                        h1 = cv2.applyColorMap(Right_CAMs[n][camid], cv2.COLORMAP_JET)
                        right_results.append(np.uint8(h1 * 0.5) + np.uint8(255 * 0.5 * right_origin))

                    print("right_results:", len(right_results))

                    Choose = np.array([21, 32, 47])
                    left = np.asarray(left_results)[Choose]
                    right = np.asarray(right_results)[Choose]

                    show(left_title, right_title, left_origin, right_origin,
                         left, right)

                outputs_list.append(outputs[0].detach().cpu().numpy())

    outputs = np.mean(outputs_list, 0)

    # loss = loss_func([outputs, None], grades, labels, None)
    # running_loss.append(loss.item())

    if len(score_funcs) > 0 and isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
        # Convert the outputs to the probability
        y_hat = sigmoid(outputs)

        y_true.extend(labels.tolist())
        y_pred.extend(y_hat)

    # End training epoch

    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    results[f"{prefix} loss"].append(np.mean(running_loss))
    for name, score_func in score_funcs.items():

        try:
            results[f"{prefix} {name}"].append(score_func(y_true.flatten(), y_pred.flatten()))
        except:
            results[f"{prefix} {name}"].append(float("NaN"))

    gc.collect()



