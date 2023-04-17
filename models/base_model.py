# -----------------------------------------------------------------------------------
# Bilateral-Knee-Network: "Expanding from Unilateral to Bilateral: a robust deep learning-based approach for radiographic osteoarthritis progression"
# Originally Written by Rui Yin, Modified by Hao Chen.
# -----------------------------------------------------------------------------------

import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import OrderedDict

try:

    from layers import GeM, SelfAttention
    from modules import (Encoder, SharedBottleneck, Classifier, ResNet18, ResNet18,
                         PHEncoder, PHSharedBottleneck, PHCResNet50)
except:
    from .layers import GeM, SelfAttention
    from .modules import (Encoder, SharedBottleneck, Classifier, ResNet18, ResNet18,
                          PHEncoder, PHSharedBottleneck, PHCResNet50)



class BaseModel(nn.Module):
    def __init__(self) -> None:
        super(BaseModel, self).__init__()
        pass

    # Missing
    # "backbone.stages_3.blocks.1.conv_dw.bias"
    # backbone.stages_3.blocks.2.mlp.fc2.weight", "backbone.stages_3.blocks.2.mlp.fc2.bias".
    # Unexpected
    # backbone.stages.3.blocks.2.mlp.fc2.bias", "backbone.head.norm.weight", "backbone.head.norm.bias".


    def create_backbone(self, model_name, pretrained, in_chans, backbone_drop_out, get_layer_feature=False):

        # Features Only
        if get_layer_feature:
            self.backbone = timm.create_model(model_name=self.model_name,
                                              pretrained=self.pretrained,
                                              in_chans=self.in_chans,
                                              num_classes=0,
                                              drop_rate=backbone_drop_out,
                                              global_pool="",
                                              features_only=True
                                              # out_indices=[2, 3, 4]  # None: with what error?
                                              )
        else:
            self.backbone = timm.create_model(model_name=self.model_name,
                                              pretrained=self.pretrained,
                                              in_chans=self.in_chans,
                                              num_classes=0,
                                              drop_rate=backbone_drop_out,
                                              global_pool="",
                                              )
        # print("baskbone:", self.backbone)

        if "swin" in self.model_name:
            self.backbone_embed = self.backbone.norm.normalized_shape[0]

        elif "convnext" in self.model_name:
            try:
                self.backbone_embed = self.backbone.head.norm.normalized_shape[0]
            except:

                self.backbone_embed = 1536


        elif "efficient" in self.model_name:
            self.backbone_embed = self.backbone.conv_head.out_channels  # 1792

        # print("baskbone:", self.backbone)
        # print("self.backbone_embed:", self.backbone_embed)
        self.global_pool = GeM(p_trainable=True)
        self.global_pool_momentum = copy.deepcopy(self.global_pool).eval()

    def update_gem(self, momentum=0.99):

        with torch.no_grad():
            self.global_pool_momentum.p.data = self.global_pool_momentum.p.data * momentum +\
                                          self.global_pool.p.data * (1-momentum)

            self.global_pool.p.data = self.global_pool_momentum.p.data


    def update_gem_(self, decay=0.95):
        model_params = OrderedDict(self.global_pool.named_parameters())
        shadow_params = OrderedDict(self.global_pool_momentum.named_parameters())

        for name, param in shadow_params.items():
            model_params[name].sub_(decay * (model_params[name] - param))

        for name, param in shadow_params.items():
            shadow_params[name].sub_((param - model_params[name]))

        model_buffers = OrderedDict(self.global_pool.named_buffers())
        shadow_buffers = OrderedDict(self.global_pool_momentum.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            shadow_buffers[name].copy_(buffer)

    def create_attention(self, backbone_embed, embed):

        # Resolution-wise
        self.attention_left_resolution = nn.Sequential(
            nn.Conv2d(backbone_embed, 1,
                      kernel_size=5, stride=1, bias=False, padding=2),
            nn.Sigmoid()
        )

        self.attention_right_resolution = nn.Sequential(
            nn.Conv2d(backbone_embed, 1,
                      kernel_size=5, stride=1, bias=False, padding=2),
            nn.Sigmoid()
        )

        # Channel-wise
        self.squeezer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(backbone_embed, embed)
        )

        self.attention_left = nn.Sequential(
            nn.Linear(2*embed, 2*embed),
            nn.Sigmoid()
        )
        self.attention_right = nn.Sequential(
            nn.Linear(2*embed, 2*embed),
            nn.Sigmoid()
        )

    def create_sl_attention(self, embed):
        # Self-attention module
        self.left_self_attn = SelfAttention(num_attention_heads=1,
                                            input_size=embed,
                                            hidden_size=embed//2, hidden_dropout_prob=0.0)
        self.right_attn_attn = SelfAttention(num_attention_heads=1,
                                             input_size=embed,
                                             hidden_size=embed//2, hidden_dropout_prob=0.0)

    def create_head(self, embed, backbone_embed, drop_out, num_classes=1):

        self.layernorm_left = nn.LayerNorm((embed,), eps=1e-06, elementwise_affine=True)
        self.layernorm_right = nn.LayerNorm((embed,), eps=1e-06, elementwise_affine=True)
        self.layernorm_sl_left = nn.LayerNorm((embed,), eps=1e-06, elementwise_affine=True)
        self.layernorm_sl_right = nn.LayerNorm((embed,), eps=1e-06, elementwise_affine=True)

        self.classifier = nn.Sequential(
            # nn.Dropout(),
            # nn.Linear(self.num_features, 512),
            nn.SiLU(), #nn.ReLU(),
            nn.Dropout(self.drop_out),  # Not sure if we need it
            nn.Linear(embed + embed // 2, 512),
            nn.SiLU(),#nn.ReLU(inplace=True),
            nn.Dropout(p=drop_out),
            nn.Linear(512, 31),   #

            nn.SiLU(),# nn.ReLU(),
            nn.Dropout(p=drop_out),
        )

        self.final_classifier = nn.Linear(64, num_classes)

        self.regressor = nn.Sequential(
            # nn.ReLU(),  # inplace=True
            nn.SiLU(),
            nn.Linear(embed + embed // 2, 512),

            # nn.ReLU(inplace=True),
            nn.SiLU(),
            # nn.Dropout(p=drop_out),
            # nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)  # backbone_embed + embed_dim
        )

        self.points_regressor = nn.Sequential(
            # nn.ReLU(),  # inplace=True
            nn.SiLU(),
            nn.Linear(backbone_embed, 512),

            # nn.ReLU(inplace=True),
            nn.SiLU(),
            # nn.Dropout(p=drop_out),
            # nn.Dropout(p=0.5),
            nn.Linear(512, 32)  # 16 * 2
        )



    def backbone_forward(self, x):
        x_left, x_right = x[0], x[1]
        x_left = self.backbone(x_left)  # Shape(B//2, 500)
        x_right = self.backbone(x_right)

        return x_left, x_right

    def res_attn_forward(self, x_left, x_right):
        x_left_score = self.attention_left_resolution(x_left)
        x_right_score = self.attention_right_resolution(x_right)
        x_left = x_left * x_left_score + x_left
        x_right = x_right * x_right_score + x_right

        return x_left, x_right

    def self_attn_forward(self, out_left, out_right):
        # Self Attentions
        left_q, left_k, left_v = self.left_self_attn.forward_feature(out_left)  # query_layer, key_layer, value_layer
        right_q, right_k, right_v = self.right_attn_attn.forward_feature(
            out_right)  # query_layer, key_layer, value_layer

        # embed // 2, query_layer, key_layer, value_layer
        self.left_sl_score = self.left_self_attn(right_q, left_k, left_v).squeeze(1)
        self.right_sl_score = self.right_attn_attn(left_q, right_k, right_v).squeeze(1)

    def channel_attn_forward(self, out_left, out_right):
        out = torch.cat([out_left, out_right], dim=1)  # Shape(B//2, 1000)
        left_score = self.attention_left(out)  # Shape(B//2, 1000)
        right_score = self.attention_right(out)  # Shape(B//2, 1000)

        self.out_left = torch.cat([left_score * out, out_left], dim=1)  # Shape(B//2, 1500)
        self.out_right = torch.cat([right_score * out, out_right], dim=1)  # Shape(B//2, 1500)

    def ln_forward(self):
        self.out_left = self.layernorm_left(self.out_left)
        self.out_right = self.layernorm_right(self.out_right)

    def header_forward(self):
        # print("self.out_feature:", self.out_feature.shape)

        cls_feature = self.classifier(self.out_feature)
        self.grade = self.regressor(self.out_feature)
        self.logit = self.final_classifier(torch.cat([cls_feature, self.grade, self.points], dim=1))

    def point_header_forward(self, out_left, out_right):
        out_left = self.points_regressor(out_left)
        out_right = self.points_regressor(out_right)

        self.points = torch.cat([out_left, out_right], dim=0)