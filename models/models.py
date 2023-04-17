from models.base_model import BaseModel
from models.base_model  import *
import torch.nn as nn

import cv2
import numpy as np


class BasicXrayNet_Feature(BaseModel):
    def __init__(self, model_name, pretrained=True, cross=False,
                 in_chans=1, drop_out=0.5, backbone_drop_out=0.1, num_classes=1,
                 embed=256, separate=True) -> None:
        super(BasicXrayNet, self).__init__()

        self.model_name = model_name
        self.pretrained = pretrained
        self.in_chans = in_chans
        self.drop_out = drop_out

        self.create_backbone(model_name, pretrained, in_chans, backbone_drop_out, get_layer_feature=True)
        self.separate = separate

        self.create_attention(self.backbone_embed, embed)

        final_embed = 3 * embed

        self.create_sl_attention(final_embed)  # final
        self.create_head(final_embed, self.backbone_embed, drop_out)
        self.embed = final_embed



    def forward(self, x):
        x_left_bk, x_right_bk = self.backbone_forward(x)

        return (x_left_bk, x_right_bk)



class BasicXrayNet(BaseModel):
    """
    The Bilateral Knee Neural Network (BikNet) for X-ray only
    Multi-modal BikNet will be coming soon
    """
    def __init__(self, model_name, pretrained=True, cross = False,
                 in_chans=1, drop_out=0.5, backbone_drop_out=0.1, num_classes=1,
                 embed = 256, separate=True, get_layer_feature=False) -> None:
        super(BasicXrayNet, self).__init__()

        self.model_name = model_name
        self.pretrained = pretrained
        self.in_chans = in_chans
        self.drop_out = drop_out
        
        self.create_backbone(model_name, pretrained, in_chans, backbone_drop_out, get_layer_feature=get_layer_feature)
        self.separate = separate

        self.create_attention(self.backbone_embed, embed)

        final_embed = 3 * embed

        self.create_sl_attention(final_embed)   # final
        self.create_head(final_embed, self.backbone_embed, drop_out)
        self.embed = final_embed

        self.get_layer_feature = get_layer_feature
        # self.layernorm_left_after_pool = nn.LayerNorm((backbone_embed,), eps=1e-06, elementwise_affine=True)
        # self.layernorm_right_after_pool = nn.LayerNorm((backbone_embed,), eps=1e-06, elementwise_affine=True)



    def forward(self, x, points, pointwise=False, return_features=False):

        x_left_bk, x_right_bk = self.backbone_forward(x)
        x_left, x_right = self.res_attn_forward(x_left_bk, x_right_bk)

        self.points = points

        # Pooling Features, [B//2, embed]
        out_left_ = self.global_pool( x_left  )[:, :, 0, 0]
        out_right_ = self.global_pool( x_right )[:, :, 0, 0]

        # Squeezing to embed
        out_left = self.squeezer(out_left_)  # Out: embed   #self.layernorm_left_after_pool(out_left))
        out_right = self.squeezer(out_right_)   #self.layernorm_right_after_pool(out_right))

        # Calulate attention

        self.channel_attn_forward(out_left, out_right)   # In embed Out 3*embed
        self.ln_forward()                                # In 3*embed Out 3*embed

        self.self_attn_forward(self.out_left, self.out_right)  # In 3*embed  Out 1.5*embed

        # print("out_left, self.left_sl_score:", out_left.shape, self.left_sl_score.shape)
        out_left = torch.cat([self.out_left, self.left_sl_score], dim=1)   # , right_sl_score
        out_right = torch.cat([self.out_right, self.right_sl_score], dim=1)  # , left_sl_score
        self.out_feature = torch.cat([out_left, out_right], dim=0)   # -> B

        # if not self.separate:
        self.point_header_forward(out_left_, out_right_)
        self.points = self.points.detach()

        self.header_forward()

        if return_features==True:
            return (self.logit, self.grade, self.points), ((out_left_, out_right_), (x_left, x_right), (x_left_bk, x_right_bk))


        return self.logit, self.grade, self.points


class LMKXrayNet(BaseModel):

    def __init__(self, model_name, pretrained=True, cross=False,
                 in_chans=1, drop_out=0.5, backbone_drop_out=0.1, num_classes=1,
                 embed=256) -> None:
        super(LMKXrayNet, self).__init__()

        self.model_name = model_name
        self.pretrained = pretrained
        self.in_chans = in_chans
        self.drop_out = drop_out

        self.create_backbone(model_name, pretrained, in_chans, backbone_drop_out)

        self.create_attention(self.backbone_embed, embed)

        final_embed = 3 * embed

        self.create_sl_attention(final_embed)  # final
        self.create_head(final_embed, self.backbone_embed, drop_out)
        self.embed = final_embed

        # self.layernorm_left_after_pool = nn.LayerNorm((backbone_embed,), eps=1e-06, elementwise_affine=True)
        # self.layernorm_right_after_pool = nn.LayerNorm((backbone_embed,), eps=1e-06, elementwise_affine=True)

    def forward(self, x, pointwise=False):
        x_left, x_right = self.backbone_forward(x)
        x_left, x_right = self.res_attn_forward(x_left, x_right)

        # Pooling Features, [B//2, embed]
        out_left_ = self.global_pool(x_left)[:, :, 0, 0]
        out_right_ = self.global_pool(x_right)[:, :, 0, 0]

        # if pointwise:
        self.point_header_forward(out_left_, out_right_)

        return self.points



class UnilateralKneeNet(nn.Module):
    """Unilateral CNN for Comparison"""
    def __init__(self, backbone_name, pretrained=True, in_chans=1, drop_out=0.5) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.in_chans = in_chans
        self.drop_out = drop_out
        
        self.backbone = timm.create_model(model_name=self.backbone_name, 
                                          pretrained=self.pretrained, 
                                          in_chans=self.in_chans )
        self.backbone = nn.Sequential(
            *(list(self.backbone.children())[:-1])
        )
        
        if self.backbone_name in ("resnet18", "resnet34"):
            num_embedding = 512
        elif self.backbone_name in ("densenet121"):
            num_embedding = 1024
        elif self.backbone_name in ("efficientnet_b0"):
            num_embedding = 1280
        elif self.backbone_name in ("resnet50", "resnext50_32x4d"):
            num_embedding = 2048
        else:
            msg = f"Unknown `num_embedding`. Checkout backbone"
            raise ValueError(msg)
        
        self.classifier = nn.Sequential(
            nn.Dropout(), 
            nn.Linear(num_embedding, 512), 
            nn.ReLU(inplace=True), 
            nn.Dropout(self.drop_out), 
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        out = self.backbone(x)
        out = self.classifier(out)

        return out



class Main_Auxilary_Model(nn.Module):
    def __init__(self, encoder_name, pretrain=False, in_chans=1, backbone_dr = 0,
                 drop_rate=0.5, embed_dim = 48, backbone_embed=1748, classes=1, single=True) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.single = single
        if single:
            f = 1
        else:
            f = 2

        self.backbone = timm.create_model(model_name=encoder_name, 
                                          pretrained=pretrain,
                                          in_chans=embed_dim*f ,   # 4
                                          num_classes=0,
                                          # num_classes=backbone_embed,  # , features_only=True
                                          drop_rate=backbone_dr,
                                          global_pool="",
                                          )

        self.global_pool = GeM(p_trainable=True)

        # self.encoder.global_pool = nn.Sequential(GeM(), nn.Flatten())
        
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))  # output_size=(12, 12)  (1, 1)
        
        self.left_encoder = nn.Sequential(nn.Conv2d(in_chans, embed_dim //2, 3, 1, 1),
                                                 # nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 # nn.Conv2d(embed_dim //2, embed_dim  //2, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim //2 , embed_dim - 1, 3, 1, 1),
                                                 # nn.LeakyReLU(negative_slope=0.2, inplace=True),  # TODO
                                                 )
        
        self.right_encoder = nn.Sequential(nn.Conv2d(in_chans, embed_dim //2, 3, 1, 1),
                                                 # nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 # nn.Conv2d(embed_dim //2, embed_dim  //2, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim //2 , embed_dim - 1, 3, 1, 1),
                                                 # nn.LeakyReLU(negative_slope=0.2, inplace=True),  # TODO
                                                 )


        # self.attention_left = nn.Sequential(    # embed_dim
        #     nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        #     nn.Linear(embed_dim, embed_dim),    # Linear
        #     nn.Sigmoid()
        # )
        # self.attention_right = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.Sigmoid()
        # )
        
        self.classifier = nn.Sequential(
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=drop_rate),
            # nn.Linear(backbone_embed + embed_dim , backbone_embed//2),
            # nn.ReLU(inplace=True),

            nn.Dropout(p=drop_rate), 
            # nn.Linear(backbone_embed//2 , classes),
            nn.Linear(1796, classes),   # backbone_embed + embed_dim

            # nn.ReLU(inplace=True), 
            # nn.Dropout(), 
            # nn.Linear(256, 1)
        )

        self.regressor = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(1796, classes),  # backbone_embed + embed_dim
        )
    
    def forward(self, x, main_view="left"):
        
        # x: [16, 2, 224, 224]
        if main_view == "left":
            x_left, x_right = x[:,0], x[:,1]  # [16, 224, 224]

        elif main_view == "right":
            x_left, x_right = x[:,1], x[:,0]
        
        x_left = x_left.unsqueeze(1)          # [16, 1, 224, 224]
        out_left = self.left_encoder(x_left)  # Shape(B//2, embed)  [1, 48, 224, 224]
        out_left = torch.cat([x_left, out_left], dim=1)

        if not self.single:
            x_right = x_right.unsqueeze(1)
            out_right = self.right_encoder(x_right)  # Shape(B//2, embed)
            out_right = torch.cat([x_right, out_right], dim=1)
            concat_feature = torch.cat([out_left, out_right], dim=1)  # Shape(B, 4*embed)  # B?
        else:
            concat_feature = out_left


        # Calulate attention
        # left_score = self.attention_left(out_left).unsqueeze(-1).unsqueeze(-1)     # Shape(B//2, embed)   [1, 48]
        # right_score = self.attention_right(out_right).unsqueeze(-1).unsqueeze(-1)  # Shape(B//2, embed)
        
        # print("out_left score:", left_score.shape)
                
        # out_left = torch.cat([left_score*out_left, out_left],     dim=1) # Shape(B, 2*embed)
        # out_right = torch.cat([right_score*out_right, out_right], dim=1) # Shape(B, 2*embed)


        # out = torch.cat([x_left, out_left], dim=1)
        out = self.global_pool(self.backbone(concat_feature))[:, :, 0, 0]   # [1, 1024]


        concat_left = self.GAP(out_left)[:, :, 0, 0]

        out = torch.cat([out, concat_left], dim=1) 
        # print("out:", out.shape)
        # Predict the left kneel
        return self.classifier(out), self.regressor(out)


class Left_and_Right_Model(nn.Module):
    def __init__(self, encoder_name, pretrain=True, in_chans=1, drop_rate=0.5, embed_dim = 48, backbone_embed=1024, classes=2) -> None:
        super().__init__()
        self.embed_dim = embed_dim 
        
        self.backbone = timm.create_model(model_name=encoder_name, 
                                         pretrained=pretrain, 
                                         in_chans=embed_dim*4, 
                                         num_classes=backbone_embed,  # 
                                         )
        
        # self.encoder.global_pool = nn.Sequential(GeM(), nn.Flatten())
        
        
        self.left_encoder = nn.Sequential(nn.Conv2d(in_chans, embed_dim //2, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim //2, embed_dim  //2, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim //2 , embed_dim - 1, 3, 1, 1),
                                                #  nn.LeakyReLU(negative_slope=0.2, inplace=True),  # TODO
                                                 )
        self.right_encoder = nn.Sequential(nn.Conv2d(in_chans, embed_dim //2, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim //2, embed_dim  //2, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim //2 , embed_dim - 1, 3, 1, 1),
                                                #  nn.LeakyReLU(negative_slope=0.2, inplace=True),  # TODO
                                                 )
        
        self.attention_left = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), 
            nn.Sigmoid()
        )
        self.attention_right = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), 
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.ReLU(), 
            nn.Dropout(p=drop_rate),
            nn.Linear(backbone_embed, backbone_embed//2), 
            nn.ReLU(inplace=True), 
            nn.Dropout(p=drop_rate), 
            nn.Linear(backbone_embed//2, 1), 
            
            # nn.ReLU(inplace=True), 
            # nn.Dropout(), 
            # nn.Linear(256, 1)
        )
    
    def forward(self, x):
        x_left, x_right = x[0], x[1]
        
        
        # Apply Encoder
        out_left = self.left_encoder(x_left)        # Shape(B//2, embed)
        out_right = self.right_encoder(x_right)     # Shape(B//2, embed)
        # out = torch.cat([out_left, out_right], dim=1) # Shape(B//2, 1000)

        # Calulate attention
        left_score = self.attention_left(out_left) # Shape(B//2, embed)
        right_score = self.attention_right(out_right) # Shape(B//2, embed)
        
                
        out_left = torch.cat([left_score*out_left, out_left], dim=1) # Shape(B, 2*embed)
        out_right = torch.cat([right_score*out_right, out_right], dim=1) # Shape(B, 2*embed)
        
        out = torch.cat([out_left, out_right], dim=1) # Shape(B, 4*embed)  # B?
        
        out = self.backbone(out)

        # Predict the left kneel
        return self.classifier(out)