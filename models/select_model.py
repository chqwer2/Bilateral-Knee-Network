import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from models.models import BasicXrayNet, SBOnet, SEnet, PHYSBOnet,  BilaterPHResnet50, MyModel2
from models.loss import FocalLossV2,  myBCE,  Focal_Reg, LMK_Reg
from models.optimizer import CustomWarmupStaticDecayLR
from models.trainer import Trainer

from models.models import Main_Auxilary_Model, Left_and_Right_Model, LMKXrayNet


def get_model(opt, get_layer_feature=False):
    opt_model = opt["model"]
    
    backbone_name = opt_model["backbone_name"]
    model_name = opt_model["model_name"]
    
    in_chans = opt_model["in_chans"]
    drop_out = opt_model["drop_out"]
    backbone_drop_out = opt_model["backbone_drop_out"]
    embed_dim = opt_model["embed_dim"]
    classes = opt_model["classes"]
    
    if model_name ==  "BasicXrayNet":
        model = BasicXrayNet(model_name=backbone_name,  
                        pretrained=True, separate=opt["separate_model"],
                        drop_out=drop_out, backbone_drop_out=backbone_drop_out, in_chans=in_chans, get_layer_feature=get_layer_feature)

        lmk_model = LMKXrayNet(model_name=backbone_name,
                             pretrained=True,
                             drop_out=drop_out, backbone_drop_out=backbone_drop_out, in_chans=in_chans)



    elif model_name ==  "Main_Auxilary_Model":
        #   embed_dim = 48, backbone_embed=1024, classes=1)
        model = Main_Auxilary_Model(encoder_name=backbone_name,  
                        pretrain=True, embed_dim=embed_dim, classes=classes,
                        drop_rate=drop_out, backbone_drop_out=backbone_drop_out, in_chans=in_chans)
    # 1/3
    model.cuda()
    lmk_model.cuda()


    optimizer = torch.optim.AdamW

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if 'backbone' in n],
            'lr': opt["lr"]/3},
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n],
            'lr': opt["lr"]}
    ]

    lmk_optimizer = optimizer(params=lmk_model.parameters(),
                          lr=opt["lr"], weight_decay=opt["wd"])

    optimizer = optimizer(params=optimizer_grouped_parameters,
                                lr=opt["lr"],  weight_decay=opt["wd"])



    scheduler = CustomWarmupStaticDecayLR(optimizer=optimizer, 
                                        epochs_warmup=5,    # Warmup
                                        epochs_static=100, 
                                        epochs_decay=1,
                                        decay_factor=0.9)     # decay_factor=
    lmk_scheduler = CustomWarmupStaticDecayLR(optimizer=lmk_optimizer,
                                          epochs_warmup=5,  # Warmup
                                          epochs_static=100,
                                          epochs_decay=1,
                                          decay_factor=0.9)  # decay_factor=


    # Loss Function
    # criterion = FocalLossV2()
    criterion = Focal_Reg(opt=opt)
    lmk_criterion = LMK_Reg()

    cls_model = {"model":model, "optimizer":optimizer, "loss_funcs":criterion,"lr_schedule":scheduler}
    lmk_model = {"model":lmk_model, "optimizer":lmk_optimizer, "loss_funcs":lmk_criterion,"lr_schedule":lmk_scheduler}

    trainer = Trainer(opt=opt, cls_model=cls_model, lmk_model=lmk_model)
    
    return trainer

