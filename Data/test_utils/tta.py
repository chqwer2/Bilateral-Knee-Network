import albumentations as A

import torch.nn as nn
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T

class tta_inference(nn.Module):
    def __init__(self):
        super(tta_inference, self).__init__()
        self.h_flip_ = A.HorizontalFlip(p=1)
        self.rotate_ = A.RandomRotate90(p=1)


        self.crop = T.CenterCrop(size=224)
        self.scale1 = T.Resize(size=246, interpolation=3)
        self.scale2 = T.Resize(size=268, interpolation=3)
        self.scale3 = T.Resize(size=290, interpolation=3)


    def forward(self, input, aug=0):
        # enter xy

        # x for scale
        # y for flip for eight
        # 0 is original

        scale = aug//10
        flip = aug%10


        for i, img in enumerate(input):
            # B, C, H, W

            if scale == 1:
                img = self.crop(self.scale1(img))
            elif scale == 2:
                img = self.crop(self.scale2(img))
            elif scale == 3:
                img = self.crop(self.scale3(img))


            if flip == 0:
                 pass
            elif flip == 1:
                img = F.hflip(img)
            elif flip == 2:
                img = F.rotate(img, angle=90)
            elif flip == 3:
                img = F.hflip(F.rotate(img, angle=90))
            elif flip == 4:
                img = F.rotate(img, angle=180)
            elif flip == 5:
                img = F.hflip(F.rotate(img, angle=180))
            elif flip == 6:
                img = F.rotate(img, angle=270)
            elif flip == 7:
                img = F.hflip(F.rotate(img, angle=270))

            input[i] = img #torch.from_array(img.copy()).cuda()


        return input