'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
try:
    from layers import PHConv, SELayer, GeM
except:
    from .layers import PHConv, SELayer, GeM
# from utils.utils import load_weights

class BasicBlock(nn.Module):
    """
    Not change the image shape by default.
    
    return: (B, planes, (H-1)//stride+1, (W-1)//stride+1)
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    """
    Double the planes by default
    
    return: (B, planes*2, (H-1)//stride+1, (W-1)//stride+1)
    
    """
    expansion = 2

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, channels=4, num_classes=10, gap_output=False, before_gap_output=False, visualize=False):
        super(ResNet, self).__init__()
        self.block = block
        self.num_blocks = num_blocks
        self.in_planes = 64
        self.gap_output = gap_output
        self.before_gap_out = before_gap_output
        self.visualize = visualize

        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.layer5 = None
        self.layer6 = None
        if not gap_output and not before_gap_output:
            self.linear = nn.Linear(512*block.expansion, num_classes)
    
    def add_top_blocks(self, num_classes=1):
        self.layer5 = self._make_layer(Bottleneck, 512, 2, stride=2)
        self.layer6 = self._make_layer(Bottleneck, 512, 2, stride=2)
        
        if not self.gap_output and not self.before_gap_out:
            self.linear = nn.Linear(1024, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out4 = self.layer4(out) # shape(B, C, H//2^3, W//2^3)

        if self.before_gap_out:
            return out4
        
        if self.layer5:
            out5 = self.layer5(out4)
            out6 = self.layer6(out5)

        n, c, _, _ = out6.size()
        out = out6.view(n, c, -1).mean(-1) # shape(B, C)

        if self.gap_output:
            return out

        out = self.linear(out)
        if self.visualize:
            return out, out4, out6
        return out

class Encoder(nn.Module):
    # (B, C, H, W) -> (B, 128, H//2, W//2)
    def __init__(self, channels):
        super(Encoder, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        return out

class SharedBottleneck(nn.Module):
    def __init__(self, in_planes):
        super(SharedBottleneck, self).__init__()
        self.in_planes = in_planes

        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.layer5 = self._make_layer(Bottleneck, 512, 2, stride=2)
        self.layer6 = self._make_layer(Bottleneck, 512, 2, stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer3(x)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        n, c, _, _ = out.size()
        out = out.view(n, c, -1).mean(-1)       
        return out

class Classifier(nn.Module):
    def __init__(self, num_classes, in_planes=512, visualize=False):
        super(Classifier, self).__init__()
        self.in_planes = in_planes
        self.visualize = visualize

        self.layer5 = self._make_layer(Bottleneck, 512, 2, stride=2)
        self.layer6 = self._make_layer(Bottleneck, 512, 2, stride=2)
        self.linear = nn.Linear(1024, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer5(x)
        feature_maps = self.layer6(out)

        n, c, _, _ = feature_maps.size()
        out = feature_maps.view(n, c, -1).mean(-1)
        out = self.linear(out)

        if self.visualize:
            return out, feature_maps

        return out


def ResNet18(num_classes=10, channels=4, gap_output=False, before_gap_output=False, visualize=False):
    return ResNet(BasicBlock, 
                  [2, 2, 2, 2], 
                  num_classes=num_classes, 
                  channels=channels, 
                  gap_output=gap_output, 
                  before_gap_output=before_gap_output,
                  visualize=visualize)

def ResNet50(num_classes=10, channels=4):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, channels=channels)


class PHBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, n=4, 
                 reduction=16):
        super(PHBasicBlock, self).__init__()
        self.conv1 = PHConv(n,
            in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = PHConv(n, planes, planes, kernel_size=3,
                               stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(channel=planes, reduction=reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                PHConv(n, in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride,),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

class PHBottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1, n=4):
        super(PHBottleneck, self).__init__()
        self.conv1 = PHConv(n, in_planes, planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = PHConv(n, planes, planes, kernel_size=3,
                               stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = PHConv(n, planes, self.expansion * planes, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                PHConv(n, in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PHEncoder(nn.Module):
    """
    Encoder branch in PHYSBOnet.
    """

    def __init__(self, channels, n):
        super(PHEncoder, self).__init__()
        self.in_planes = 64

        self.conv1 = PHConv(n, channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(PHBasicBlock, 64, 2, stride=1, n=n)
        self.layer2 = self._make_layer(PHBasicBlock, 128, 2, stride=2, n=n)

    def _make_layer(self, block, planes, num_blocks, stride, n):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, n))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        return out


class PHSharedBottleneck(nn.Module):
    """
    SharedBottleneck in PHYSBOnet.
    """

    def __init__(self, n, in_planes):
        super(PHSharedBottleneck, self).__init__()
        self.in_planes = in_planes
        
        self.layer3 = self._make_layer(PHBasicBlock, 256, 2, stride=2, n=n)
        self.layer4 = self._make_layer(PHBasicBlock, 512, 2, stride=2, n=n)
        self.layer5 = self._make_layer(PHBottleneck, 512, 2, stride=2, n=n)
        self.layer6 = self._make_layer(PHBottleneck, 512, 2, stride=2, n=n)
        self.pooling = GeM()

    def _make_layer(self, block, planes, num_blocks, stride, n):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, n))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer3(x)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        # n, c, _, _ = out.size()
        # out = out.view(n, c, -1).mean(-1)
        out = self.pooling(out).flatten(1) # shape(B, 1024)
        return out
    

class PHCResNet(nn.Module):
    """
    PHCResNet.

    Parameters:
    - before_gap_output: True to return the output before refiner blocks and gap
    - gap_output: True to rerurn the output after gap and before final linear layer
    """

    def __init__(self, block, num_blocks, channels=4, n=4, num_classes=10, 
                 before_gap_output=False, gap_output=False, visualize=False):
        super(PHCResNet, self).__init__()
        self.block = block
        self.num_blocks = num_blocks
        self.in_planes = 64
        self.n = n
        self.before_gap_out = before_gap_output
        self.gap_output = gap_output
        self.visualize = visualize

        self.conv1 = PHConv(n, channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, n=n)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, n=n)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, n=n)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, n=n)
        
        # Refiner blocks
        self.layer5 = None
        self.layer6 = None
        
        if not before_gap_output and not gap_output:
            self.linear = nn.Linear(512*block.expansion, num_classes)
        
    def add_top_blocks(self, num_classes=1):
        #print("Adding top blocks with n = ", self.n)
        self.layer5 = self._make_layer(PHBottleneck, 512, 2, stride=2, n=self.n)
        self.layer6 = self._make_layer(PHBottleneck, 512, 2, stride=2, n=self.n)
        
        if not self.before_gap_out and not self.gap_output:
            self.linear = nn.Linear(1024, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, n):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, n))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out4 = self.layer4(out)
        
        if self.before_gap_out:
            return out4
        
        if self.layer5:
            out5 = self.layer5(out4)
            out6 = self.layer6(out5)
        
        # global average pooling (GAP)
        n, c, _, _ = out6.size()
        out = out6.view(n, c, -1).mean(-1)
        
        if self.gap_output:
            return out

        out = self.linear(out)

        if self.visualize:
            # return the final output and activation maps at two different levels
            return out, out4, out6
        return out
    

def PHCResNet50(channels=2, n=2, num_classes=1, 
                before_gap_output=False, gap_output=True, visualize=False):
    net =  PHCResNet(PHBasicBlock, 
                    [3, 4, 6, 3], 
                    channels=channels, 
                    n=n, 
                    num_classes=num_classes, 
                    before_gap_output=before_gap_output, 
                    gap_output=gap_output,
                    visualize=visualize)
    net.add_top_blocks()
    
    return net