import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os


class Hardswish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.relu6(x + 3.0) / 6.0


class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, act=True):
        super(Conv, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g),
            nn.BatchNorm2d(c2),
            Hardswish() if act else nn.Identity()
        )

    def forward(self, x):
        return self.convs(x)


class ResBlock(nn.Module):
    def __init__(self, c1, n=1):
        super(ResBlock, self).__init__()
        self.module_list = nn.ModuleList()
        c2 = c1 // 2
        for _ in range(n):
            resblock_one = nn.Sequential(
                Conv(c1, c2, k=1),
                Conv(c2, c1, k=3, p=1)
            )
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            x = module(x) + x
        return x


# Copy from yolov5
class Focus(nn.Module):
    """
        Focus module proposed by yolov5.
    """
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, p=0, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k=k, s=s, p=p, g=g, act=act)

    def forward(self, x):  # x(B, C, H, W) -> y(B, 4C, H/2, W/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


# Copy from yolov5
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k=1)
        self.cv2 = Conv(c_, c2, k=3, p=1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k=1)
        self.cv2 = nn.Conv2d(c1, c_, kernel_size=1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, kernel_size=1, bias=False)
        self.cv4 = Conv(2 * c_, c2, k=1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class CSPDarknet_large(nn.Module):
    """
    CSPDarknet_large.
    """
    def __init__(self, cfg='l', num_classes=1000):
        super(CSPDarknet_large, self).__init__()
            
        self.layer_1 = nn.Sequential(
            Focus(c1=3, c2=64, k=3, p=1),           
            ResBlock(c1=64, n=1)                    # P1/2
        )
        self.layer_2 = nn.Sequential(
            Conv(c1=64, c2=128, k=3, p=1, s=2),     
            BottleneckCSP(c1=128, c2=128, n=2)      # P2/4
        )
        self.layer_3 = nn.Sequential(
            Conv(c1=128, c2=256, k=3, p=1, s=2),    
            BottleneckCSP(c1=256, c2=256, n=8)      # P3/8
        )
        self.layer_4 = nn.Sequential(
            Conv(c1=256, c2=512, k=3, p=1, s=2),    
            BottleneckCSP(c1=512, c2=512, n=8)      # P4/16
        )
        self.layer_5 = nn.Sequential(
            Conv(c1=512, c2=1024, k=3, p=1, s=2),   
            BottleneckCSP(c1=1024, c2=1024, n=4)    # P5/32
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x, targets=None):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class CSPDarknet_medium(nn.Module):
    def __init__(self, cfg='l', num_classes=1000):
        super(CSPDarknet_medium, self).__init__()
            
        self.layer_1 = nn.Sequential(
            Focus(c1=3, c2=64, k=3, p=1),           
            ResBlock(c1=64, n=1)                    # P1/2
        )
        self.layer_2 = nn.Sequential(
            Conv(c1=64, c2=128, k=3, p=1, s=2),     
            BottleneckCSP(c1=128, c2=128, n=2)      # P2/4
        )
        self.layer_3 = nn.Sequential(
            Conv(c1=128, c2=256, k=3, p=1, s=2),    
            BottleneckCSP(c1=256, c2=256, n=2)      # P3/8
        )
        self.layer_4 = nn.Sequential(
            Conv(c1=256, c2=512, k=3, p=1, s=2),    
            BottleneckCSP(c1=512, c2=512, n=2)      # P4/16
        )
        self.layer_5 = nn.Sequential(
            Conv(c1=512, c2=1024, k=3, p=1, s=2),   
            BottleneckCSP(c1=1024, c2=1024, n=2)    # P5/32
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x, targets=None):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class CSPDarknet_small(nn.Module):
    def __init__(self, cfg='l', num_classes=1000):
        super(CSPDarknet_small, self).__init__()
            
        self.layer_1 = nn.Sequential(
            Focus(c1=3, c2=64, k=3, p=1),           
            ResBlock(c1=64, n=1)                    # P1/2
        )
        self.layer_2 = nn.Sequential(
            Conv(c1=64, c2=128, k=3, p=1, s=2),     
            BottleneckCSP(c1=128, c2=128, n=1)      # P2/4
        )
        self.layer_3 = nn.Sequential(
            Conv(c1=128, c2=256, k=3, p=1, s=2),    
            BottleneckCSP(c1=256, c2=256, n=1)      # P3/8
        )
        self.layer_4 = nn.Sequential(
            Conv(c1=256, c2=512, k=3, p=1, s=2),    
            BottleneckCSP(c1=512, c2=512, n=1)      # P4/16
        )
        self.layer_5 = nn.Sequential(
            Conv(c1=512, c2=1024, k=3, p=1, s=2),   
            BottleneckCSP(c1=1024, c2=1024, n=1)    # P5/32
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x, targets=None):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class CSPDarknet_slim(nn.Module):
    def __init__(self, cfg='l', num_classes=1000):
        super(CSPDarknet_slim, self).__init__()
            
        self.layer_1 = nn.Sequential(
            Focus(c1=3, c2=32, k=3, p=1),           
            ResBlock(c1=32, n=1)                    # P1/2
        )
        self.layer_2 = nn.Sequential(
            Conv(c1=32, c2=64, k=3, p=1, s=2),     
            BottleneckCSP(c1=64, c2=64, n=1)      # P2/4
        )
        self.layer_3 = nn.Sequential(
            Conv(c1=64, c2=128, k=3, p=1, s=2),    
            BottleneckCSP(c1=128, c2=128, n=1)      # P3/8
        )
        self.layer_4 = nn.Sequential(
            Conv(c1=128, c2=256, k=3, p=1, s=2),    
            BottleneckCSP(c1=256, c2=256, n=1)      # P4/16
        )
        self.layer_5 = nn.Sequential(
            Conv(c1=256, c2=512, k=3, p=1, s=2),   
            BottleneckCSP(c1=512, c2=512, n=1)    # P5/32
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, targets=None):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class CSPDarknet_tiny(nn.Module):
    def __init__(self, cfg='l', num_classes=1000):
        super(CSPDarknet_tiny, self).__init__()
            
        self.layer_1 = nn.Sequential(
            Focus(c1=3, c2=32, k=3, p=1),           
            ResBlock(c1=32, n=1)                    # P1/2
        )
        self.layer_2 = nn.Sequential(
            nn.MaxPool2d((2, 2), 2),     
            BottleneckCSP(c1=32, c2=64, n=1)      # P2/4
        )
        self.layer_3 = nn.Sequential(
            nn.MaxPool2d((2, 2), 2),     
            BottleneckCSP(c1=64, c2=128, n=1)      # P3/8
        )
        self.layer_4 = nn.Sequential(
            nn.MaxPool2d((2, 2), 2),     
            BottleneckCSP(c1=128, c2=256, n=1)      # P4/16
        )
        self.layer_5 = nn.Sequential(
            nn.MaxPool2d((2, 2), 2),     
            BottleneckCSP(c1=256, c2=512, n=1)    # P5/32
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, targets=None):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def cspdarknet_large(pretrained=False, **kwargs):
    """Constructs a CSPDarknet_large model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = CSPDarknet_large()
    if pretrained:
        path_to_dir = os.path.dirname(os.path.abspath(__file__))
        print('Loading the cspdarknet_large ...')
        model.load_state_dict(torch.load(path_to_dir + '/weights/cspdarknet_large_75.42.pth', map_location='cuda'), strict=False)
    return model


def cspdarknet_medium(pretrained=False, **kwargs):
    """Constructs a CSPDarknet_medium model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = CSPDarknet_medium()
    if pretrained:
        path_to_dir = os.path.dirname(os.path.abspath(__file__))
        print('Loading the cspdarknet_medium ...')
        model.load_state_dict(torch.load(path_to_dir + '/weights/cspdarknet_medium_75.42.pth', map_location='cuda'), strict=False)
    return model


def cspdarknet_small(pretrained=False, **kwargs):
    """Constructs a CSPDarknet_small model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = CSPDarknet_small()
    if pretrained:
        path_to_dir = os.path.dirname(os.path.abspath(__file__))
        print('Loading the cspdarknet_small ...')
        model.load_state_dict(torch.load(path_to_dir + '/weights/cspdarknet_small_75.42.pth', map_location='cuda'), strict=False)
    return model


def cspdarknet_slim(pretrained=False, **kwargs):
    """Constructs a CSPDarknet_slim model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = CSPDarknet_slim()
    if pretrained:
        path_to_dir = os.path.dirname(os.path.abspath(__file__))
        print('Loading the cspdarknet_slim ...')
        model.load_state_dict(torch.load(path_to_dir + '/weights/cspdarknet_slim_75.42.pth', map_location='cuda'), strict=False)
    return model


def cspdarknet_tiny(pretrained=False, **kwargs):
    """Constructs a CSPDarknet_tiny model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = CSPDarknet_tiny()
    if pretrained:
        path_to_dir = os.path.dirname(os.path.abspath(__file__))
        print('Loading the cspdarknet_tiny ...')
        model.load_state_dict(torch.load(path_to_dir + '/weights/cspdarknet_tiny_75.42.pth', map_location='cuda'), strict=False)
    return model
