import torch
import torch.nn as nn



class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, act=True, bias=False):
        super(Conv, self).__init__()
        if act:
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=bias),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True)
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=bias),
                nn.BatchNorm2d(c2)
            )

    def forward(self, x):
        return self.convs(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c, k=7, p=3, e=0.5, act='relu'):
        super(Bottleneck, self).__init__()
        c_ = int(c * e)
        self.branch = nn.Sequential(
            Conv(c, c_, k=1, act=act),
            Conv(c_, c_, k=k, p=p, act=act),
            Conv(c_, c, k=1, act=act)
        )

    def forward(self, x):
        return x + self.branch(x)


class ConvMixer(nn.Module):
    def __init__(self, dim=512, depth=1, num_classes=1000):
        super().__init__()
        self.patch_embedding = Conv(3, dim, k=32, s=32)
        self.blocks = nn.Sequential(*[Bottleneck(c=dim, k=7, p=3) for _ in range(depth)])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.blocks(x)
        x = self.avgpool(x).view(x.size(0), -1)
        x = self.fc(x)

        return x


def convmixer_384_32(pretrained=False, **kwargs):
    model = ConvMixer(dim=384, depth=32)
    return model

def convmixer_768_32(pretrained=False, **kwargs):
    model = ConvMixer(dim=768, depth=32)
    return model

def convmixer_1024_20(pretrained=False, **kwargs):
    model = ConvMixer(dim=1024, depth=20)
    return model

def convmixer_1536_20(pretrained=False, **kwargs):
    model = ConvMixer(dim=1536, depth=20)
    return model


if __name__ == '__main__':
    x = torch.randn(2, 3, 640, 640)
    model = convmixer_384_32()
    y = model(x)
