import torch
import numpy as np
import torch.nn as nn
from .transformer import build_encoder
import math


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, p=0, s=1, d=1, g=1, act=True, bias=False):
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


class ViT(nn.Module):
    def __init__(self, 
                 args,
                 num_classes=1000):
        super().__init__()
        self.num_classes = num_classes
        # position embedding
        self.pos_embed = self.position_embedding(hs=args["img_size"]//16, ws=args["img_size"]//16, 
                                                    num_pos_feats=args["hidden_dim"]//2, normalize=True)

        # patch embedding
        self.patch_embedding = nn.Sequential(
            Conv(3, args["hidden_dim"], k=16, s=16, act=False),
            Conv(args["hidden_dim"], args["hidden_dim"], k=7, p=3, s=1)
        ) 
        
        # transformer encoder
        self.encoder = build_encoder(args)

        # output
        self.fc = nn.Linear(args["hidden_dim"], num_classes)
    
        
    # Position Embedding
    def position_embedding(self, hs=7, ws=7, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        # generate xy coord mat
        # y_embed = [[0, 0, 0, ...], [1, 1, 1, ...]...]
        # x_embed = [[0, 1, 2, ...], [0, 1, 2, ...]...]
        y_embed, x_embed = torch.meshgrid([torch.arange(1, hs+1, dtype=torch.float32), 
                                        torch.arange(1, ws+1, dtype=torch.float32)])
        if normalize:
            eps = 1e-6
            y_embed = y_embed / (hs + eps) * scale
            x_embed = x_embed / (ws + eps) * scale
    
        # [H, W] -> [1, H, W]
        y_embed = y_embed[None, :, :]
        x_embed = x_embed[None, :, :]


        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        # torch.div(a, b, rounding_mode='floor') == (a // b)
        dim_t_ = torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats
        dim_t = temperature ** (2 * dim_t_)

        pos_x = torch.div(x_embed[:, :, :, None], dim_t)
        pos_y = torch.div(y_embed[:, :, :, None], dim_t)
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # [B, d, H, W]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        return pos


    def forward(self, x):
        # patch embedding
        x = self.patch_embedding(x)
        x = x.flatten(2).permute(2, 0, 1)
        pos_embed = self.pos_embed.flatten(2).permute(2, 0, 1).to(x.device)

        # transformer
        x = self.encoder(src=x, pos=pos_embed)
        # [N, B, C] -> [B, N, C]
        x = x.permute(1, 0, 2)

        # [B, N, C] -> [B, C]
        x = x.mean(1)
        x = self.fc(x)

        return x


def vit_256_6(pretrained=False, **kwargs):
    args={
        "img_size": 224,
        "hidden_dim": 256,
        "dropout": 0.1,
        "num_heads": 8,
        "mlp_dim": 2048,
        "num_encoders": 6,
        "pre_norm": False,
    }
    model = ViT(args=args)
    return model

def vit_256_12(pretrained=False, **kwargs):
    args={
        "img_size": 224,
        "hidden_dim": 256,
        "dropout": 0.1,
        "num_heads": 8,
        "mlp_dim": 2048,
        "num_encoders": 12,
        "pre_norm": False,
    }
    model = ViT(args=args)
    return model

def vit_512_12(pretrained=False, **kwargs):
    args={
        "img_size": 224,
        "hidden_dim": 512,
        "dropout": 0.1,
        "num_heads": 8,
        "mlp_dim": 2048,
        "num_encoders": 12,
        "pre_norm": False,
    }
    model = ViT(args=args)
    return model


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = vit_256_6()
    y = model(x)
