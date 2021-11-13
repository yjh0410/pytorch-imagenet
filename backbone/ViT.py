import torch
import numpy as np
import torch.nn as nn
from .transformer import build_transformer
import math


class ViT(nn.Module):
    def __init__(self, 
                 args,
                 num_classes=1000):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = 48

        # backbone
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(3, args["hidden_dim"], kernel_size=32, stride=32, bias=False),
            nn.BatchNorm2d(args["hidden_dim"]),
            nn.ReLU(inplace=True),
            nn.Conv2d(args["hidden_dim"], args["hidden_dim"], kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(args["hidden_dim"]),
            nn.ReLU(inplace=True)
        ) 
        
        # transformer
        self.transformer = build_transformer(args)

        # object query
        self.query_embed = nn.Embedding(self.num_queries, args["hidden_dim"])
        # position embedding
        self.pos_embed = self.position_embedding(hs=args["img_size"]//32, ws=args["img_size"]//32, 
                                                    num_pos_feats=args["hidden_dim"]//2, normalize=True)

        # transformer
        self.transformer = build_transformer(args)

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
        
        return pos.cuda()


    def forward(self, x):
        # patch embedding
        x = self.patch_embedding(x)

        # transformer
        x = self.transformer(x, self.query_embed.weight, self.pos_embed)[0]

        # output: [M, B, N, C] where M = num_decoder since we use all intermediate outputs of decoder
        # [M, B, N, C] -> [B, N, C] -> [B, C]
        x = x[-1].mean(-2)
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
        "num_decoders": 6,
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
        "num_decoders": 12,
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
        "num_decoders": 12,
        "pre_norm": False,
    }
    model = ViT(args=args)
    return model

