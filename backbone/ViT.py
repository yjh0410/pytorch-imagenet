import torch
import torch.nn as nn
from .transformer import TransformerEncoder


class ViT(nn.Module):
    def __init__(self, 
                 img_size=224,
                 patch_size=16,
                 hidden_dim=256,
                 num_heads=8,
                 depth=6,
                 mlp_dim=2048,
                 dropout=0.,
                 embed_dropout=0.,
                 pool='cls',
                 num_classes=1000):
        super().__init__()
        self.num_classes = num_classes
        num_patches = (img_size // patch_size) ** 2
        # patch embedding
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7, padding=3, stride=1)
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.dropout = nn.Dropout(embed_dropout)
        
        # transformer encoder
        self.encoder = TransformerEncoder(dim=hidden_dim,
                                          depth=depth,
                                          heads=num_heads,
                                          dim_head=hidden_dim // num_heads,
                                          mlp_dim=mlp_dim,
                                          dropout=dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        # output
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        ) 
    
        
    def forward(self, x):
        # patch embedding
        x = self.patch_embedding(x)
        # [B, C, H, W] -> [B, N, C]
        x = x.flatten(2).permute(0, 2, 1)
        B, N, _ = x.shape

        cls_tokens = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding

        # transformer
        x = self.encoder(x)

        # [B, N, C] -> [B, C]
        x = x.mean(1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)

        return self.out(x)

def vit_t(pretrained=False, **kwargs):
    model = ViT(img_size=224,
                patch_size=16,
                hidden_dim=384,
                num_heads=8,
                depth=6,
                mlp_dim=2048,
                dropout=0.1,
                embed_dropout=0.1,
                pool='cls',
                num_classes=1000)

    return model


def vit_s(pretrained=False, **kwargs):
    model = ViT(img_size=224,
                patch_size=16,
                hidden_dim=512,
                num_heads=8,
                depth=8,
                mlp_dim=2048,
                dropout=0.1,
                embed_dropout=0.1,
                pool='cls',
                num_classes=1000)

    return model


def vit_m(pretrained=False, **kwargs):
    model = ViT(img_size=224,
                patch_size=16,
                hidden_dim=768,
                num_heads=12,
                depth=12,
                mlp_dim=2048,
                dropout=0.1,
                embed_dropout=0.1,
                pool='cls',
                num_classes=1000)

    return model


def vit_l(pretrained=False, **kwargs):
    model = ViT(img_size=224,
                patch_size=16,
                hidden_dim=1024,
                num_heads=16,
                depth=16,
                mlp_dim=2048,
                dropout=0.1,
                embed_dropout=0.1,
                pool='cls',
                num_classes=1000)

    return model


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = vit()
    y = model(x)
    print(y.size())
