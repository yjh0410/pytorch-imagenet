import torch
import torch.nn as nn


# PreNorm
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


# FFN
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


# MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 dim,
                 heads=8, 
                 dim_head = 64,
                 dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5     # 1 / sqrt(d_k)

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # [B, N, h*d] -> [B, N, h, d] -> [B, h, N, d]
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        B, NQ = q.shape[:2]
        B, NK = k.shape[:2]
        B, NV = v.shape[:2]
        q = q.view(B, NQ, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, NK, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, NV, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

        # Q*K^T / sqrt(d_k) : [B, h, N, d] X [B, h, d, N] = [B, h, N, N]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # softmax
        attn = self.attend(dots)
        # softmax(Q*K^T / sqrt(d_k)) * V ：[B, h, N, N] X [B, h, N, d] = [B, h, N, d]
        out = torch.matmul(attn, v)
        # [B, h, N, d] -> [B, N, h*d]=[B, N, C_out], C_out = h*d
        out = out.permute(0, 2, 1, 3).contiguous().view(B, NQ, -1)
        
        return self.to_out(out)


# Transformer Encoder Layer
class TransformerEncoder(nn.Module):
    def __init__(self, 
                 dim,            # hidden_dim
                 depth,
                 heads, 
                 dim_head,
                 mlp_dim=2048,
                 dropout = 0.):
        super().__init__()
        self.encoder_layers = nn.ModuleList([])
        for _ in range(depth):
            self.encoder_layers.append(nn.ModuleList([
                PreNorm(dim, MultiHeadAttention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        # x -> [B, N, d_in]
        for attn, ffn in self.encoder_layers:
            x = attn(x) + x
            x = ffn(x) + x
        return x
