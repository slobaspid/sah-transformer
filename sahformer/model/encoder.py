import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sahformer.model.config import ModelConfig
from sahformer.model.norm import RMSNorm
from sahformer.model.gab import GeometricAttentionBias

class MHA(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h = cfg.num_heads
        self.dh = cfg.head_dim
        self.qkv = nn.Linear(cfg.dim_vit, 3 * cfg.dim_vit, bias=False)
        self.out = nn.Linear(cfg.dim_vit, cfg.dim_vit, bias=False)

    def forward(self, x, gab_bias):
        b, s, _ = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh) + gab_bias
        attn = F.softmax(logits, dim=-1)
        ctx = (attn @ v).transpose(1, 2).reshape(b, s, self.h * self.dh)
        return self.out(ctx)

class EncoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, time_conditioned: bool = False):
        super().__init__()
        self.attn = MHA(cfg)
        self.gab = GeometricAttentionBias(cfg, time_conditioned=time_conditioned)
        self.norm1 = RMSNorm(cfg.dim_vit)
        self.norm2 = RMSNorm(cfg.dim_vit)
        hidden = cfg.dim_vit * cfg.mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(cfg.dim_vit, hidden), nn.GELU(), nn.Linear(hidden, cfg.dim_vit)
        )

    def forward(self, x, gab_weight, t=None, film=None):
        bias = self.gab(x, gab_weight, t=t)
        x = self.norm1(x + self.attn(x, bias))          # post-norm
        x = self.norm2(x + self.mlp(x))
        if film is not None:
            gamma, beta = film
            x = gamma.unsqueeze(1) * x + beta.unsqueeze(1)
        return x

class Encoder(nn.Module):
    def __init__(self, cfg: ModelConfig, time_conditioned: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList(
            [EncoderBlock(cfg, time_conditioned=time_conditioned) for _ in range(cfg.num_blocks)]
        )
        # shared-across-layers final GAB projection
        self.gab_weight = nn.Parameter(torch.randn(64 * 64, cfg.gab_gen_size) * 0.02)

    def forward(self, x, t=None, film=None):
        for i, blk in enumerate(self.blocks):
            layer_film = None
            if film is not None:
                g, be = film
                layer_film = (g[:, i, :], be[:, i, :])
            x = blk(x, self.gab_weight, t=t, film=layer_film)
        return x
