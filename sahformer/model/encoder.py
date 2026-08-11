import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sahformer.model.config import ModelConfig

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h = cfg.n_heads
        self.dh = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x, attn_bias=None):
        b, s, _ = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        if attn_bias is not None:
            logits = logits + attn_bias
        attn = F.softmax(logits, dim=-1)
        ctx = attn @ v
        ctx = ctx.transpose(1, 2).reshape(b, s, self.h * self.dh)
        return self.out(ctx)

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        hidden = cfg.d_model * cfg.mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, hidden), nn.GELU(), nn.Linear(hidden, cfg.d_model)
        )

    def forward(self, x, attn_bias=None, film=None):
        x = x + self.attn(self.ln1(x), attn_bias=attn_bias)
        x = x + self.mlp(self.ln2(x))
        if film is not None:
            gamma, beta = film                      # each (B, d_model)
            x = gamma.unsqueeze(1) * x + beta.unsqueeze(1)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

    def forward(self, x, attn_bias=None, film=None):
        for i, blk in enumerate(self.blocks):
            layer_film = None
            if film is not None:
                gamma_all, beta_all = film          # each (B, n_layers, d_model)
                layer_film = (gamma_all[:, i, :], beta_all[:, i, :])
            x = blk(x, attn_bias=attn_bias, film=layer_film)
        return self.ln_f(x)
