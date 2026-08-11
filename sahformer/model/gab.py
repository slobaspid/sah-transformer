import torch
import torch.nn as nn
from sahformer.model.config import ModelConfig

class GeometricAttentionBias(nn.Module):
    """Per-layer GAB, 5M avg-pool variant. Optionally conditioned on temporal ctx t."""
    def __init__(self, cfg: ModelConfig, time_conditioned: bool = False):
        super().__init__()
        self.cfg = cfg
        self.h = cfg.num_heads
        self.time_conditioned = time_conditioned
        in_dim = cfg.dim_vit + (cfg.t_ctx if time_conditioned else 0)
        self.lin1 = nn.Linear(in_dim, cfg.gab_intermediate_dim)
        self.ln1 = nn.LayerNorm(cfg.gab_intermediate_dim)
        self.lin2 = nn.Linear(cfg.gab_intermediate_dim, self.h * cfg.gab_gen_size)
        self.ln2 = nn.LayerNorm(self.h * cfg.gab_gen_size)
        self.act = nn.GELU()

    def forward(self, tokens, gab_weight, t=None):
        b = tokens.shape[0]
        pooled = tokens.mean(dim=1)                      # (B, dim_vit) avg-pool variant
        if self.time_conditioned:
            assert t is not None, "time_conditioned GAB requires t"
            pooled = torch.cat([pooled, t], dim=-1)
        y = self.ln1(self.act(self.lin1(pooled)))
        y = self.ln2(self.act(self.lin2(y)))
        y = y.view(b, self.h, self.cfg.gab_gen_size)     # (B, h, d3)
        # shared final projection: templates -> 4096 -> 64x64
        flat = torch.einsum("bhi,oi->bho", y, gab_weight)  # (B, h, 4096)
        return flat.view(b, self.h, 64, 64)
