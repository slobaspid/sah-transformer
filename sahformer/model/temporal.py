import torch
import torch.nn as nn
from sahformer.model.config import ModelConfig

class TemporalEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.temporal_dim, cfg.t_ctx), nn.GELU(),
            nn.Linear(cfg.t_ctx, cfg.t_ctx),
        )
        self.ln = nn.LayerNorm(cfg.t_ctx)

    def forward(self, temporal):
        return self.ln(self.net(temporal.float()))

class FiLMGenerator(nn.Module):
    """Per-block per-channel (gamma, beta) from t; near identity at init."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n = cfg.num_blocks
        self.d = cfg.dim_vit
        self.gen = nn.Linear(cfg.t_ctx, 2 * cfg.num_blocks * cfg.dim_vit)
        nn.init.normal_(self.gen.weight, std=0.02)
        nn.init.zeros_(self.gen.bias)

    def forward(self, t):
        b = t.shape[0]
        raw = self.gen(t).view(b, self.n, 2, self.d)
        return 1.0 + raw[:, :, 0, :], raw[:, :, 1, :]
