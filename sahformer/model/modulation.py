import math
import torch
import torch.nn as nn
from sahformer.model.config import ModelConfig

class FiLMGenerator(nn.Module):
    """Produce per-layer, per-channel (gamma, beta) from the temporal context t.
    Initialized so gamma~1, beta~0 (near identity at start)."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_layers = cfg.n_layers
        self.d_model = cfg.d_model
        self.gen = nn.Linear(cfg.t_ctx, 2 * cfg.n_layers * cfg.d_model)
        nn.init.normal_(self.gen.weight, std=0.02)
        nn.init.zeros_(self.gen.bias)

    def forward(self, t: torch.Tensor):
        b = t.shape[0]
        raw = self.gen(t).view(b, self.n_layers, 2, self.d_model)
        gamma = 1.0 + raw[:, :, 0, :]
        beta = raw[:, :, 1, :]
        return gamma, beta
