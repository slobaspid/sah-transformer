import torch
import torch.nn as nn
from sahformer.model.config import ModelConfig

class TemporalEncoder(nn.Module):
    """Map the 21-dim in-game temporal feature vector to a context vector t."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.temporal_dim, cfg.t_ctx),
            nn.GELU(),
            nn.Linear(cfg.t_ctx, cfg.t_ctx),
        )
        self.ln = nn.LayerNorm(cfg.t_ctx)

    def forward(self, temporal: torch.Tensor) -> torch.Tensor:
        return self.ln(self.net(temporal.float()))
