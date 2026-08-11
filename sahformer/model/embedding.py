import torch
import torch.nn as nn
from sahformer.model.config import ModelConfig

class InputEmbedding(nn.Module):
    """Board + 7 history planes -> 64 tokens of width d_model, with learned
    absolute positional embeddings."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.in_channels, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(cfg.n_squares, cfg.d_model))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, board: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        b = board.shape[0]
        cur = board.reshape(b, 64, 12)                       # (B,64,12)
        hist = history.permute(0, 2, 3, 1, 4).reshape(b, 64, 7 * 12)  # (B,64,84)
        x = torch.cat([cur, hist], dim=-1)                   # (B,64,96)
        x = self.proj(x)                                     # (B,64,d_model)
        return x + self.pos.unsqueeze(0)

class SkillEmbedding(nn.Module):
    """Interpolated Elo embedding between a weak (0) and strong (5000) endpoint,
    projected to d_model."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.weak = nn.Parameter(torch.randn(cfg.skill_emb) * 0.02)
        self.strong = nn.Parameter(torch.randn(cfg.skill_emb) * 0.02)
        self.proj = nn.Linear(cfg.skill_emb, cfg.d_model)

    def forward(self, elo: torch.Tensor) -> torch.Tensor:
        elo = elo.float()
        gamma = torch.clamp((5000.0 - elo) / 5000.0, 0.0, 1.0).unsqueeze(-1)  # (B,1)
        emb = gamma * self.weak + (1.0 - gamma) * self.strong                 # (B,skill_emb)
        return self.proj(emb)                                                 # (B,d_model)
