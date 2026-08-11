import torch
import torch.nn as nn
from sahformer.model.config import ModelConfig

class SkillEmbedding(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.weak = nn.Parameter(torch.randn(cfg.dim_emb) * 0.02)
        self.strong = nn.Parameter(torch.randn(cfg.dim_emb) * 0.02)

    def forward(self, elo: torch.Tensor) -> torch.Tensor:
        gamma = torch.clamp((5000.0 - elo.float()) / 5000.0, 0.0, 1.0).unsqueeze(-1)
        return gamma * self.weak + (1.0 - gamma) * self.strong

class InputEmbedding(nn.Module):
    """64 tokens: board planes (current+7 history) concatenated per square with the
    two skill embeddings, projected to dim_vit. No absolute positional embedding."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.skill = SkillEmbedding(cfg)
        self.proj = nn.Linear(cfg.token_in, cfg.dim_vit)

    def forward(self, board, history, elo_self, elo_opp):
        b = board.shape[0]
        cur = board.reshape(b, 64, 12)                                   # (B,64,12)
        hist = history.permute(0, 2, 3, 1, 4).reshape(b, 64, 7 * 12)     # (B,64,84)
        planes = torch.cat([cur, hist], dim=-1)                          # (B,64,96)
        ss = self.skill(elo_self).unsqueeze(1).expand(b, 64, -1)         # (B,64,128)
        so = self.skill(elo_opp).unsqueeze(1).expand(b, 64, -1)         # (B,64,128)
        tok = torch.cat([planes, ss, so], dim=-1)                        # (B,64,352)
        return self.proj(tok)                                            # (B,64,dim_vit)
