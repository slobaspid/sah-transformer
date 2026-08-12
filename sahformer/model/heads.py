import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sahformer.model.config import ModelConfig
from sahformer.model.norm import RMSNorm

def move_to_index(from_sq: int, to_sq: int, promo: int) -> int:
    """4352-class move index. Non-promo: from*64+to. Promo: 4096 + to*4 + (promo-1)."""
    if promo == 0:
        return from_sq * 64 + to_sq
    return 4096 + to_sq * 4 + (promo - 1)

class PolicyHead(nn.Module):
    """From-to bilinear (4096) + promotion logits (64 dest x 4 pieces = 256) = 4352."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.proj_from = nn.Linear(cfg.dim_vit, cfg.head_hid_dim, bias=False)
        self.proj_to = nn.Linear(cfg.dim_vit, cfg.head_hid_dim, bias=False)
        self.scale = 1.0 / math.sqrt(cfg.head_hid_dim)
        self.promo = nn.Linear(cfg.dim_vit, 4)

    def forward(self, tokens):
        b = tokens.shape[0]
        qf = self.proj_from(tokens)                       # (B,64,hid)
        kt = self.proj_to(tokens)                         # (B,64,hid)
        moves = torch.einsum("bid,bjd->bij", qf, kt) * self.scale  # (B,64,64)
        moves = moves.reshape(b, 4096)
        promo = self.promo(tokens).reshape(b, 64 * 4)     # (B,256): dest*4 + piece
        return torch.cat([moves, promo], dim=-1)          # (B,4352)

class ValueHead(nn.Module):
    """Norm -> mean-pool -> Linear+ReLU -> 3 (W/D/L)."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.dim_vit)
        self.hid = nn.Linear(cfg.dim_vit, cfg.head_hid_dim)
        self.out = nn.Linear(cfg.head_hid_dim, 3)

    def forward(self, tokens):
        pooled = self.norm(tokens).mean(dim=1)
        return self.out(F.relu(self.hid(pooled)))

def policy_difficulty(move_logits, eps: float = 1e-9):
    """Two difficulty signals derived from the move distribution's spread, DETACHED so the
    think-time loss can't push the move-guesser around:
      - normalized entropy: ~0 when one move dominates (forced/easy), ~1 when moves are
        spread out (ambiguous/hard)
      - top-move probability: how confident the best move is
    Returns (B, 2)."""
    p = F.softmax(move_logits.detach(), dim=-1)
    n = p.shape[-1]
    entropy = -(p * (p + eps).log()).sum(-1, keepdim=True) / math.log(n)
    top = p.max(-1, keepdim=True).values
    return torch.cat([entropy, top], dim=-1)

class ThinkTimeMDNHead(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        m = cfg.mdn_components
        self.trunk = nn.Sequential(
            nn.Linear(cfg.dim_vit + cfg.think_extra, cfg.head_hid_dim), nn.ReLU())
        self.pi = nn.Linear(cfg.head_hid_dim, m)
        self.mu = nn.Linear(cfg.head_hid_dim, m)
        self.sigma = nn.Linear(cfg.head_hid_dim, m)

    def forward(self, pooled):
        h = self.trunk(pooled)
        return self.pi(h), self.mu(h), self.sigma(h)

def mdn_nll_per_sample(pi_logits, mu, sigma_param, target_time, eps: float = 1e-6):
    x = torch.log(target_time.clamp_min(eps)).unsqueeze(-1)
    log_pi = F.log_softmax(pi_logits, dim=-1)
    sigma = F.softplus(sigma_param) + 1e-3
    z = (x - mu) / sigma
    comp = -0.5 * z * z - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
    return -(torch.logsumexp(log_pi + comp, dim=-1))          # (B,)

def mdn_nll(pi_logits, mu, sigma_param, target_time, eps: float = 1e-6):
    return mdn_nll_per_sample(pi_logits, mu, sigma_param, target_time, eps).mean()
