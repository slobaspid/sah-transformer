import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sahformer.model.config import ModelConfig

class PolicyHead(nn.Module):
    """Source-destination attention policy. move_logits[b, from, to]."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.q = nn.Linear(cfg.d_model, cfg.d_model)   # source (from) queries
        self.k = nn.Linear(cfg.d_model, cfg.d_model)   # destination (to) keys
        self.scale = 1.0 / math.sqrt(cfg.d_model)
        self.promo = nn.Linear(cfg.d_model, 4)         # from destination keys

    def forward(self, board_tokens: torch.Tensor):
        q = self.q(board_tokens)                       # (B,64,d)
        k = self.k(board_tokens)                       # (B,64,d)
        move_logits = (q @ k.transpose(-2, -1)) * self.scale  # (B,64,64)
        promo_logits = self.promo(k)                   # (B,64,4)
        return move_logits, promo_logits

class ValueHead(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 128), nn.ReLU(), nn.Linear(128, 3)
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)

class ThinkTimeMDNHead(nn.Module):
    """Predict a mixture of Gaussians over log think-time."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        m = cfg.mdn_components
        self.trunk = nn.Sequential(nn.Linear(cfg.d_model, 128), nn.ReLU())
        self.pi = nn.Linear(128, m)
        self.mu = nn.Linear(128, m)
        self.sigma = nn.Linear(128, m)

    def forward(self, pooled: torch.Tensor):
        h = self.trunk(pooled)
        return self.pi(h), self.mu(h), self.sigma(h)

def mdn_nll(pi_logits, mu, sigma_param, target_time, eps: float = 1e-6):
    """Negative log-likelihood of log(target_time) under the Gaussian mixture."""
    x = torch.log(target_time.clamp_min(eps)).unsqueeze(-1)   # (B,1)
    log_pi = F.log_softmax(pi_logits, dim=-1)                 # (B,M)
    sigma = F.softplus(sigma_param) + 1e-3                    # (B,M)
    z = (x - mu) / sigma
    comp_logpdf = -0.5 * z * z - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
    log_prob = torch.logsumexp(log_pi + comp_logpdf, dim=-1)  # (B,)
    return -log_prob.mean()
