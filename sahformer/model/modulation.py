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

class GeometricAttentionBias(nn.Module):
    """Low-rank, time-conditioned attention bias over the 64 board tokens.
    bias[b,h,i,j] = (u_i . v_j) / sqrt(rank), padded with zeros for skill tokens."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.h = cfg.n_heads
        self.r = cfg.gab_rank
        self.nsq = cfg.n_squares
        self.nsk = cfg.n_skill_tokens
        self.token_compress = nn.Linear(cfg.d_model, 32)
        self.board_ctx = nn.Sequential(
            nn.Linear(cfg.n_squares * 32, 128), nn.GELU(), nn.LayerNorm(128)
        )
        cin = 128 + cfg.t_ctx + 2 * cfg.d_model
        self.fuse = nn.Sequential(nn.Linear(cin, 128), nn.GELU())
        self.to_u = nn.Linear(128, self.h * self.nsq * self.r)
        self.to_v = nn.Linear(128, self.h * self.nsq * self.r)
        for lin in (self.to_u, self.to_v):
            nn.init.normal_(lin.weight, std=0.02)
            nn.init.zeros_(lin.bias)

    def forward(self, board_tokens, t, skill_self, skill_opp):
        b = board_tokens.shape[0]
        comp = self.token_compress(board_tokens).reshape(b, -1)     # (B, 64*32)
        bctx = self.board_ctx(comp)                                 # (B, 128)
        c = torch.cat([bctx, t, skill_self, skill_opp], dim=-1)
        c = self.fuse(c)                                            # (B, 128)
        u = self.to_u(c).view(b, self.h, self.nsq, self.r)
        v = self.to_v(c).view(b, self.h, self.nsq, self.r)
        board_bias = torch.einsum("bhir,bhjr->bhij", u, v) / math.sqrt(self.r)
        s = self.cfg.seq_len
        bias = torch.zeros(b, self.h, s, s, device=board_bias.device, dtype=board_bias.dtype)
        bias[:, :, self.nsk:, self.nsk:] = board_bias
        return bias
