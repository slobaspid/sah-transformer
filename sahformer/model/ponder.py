import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sahformer.model.config import ModelConfig
from sahformer.model.norm import RMSNorm
from sahformer.model.embedding import InputEmbedding
from sahformer.model.encoder import Encoder
from sahformer.model.temporal import TemporalEncoder, FiLMGenerator
from sahformer.model.heads import PolicyHead, ValueHead, ThinkTimeMDNHead, policy_difficulty

class _PonderMHA(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h = cfg.num_heads
        self.dh = cfg.head_dim
        self.qkv = nn.Linear(cfg.dim_vit, 3 * cfg.dim_vit, bias=False)
        self.out = nn.Linear(cfg.dim_vit, cfg.dim_vit, bias=False)

    def forward(self, x):
        b, s, _ = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.dh), dim=-1)
        ctx = (attn @ v).transpose(1, 2).reshape(b, s, self.h * self.dh)
        return self.out(ctx)

class PonderBlock(nn.Module):
    """One shared recurrent refinement step: pre-norm self-attention + MLP (no GAB)."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim_vit)
        self.attn = _PonderMHA(cfg)
        self.n2 = RMSNorm(cfg.dim_vit)
        hidden = cfg.dim_vit * cfg.mlp_ratio
        self.mlp = nn.Sequential(nn.Linear(cfg.dim_vit, hidden), nn.GELU(),
                                 nn.Linear(hidden, cfg.dim_vit))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x

class HaltHead(nn.Module):
    """P(halt at this step | reached it), from the pooled state + the clock context t."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.dim_vit + cfg.t_ctx, cfg.head_hid_dim), nn.ReLU(),
            nn.Linear(cfg.head_hid_dim, 1))

    def forward(self, pooled, t):
        return torch.sigmoid(self.net(torch.cat([pooled, t], dim=-1))).squeeze(-1)  # (B,)

class PonderChessformer(nn.Module):
    """Clock-aware backbone + adaptive-computation ponder loop (PonderNet halting)."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_emb = InputEmbedding(cfg)
        self.temporal_enc = TemporalEncoder(cfg)
        self.film_gen = FiLMGenerator(cfg)
        self.encoder = Encoder(cfg, time_conditioned=True)
        self.ponder = PonderBlock(cfg)
        self.halt = HaltHead(cfg)
        self.policy = PolicyHead(cfg)
        self.value = ValueHead(cfg)
        self.think = ThinkTimeMDNHead(cfg)
        self.t_to_d = nn.Linear(cfg.t_ctx, cfg.dim_vit)

    def _encode(self, batch):
        tok = self.input_emb(batch["board"].float(), batch["history"].float(),
                             batch["elo_self"], batch["elo_opp"])
        t = self.temporal_enc(batch["temporal"])
        film = self.film_gen(t)
        enc = self.encoder(tok, t=t, film=film)
        return enc, t

    def _heads(self, h, t):
        pooled = h.mean(dim=1) + self.t_to_d(t)
        move_logits = self.policy(h)
        think_in = torch.cat([pooled, policy_difficulty(move_logits)], dim=-1)
        return {"move_logits": move_logits, "value_logits": self.value(h),
                "mdn": self.think(think_in)}

    def ponder_train(self, batch):
        """Per-step head outputs + halting distribution p_halt (B, K) for the PonderNet loss."""
        enc, t = self._encode(batch)
        K = self.cfg.max_ponder
        h = enc
        steps, lambdas = [], []
        for n in range(K):
            lam = self.halt(h.mean(dim=1), t)
            if n == K - 1:
                lam = torch.ones_like(lam)           # force halt at the last step
            lambdas.append(lam)
            steps.append(self._heads(h, t))
            if n < K - 1:
                h = self.ponder(h)
        p, remain = [], torch.ones_like(lambdas[0])
        for n in range(K):
            p.append(remain * lambdas[n])
            remain = remain * (1 - lambdas[n])
        return {"steps": steps, "p_halt": torch.stack(p, dim=1)}

    @torch.no_grad()
    def forward(self, batch, halt_threshold=0.5, max_steps=None):
        """Drop-in inference: run the ponder loop, halt early, return the usual output dict
        (+ ponder_steps). max_steps caps depth (the clock budget)."""
        enc, t = self._encode(batch)
        K = self.cfg.max_ponder if max_steps is None else min(int(max_steps), self.cfg.max_ponder)
        K = max(1, K)
        h = enc
        cum = torch.zeros(enc.shape[0], device=enc.device)
        out, used = None, 1
        for n in range(K):
            out = self._heads(h, t)
            cum = cum + self.halt(h.mean(dim=1), t)
            used = n + 1
            if float(cum.mean()) >= halt_threshold or n == K - 1:
                break
            h = self.ponder(h)
        out["ponder_steps"] = used
        return out
