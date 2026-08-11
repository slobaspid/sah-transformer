import torch
import torch.nn as nn
from sahformer.model.config import ModelConfig
from sahformer.model.embedding import InputEmbedding
from sahformer.model.encoder import Encoder
from sahformer.model.heads import PolicyHead, ValueHead, ThinkTimeMDNHead
from sahformer.model.temporal import TemporalEncoder, FiLMGenerator

class ClockAwareChessformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_emb = InputEmbedding(cfg)
        self.temporal_enc = TemporalEncoder(cfg)
        self.film_gen = FiLMGenerator(cfg)
        self.encoder = Encoder(cfg, time_conditioned=True)
        self.policy = PolicyHead(cfg)
        self.value = ValueHead(cfg)
        self.think = ThinkTimeMDNHead(cfg)
        self.t_to_d = nn.Linear(cfg.t_ctx, cfg.dim_vit)

    def forward(self, batch: dict) -> dict:
        tok = self.input_emb(batch["board"].float(), batch["history"].float(),
                             batch["elo_self"], batch["elo_opp"])
        t = self.temporal_enc(batch["temporal"])
        film = self.film_gen(t)
        enc = self.encoder(tok, t=t, film=film)        # (B,64,dim_vit)
        pooled = enc.mean(dim=1) + self.t_to_d(t)
        return {
            "move_logits": self.policy(enc),
            "value_logits": self.value(enc),
            "mdn": self.think(pooled),
        }
