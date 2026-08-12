import torch
import torch.nn as nn
from sahformer.model.config import ModelConfig
from sahformer.model.embedding import InputEmbedding
from sahformer.model.encoder import Encoder
from sahformer.model.heads import PolicyHead, ValueHead, ThinkTimeMDNHead, policy_difficulty

class FaithfulChessformer(nn.Module):
    """Clock-blind Maia-3-faithful backbone + our MDN think-time head."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_emb = InputEmbedding(cfg)
        self.encoder = Encoder(cfg)
        self.policy = PolicyHead(cfg)
        self.value = ValueHead(cfg)
        self.think = ThinkTimeMDNHead(cfg)

    def forward(self, batch: dict) -> dict:
        tok = self.input_emb(batch["board"].float(), batch["history"].float(),
                             batch["elo_self"], batch["elo_opp"])
        enc = self.encoder(tok)                       # (B,64,dim_vit)
        pooled = enc.mean(dim=1)
        move_logits = self.policy(enc)
        think_in = torch.cat([pooled, policy_difficulty(move_logits)], dim=-1)
        return {
            "move_logits": move_logits,
            "value_logits": self.value(enc),
            "mdn": self.think(think_in),
        }
