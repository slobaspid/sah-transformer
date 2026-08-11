import torch
import torch.nn as nn
from sahformer.model.config import ModelConfig
from sahformer.model.embedding import InputEmbedding, SkillEmbedding
from sahformer.model.encoder import TransformerEncoder
from sahformer.model.heads import PolicyHead, ValueHead, ThinkTimeMDNHead

class Chessformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_emb = InputEmbedding(cfg)
        self.skill_emb = SkillEmbedding(cfg)
        self.encoder = TransformerEncoder(cfg)
        self.policy = PolicyHead(cfg)
        self.value = ValueHead(cfg)
        self.think = ThinkTimeMDNHead(cfg)

    def forward(self, batch: dict) -> dict:
        board = batch["board"].float()
        history = batch["history"].float()
        board_tok = self.input_emb(board, history)          # (B,64,d)
        self_tok = self.skill_emb(batch["elo_self"]).unsqueeze(1)   # (B,1,d)
        opp_tok = self.skill_emb(batch["elo_opp"]).unsqueeze(1)     # (B,1,d)
        seq = torch.cat([self_tok, opp_tok, board_tok], dim=1)      # (B,66,d)
        enc = self.encoder(seq)                             # (B,66,d)
        board_out = enc[:, self.cfg.n_skill_tokens:, :]     # (B,64,d)
        pooled = board_out.mean(dim=1)                      # (B,d)
        move_logits, promo_logits = self.policy(board_out)
        value_logits = self.value(pooled)
        mdn = self.think(pooled)
        return {
            "move_logits": move_logits,
            "promo_logits": promo_logits,
            "value_logits": value_logits,
            "mdn": mdn,
        }
