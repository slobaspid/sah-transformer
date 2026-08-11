import torch
import torch.nn as nn
from sahformer.model.config import ModelConfig
from sahformer.model.embedding import InputEmbedding, SkillEmbedding
from sahformer.model.encoder import TransformerEncoder
from sahformer.model.heads import PolicyHead, ValueHead, ThinkTimeMDNHead
from sahformer.model.temporal import TemporalEncoder
from sahformer.model.modulation import FiLMGenerator, GeometricAttentionBias

class ClockAwareChessformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_emb = InputEmbedding(cfg)
        self.skill_emb = SkillEmbedding(cfg)
        self.temporal_enc = TemporalEncoder(cfg)
        self.film_gen = FiLMGenerator(cfg)
        self.gab = GeometricAttentionBias(cfg)
        self.encoder = TransformerEncoder(cfg)
        self.policy = PolicyHead(cfg)
        self.value = ValueHead(cfg)
        self.think = ThinkTimeMDNHead(cfg)
        self.t_to_d = nn.Linear(cfg.t_ctx, cfg.d_model)

    def forward(self, batch: dict) -> dict:
        board = batch["board"].float()
        history = batch["history"].float()
        board_tok = self.input_emb(board, history)              # (B,64,d)
        self_tok = self.skill_emb(batch["elo_self"])            # (B,d)
        opp_tok = self.skill_emb(batch["elo_opp"])              # (B,d)
        t = self.temporal_enc(batch["temporal"])               # (B,t_ctx)

        gamma, beta = self.film_gen(t)                         # each (B,L,d)
        bias = self.gab(board_tok, t, self_tok, opp_tok)       # (B,h,66,66)

        seq = torch.cat([self_tok.unsqueeze(1), opp_tok.unsqueeze(1), board_tok], dim=1)
        enc = self.encoder(seq, attn_bias=bias, film=(gamma, beta))
        board_out = enc[:, self.cfg.n_skill_tokens:, :]        # (B,64,d)
        pooled = board_out.mean(dim=1) + self.t_to_d(t)        # (B,d), clock-aware

        move_logits, promo_logits = self.policy(board_out)
        value_logits = self.value(pooled)
        mdn = self.think(pooled)
        return {
            "move_logits": move_logits,
            "promo_logits": promo_logits,
            "value_logits": value_logits,
            "mdn": mdn,
        }
