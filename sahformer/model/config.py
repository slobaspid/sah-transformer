from dataclasses import dataclass

@dataclass
class ModelConfig:
    n_squares: int = 64
    in_channels: int = 96          # 12 planes x (1 current + 7 history)
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    mlp_ratio: int = 2
    skill_emb: int = 128
    temporal_dim: int = 21
    mdn_components: int = 3
    n_skill_tokens: int = 2        # self + opponent

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def seq_len(self) -> int:
        return self.n_skill_tokens + self.n_squares
