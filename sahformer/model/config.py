from dataclasses import dataclass

@dataclass
class ModelConfig:
    history: int = 8              # current + 7 past
    dim_emb: int = 128            # skill embedding width
    dim_vit: int = 256            # d_model
    num_blocks: int = 8
    num_heads: int = 8
    mlp_ratio: int = 2
    gab_gen_size: int = 64        # d3
    gab_intermediate_dim: int = 64  # d2
    gab_per_square_dim: int = 0   # d1 (0 => avg-pool variant)
    head_hid_dim: int = 256
    mdn_components: int = 3
    think_extra: int = 2          # difficulty features (policy entropy + top-move prob) into the time head
    max_ponder: int = 4           # max adaptive-computation (ponder) steps
    ponder_prior: float = 0.4     # geometric-prior halting rate for PonderNet
    n_squares: int = 64
    temporal_dim: int = 21        # Plan 3
    t_ctx: int = 128              # Plan 3

    @property
    def in_channels(self) -> int:
        return 12 * self.history

    @property
    def token_in(self) -> int:
        return self.in_channels + 2 * self.dim_emb

    @property
    def head_dim(self) -> int:
        return self.dim_vit // self.num_heads
