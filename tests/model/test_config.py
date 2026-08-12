from sahformer.model.config import ModelConfig

def test_faithful_dims():
    c = ModelConfig()
    assert c.history == 8
    assert c.in_channels == 96
    assert c.dim_emb == 128
    assert c.token_in == 352
    assert c.dim_vit == 256
    assert c.num_blocks == 8
    assert c.num_heads == 8
    assert c.head_dim == 32
    assert c.gab_gen_size == 64
    assert c.gab_intermediate_dim == 64
    assert c.gab_per_square_dim == 0
    assert c.head_hid_dim == 256
    assert c.mdn_components == 3
    assert c.n_squares == 64

def test_ponder_defaults():
    c = ModelConfig()
    assert c.max_ponder == 4
    assert abs(c.ponder_prior - 0.4) < 1e-9
