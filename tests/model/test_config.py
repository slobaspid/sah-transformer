from sahformer.model.config import ModelConfig

def test_defaults_match_5m_class():
    c = ModelConfig()
    assert c.n_squares == 64
    assert c.in_channels == 96
    assert c.d_model == 256
    assert c.n_layers == 8
    assert c.n_heads == 8
    assert c.d_model % c.n_heads == 0
    assert c.skill_emb == 128
    assert c.mdn_components == 3
    assert c.seq_len == 66  # 2 skill + 64 board

def test_head_dim_derived():
    c = ModelConfig()
    assert c.head_dim == 32

def test_modulation_dims():
    c = ModelConfig()
    assert c.t_ctx == 128
    assert c.gab_rank == 4
