import torch
from sahformer.model.config import ModelConfig
from sahformer.model.modulation import FiLMGenerator
from sahformer.model.encoder import TransformerEncoder

def test_film_generator_shapes():
    c = ModelConfig()
    gen = FiLMGenerator(c)
    gamma, beta = gen(torch.randn(4, c.t_ctx))
    assert gamma.shape == (4, c.n_layers, c.d_model)
    assert beta.shape == (4, c.n_layers, c.d_model)

def test_film_starts_near_identity():
    c = ModelConfig()
    gen = FiLMGenerator(c)
    gamma, beta = gen(torch.zeros(2, c.t_ctx))
    assert torch.allclose(gamma, torch.ones_like(gamma), atol=0.2)
    assert torch.allclose(beta, torch.zeros_like(beta), atol=0.2)

def test_encoder_film_changes_output():
    c = ModelConfig()
    enc = TransformerEncoder(c)
    x = torch.randn(2, c.seq_len, c.d_model)
    base = enc(x)
    gamma = torch.full((2, c.n_layers, c.d_model), 1.5)
    beta = torch.full((2, c.n_layers, c.d_model), 0.3)
    modulated = enc(x, film=(gamma, beta))
    assert not torch.allclose(base, modulated)

def test_encoder_no_film_matches_baseline():
    c = ModelConfig()
    enc = TransformerEncoder(c)
    x = torch.randn(2, c.seq_len, c.d_model)
    assert torch.allclose(enc(x), enc(x, film=None))

from sahformer.model.modulation import GeometricAttentionBias

def test_gab_bias_shape():
    c = ModelConfig()
    gab = GeometricAttentionBias(c)
    board = torch.randn(3, 64, c.d_model)
    t = torch.randn(3, c.t_ctx)
    ss = torch.randn(3, c.d_model)
    so = torch.randn(3, c.d_model)
    bias = gab(board, t, ss, so)
    assert bias.shape == (3, c.n_heads, c.seq_len, c.seq_len)

def test_gab_skill_token_rows_are_zero():
    c = ModelConfig()
    gab = GeometricAttentionBias(c)
    bias = gab(torch.randn(2, 64, c.d_model), torch.randn(2, c.t_ctx),
               torch.randn(2, c.d_model), torch.randn(2, c.d_model))
    nsk = c.n_skill_tokens
    assert torch.count_nonzero(bias[:, :, :nsk, :]) == 0
    assert torch.count_nonzero(bias[:, :, :, :nsk]) == 0

def test_gab_bias_depends_on_time():
    c = ModelConfig()
    gab = GeometricAttentionBias(c)
    board = torch.randn(2, 64, c.d_model)
    ss = torch.randn(2, c.d_model)
    so = torch.randn(2, c.d_model)
    b_calm = gab(board, torch.zeros(2, c.t_ctx), ss, so)
    b_panic = gab(board, torch.ones(2, c.t_ctx) * 3.0, ss, so)
    assert not torch.allclose(b_calm, b_panic)

def test_gab_gradients():
    c = ModelConfig()
    gab = GeometricAttentionBias(c)
    t = torch.randn(2, c.t_ctx, requires_grad=True)
    gab(torch.randn(2, 64, c.d_model), t, torch.randn(2, c.d_model),
        torch.randn(2, c.d_model)).sum().backward()
    assert t.grad is not None and torch.isfinite(t.grad).all()
