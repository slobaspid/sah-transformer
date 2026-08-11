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
