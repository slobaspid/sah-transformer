import torch
from sahformer.model.config import ModelConfig
from sahformer.model.temporal import TemporalEncoder, FiLMGenerator

def test_temporal_encoder_shape():
    c = ModelConfig()
    out = TemporalEncoder(c)(torch.zeros(5, c.temporal_dim))
    assert out.shape == (5, c.t_ctx)

def test_temporal_varies():
    c = ModelConfig()
    enc = TemporalEncoder(c)
    assert not torch.allclose(enc(torch.zeros(1, c.temporal_dim)),
                              enc(torch.ones(1, c.temporal_dim)))

def test_film_shapes_and_identity():
    c = ModelConfig()
    gen = FiLMGenerator(c)
    gamma, beta = gen(torch.zeros(2, c.t_ctx))
    assert gamma.shape == (2, c.num_blocks, c.dim_vit)
    assert beta.shape == (2, c.num_blocks, c.dim_vit)
    assert torch.allclose(gamma, torch.ones_like(gamma), atol=0.2)
    assert torch.allclose(beta, torch.zeros_like(beta), atol=0.2)
