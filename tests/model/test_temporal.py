import torch
from sahformer.model.config import ModelConfig
from sahformer.model.temporal import TemporalEncoder

def test_temporal_encoder_shape():
    c = ModelConfig()
    enc = TemporalEncoder(c)
    out = enc(torch.zeros(5, c.temporal_dim))
    assert out.shape == (5, c.t_ctx)

def test_temporal_encoder_varies_with_input():
    c = ModelConfig()
    enc = TemporalEncoder(c)
    a = enc(torch.zeros(1, c.temporal_dim))
    b = enc(torch.ones(1, c.temporal_dim))
    assert not torch.allclose(a, b)

def test_temporal_encoder_gradients():
    c = ModelConfig()
    enc = TemporalEncoder(c)
    x = torch.randn(3, c.temporal_dim, requires_grad=True)
    enc(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
