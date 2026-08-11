import torch
from sahformer.model.config import ModelConfig
from sahformer.model.gab import GeometricAttentionBias

def _gw(c):
    return torch.randn(64 * 64, c.gab_gen_size)

def test_gab_shape():
    c = ModelConfig()
    gab = GeometricAttentionBias(c)
    bias = gab(torch.randn(3, 64, c.dim_vit), _gw(c))
    assert bias.shape == (3, c.num_heads, 64, 64)

def test_gab_depends_on_board():
    c = ModelConfig()
    gab = GeometricAttentionBias(c)
    gw = _gw(c)
    a = gab(torch.zeros(2, 64, c.dim_vit), gw)
    b = gab(torch.randn(2, 64, c.dim_vit), gw)
    assert not torch.allclose(a, b)

def test_gab_gradients():
    c = ModelConfig()
    gab = GeometricAttentionBias(c)
    x = torch.randn(2, 64, c.dim_vit, requires_grad=True)
    gab(x, _gw(c)).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

def test_gab_time_conditioned_shape():
    c = ModelConfig()
    gab = GeometricAttentionBias(c, time_conditioned=True)
    bias = gab(torch.randn(2, 64, c.dim_vit), _gw(c), t=torch.randn(2, c.t_ctx))
    assert bias.shape == (2, c.num_heads, 64, 64)
