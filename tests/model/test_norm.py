import torch
from sahformer.model.config import ModelConfig
from sahformer.model.norm import RMSNorm

def test_rmsnorm_shape_and_scale():
    n = RMSNorm(256)
    x = torch.randn(2, 64, 256)
    out = n(x)
    assert out.shape == x.shape
    # RMS of output along last dim ~ 1 at init (weight=1)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)
