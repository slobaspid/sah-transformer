import torch
from sahformer.model.config import ModelConfig
from sahformer.model.encoder import Encoder

def test_encoder_shape():
    c = ModelConfig()
    enc = Encoder(c)
    x = torch.randn(2, 64, c.dim_vit)
    assert enc(x).shape == (2, 64, c.dim_vit)

def test_encoder_gradients():
    c = ModelConfig()
    enc = Encoder(c)
    x = torch.randn(2, 64, c.dim_vit, requires_grad=True)
    enc(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

def test_shared_gab_weight_is_single_param():
    c = ModelConfig()
    enc = Encoder(c)
    # exactly one parameter named gab_weight, shape (4096, gab_gen_size)
    gw = [p for n, p in enc.named_parameters() if n.endswith("gab_weight")]
    assert len(gw) == 1 and tuple(gw[0].shape) == (4096, c.gab_gen_size)
