import torch
from sahformer.model.config import ModelConfig
from sahformer.model.encoder import TransformerBlock, TransformerEncoder

def test_block_preserves_shape():
    c = ModelConfig()
    blk = TransformerBlock(c)
    x = torch.randn(2, c.seq_len, c.d_model)
    out = blk(x)
    assert out.shape == x.shape

def test_encoder_preserves_shape():
    c = ModelConfig()
    enc = TransformerEncoder(c)
    x = torch.randn(2, c.seq_len, c.d_model)
    out = enc(x)
    assert out.shape == x.shape

def test_attention_bias_accepted():
    c = ModelConfig()
    blk = TransformerBlock(c)
    x = torch.randn(2, c.seq_len, c.d_model)
    bias = torch.zeros(2, c.n_heads, c.seq_len, c.seq_len)
    out = blk(x, attn_bias=bias)
    assert out.shape == x.shape

def test_gradients_flow():
    c = ModelConfig()
    enc = TransformerEncoder(c)
    x = torch.randn(2, c.seq_len, c.d_model, requires_grad=True)
    enc(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
