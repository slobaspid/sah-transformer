import torch
from sahformer.model.config import ModelConfig
from sahformer.model.chessformer import FaithfulChessformer

def _batch(b=3):
    return {"board": torch.zeros(b, 8, 8, 12), "history": torch.zeros(b, 7, 8, 8, 12),
            "elo_self": torch.tensor([1500] * b), "elo_opp": torch.tensor([1600] * b)}

def test_forward_shapes():
    c = ModelConfig()
    out = FaithfulChessformer(c)(_batch(3))
    assert out["move_logits"].shape == (3, 4352)
    assert out["value_logits"].shape == (3, 3)
    assert out["mdn"][0].shape == (3, c.mdn_components)

def test_param_count_5m_class():
    c = ModelConfig()
    n = sum(p.numel() for p in FaithfulChessformer(c).parameters())
    assert 3_000_000 < n < 9_000_000, n

def test_backward():
    c = ModelConfig()
    out = FaithfulChessformer(c)(_batch(2))
    out["move_logits"].sum().backward()
