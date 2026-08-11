import torch
from sahformer.model.config import ModelConfig
from sahformer.model.chessformer import Chessformer

def _batch(b=3):
    return {
        "board": torch.zeros(b, 8, 8, 12),
        "history": torch.zeros(b, 7, 8, 8, 12),
        "elo_self": torch.tensor([1500] * b),
        "elo_opp": torch.tensor([1600] * b),
    }

def test_forward_output_shapes():
    c = ModelConfig()
    model = Chessformer(c)
    out = model(_batch(3))
    assert out["move_logits"].shape == (3, 64, 64)
    assert out["promo_logits"].shape == (3, 64, 4)
    assert out["value_logits"].shape == (3, 3)
    pi, mu, sigma_p = out["mdn"]
    assert pi.shape == (3, c.mdn_components)

def test_param_count_is_5m_class():
    c = ModelConfig()
    model = Chessformer(c)
    n = sum(p.numel() for p in model.parameters())
    assert 2_000_000 < n < 12_000_000, n

def test_backward_runs():
    c = ModelConfig()
    model = Chessformer(c)
    out = model(_batch(2))
    loss = out["move_logits"].sum() + out["value_logits"].sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
