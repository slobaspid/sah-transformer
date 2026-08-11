import torch
from sahformer.model.config import ModelConfig
from sahformer.model.clockaware import ClockAwareChessformer

def _batch(b=3, temporal_fill=0.0):
    return {
        "board": torch.zeros(b, 8, 8, 12),
        "history": torch.zeros(b, 7, 8, 8, 12),
        "elo_self": torch.tensor([1500] * b),
        "elo_opp": torch.tensor([1600] * b),
        "temporal": torch.full((b, 21), temporal_fill),
    }

def test_forward_shapes():
    c = ModelConfig()
    model = ClockAwareChessformer(c)
    out = model(_batch(3))
    assert out["move_logits"].shape == (3, 64, 64)
    assert out["promo_logits"].shape == (3, 64, 4)
    assert out["value_logits"].shape == (3, 3)
    assert out["mdn"][0].shape == (3, c.mdn_components)

def test_param_budget():
    c = ModelConfig()
    n = sum(p.numel() for p in ClockAwareChessformer(c).parameters())
    assert 4_500_000 < n < 12_000_000, n

def test_backward_runs():
    c = ModelConfig()
    model = ClockAwareChessformer(c)
    out = model(_batch(2))
    (out["move_logits"].sum() + out["value_logits"].sum()).backward()
    assert any(p.grad is not None for p in model.parameters())

def test_time_actually_changes_output():
    c = ModelConfig()
    model = ClockAwareChessformer(c)
    model.eval()
    with torch.no_grad():
        calm = model(_batch(2, temporal_fill=0.0))
        panic = model(_batch(2, temporal_fill=1.0))
    assert not torch.allclose(calm["move_logits"], panic["move_logits"], atol=1e-6)
    assert not torch.allclose(calm["mdn"][1], panic["mdn"][1], atol=1e-6)
