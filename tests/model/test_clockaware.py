import torch
from sahformer.model.config import ModelConfig
from sahformer.model.clockaware import ClockAwareChessformer

def _batch(b=2, fill=0.0):
    return {"board": torch.zeros(b, 8, 8, 12), "history": torch.zeros(b, 7, 8, 8, 12),
            "elo_self": torch.tensor([1500] * b), "elo_opp": torch.tensor([1600] * b),
            "temporal": torch.full((b, 21), fill)}

def test_shapes():
    c = ModelConfig()
    out = ClockAwareChessformer(c)(_batch(3))
    assert out["move_logits"].shape == (3, 4352)
    assert out["value_logits"].shape == (3, 3)
    assert out["mdn"][0].shape == (3, c.mdn_components)

def test_param_budget():
    c = ModelConfig()
    n = sum(p.numel() for p in ClockAwareChessformer(c).parameters())
    assert 3_000_000 < n < 12_000_000, n

def test_time_changes_output():
    c = ModelConfig()
    m = ClockAwareChessformer(c); m.eval()
    with torch.no_grad():
        a = m(_batch(2, 0.0)); b = m(_batch(2, 1.0))
    assert not torch.allclose(a["move_logits"], b["move_logits"], atol=1e-6)
    assert not torch.allclose(a["mdn"][1], b["mdn"][1], atol=1e-6)
