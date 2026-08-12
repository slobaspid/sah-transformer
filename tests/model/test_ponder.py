import torch
from sahformer.model.config import ModelConfig
from sahformer.model.ponder import PonderBlock, HaltHead, PonderChessformer

def _batch(b=2, fill=0.0):
    return {"board": torch.zeros(b, 8, 8, 12), "history": torch.zeros(b, 7, 8, 8, 12),
            "elo_self": torch.tensor([1500] * b), "elo_opp": torch.tensor([1600] * b),
            "temporal": torch.full((b, 21), fill)}

def test_ponder_block_shape_and_grad():
    c = ModelConfig()
    blk = PonderBlock(c)
    x = torch.randn(2, 64, c.dim_vit, requires_grad=True)
    out = blk(x)
    assert out.shape == (2, 64, c.dim_vit)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

def test_halt_head_in_range_and_uses_clock():
    c = ModelConfig()
    hh = HaltHead(c)
    pooled = torch.randn(3, c.dim_vit)
    lam_a = hh(pooled, torch.zeros(3, c.t_ctx))
    lam_b = hh(pooled, torch.ones(3, c.t_ctx))
    assert lam_a.shape == (3,)
    assert (lam_a >= 0).all() and (lam_a <= 1).all()
    assert not torch.allclose(lam_a, lam_b)          # halting depends on the clock context

def test_ponder_train_halting_sums_to_one():
    c = ModelConfig()
    m = PonderChessformer(c)
    out = m.ponder_train(_batch(3))
    assert len(out["steps"]) == c.max_ponder
    assert out["p_halt"].shape == (3, c.max_ponder)
    assert (out["p_halt"] >= 0).all()
    assert torch.allclose(out["p_halt"].sum(dim=1), torch.ones(3), atol=1e-5)

def test_ponder_forward_is_dropin():
    c = ModelConfig()
    m = PonderChessformer(c); m.eval()
    out = m(_batch(2))
    assert out["move_logits"].shape == (2, 4352)
    assert out["value_logits"].shape == (2, 3)
    assert out["mdn"][0].shape == (2, c.mdn_components)
    assert 1 <= out["ponder_steps"] <= c.max_ponder

def test_ponder_forward_respects_step_cap():
    c = ModelConfig()
    m = PonderChessformer(c); m.eval()
    out = m(_batch(2), max_steps=1)
    assert out["ponder_steps"] == 1                  # clock budget caps depth
