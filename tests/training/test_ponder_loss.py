import torch
from sahformer.training.losses import per_sample_loss, ponder_loss

def _step(b=3):
    return {"move_logits": torch.randn(b, 4352, requires_grad=True),
            "value_logits": torch.randn(b, 3, requires_grad=True),
            "mdn": (torch.randn(b, 3, requires_grad=True),
                    torch.randn(b, 3, requires_grad=True),
                    torch.randn(b, 3, requires_grad=True))}

def _batch(b=3):
    return {"move_from": torch.tensor([3, 0, 10]), "move_to": torch.tensor([19, 0, 63]),
            "promo": torch.tensor([0, 4, 1]), "result": torch.tensor([2, 1, 0]),
            "think_time": torch.tensor([1.5, 0.05, 12.0])}

def test_per_sample_loss_shape():
    l = per_sample_loss(_step(3), _batch(3))
    assert l.shape == (3,)

def test_ponder_loss_finite_and_backward():
    K = 4
    steps = [_step(3) for _ in range(K)]
    p = torch.softmax(torch.randn(3, K), dim=1)      # a valid halting distribution
    out = ponder_loss({"steps": steps, "p_halt": p}, _batch(3), prior_lambda=0.4, beta=0.01)
    assert torch.isfinite(out["total"])
    assert 1.0 <= out["avg_steps"] <= K
    out["total"].backward()
    assert steps[0]["move_logits"].grad is not None

def test_ponder_floor_penalizes_collapse():
    K = 4
    steps = [_step(3) for _ in range(K)]
    p = torch.tensor([[0.9, 0.05, 0.03, 0.02]] * 3)  # concentrated on step 0 -> ~1 avg step
    hi = ponder_loss({"steps": steps, "p_halt": p}, _batch(3), min_steps=3.0, floor_beta=1.0)
    lo = ponder_loss({"steps": steps, "p_halt": p}, _batch(3), min_steps=3.0, floor_beta=0.0)
    assert hi["total"] > lo["total"]                 # floor adds cost when avg_steps < min_steps

def test_ponder_uniform_warmup_runs():
    K = 4
    steps = [_step(3) for _ in range(K)]
    p = torch.softmax(torch.randn(3, K), dim=1)
    out = ponder_loss({"steps": steps, "p_halt": p}, _batch(3), uniform=True)
    assert torch.isfinite(out["total"])
