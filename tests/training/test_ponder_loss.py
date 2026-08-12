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
