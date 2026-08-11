import torch
from sahformer.model.heads import move_to_index
from sahformer.training.losses import move_target_index, move_accuracy, compute_losses

def test_move_target_index_matches_scalar():
    frm = torch.tensor([3, 0, 10, 20])
    to = torch.tensor([19, 0, 63, 40])
    promo = torch.tensor([0, 4, 1, 0])
    got = move_target_index(frm, to, promo)
    exp = torch.tensor([move_to_index(3, 19, 0), move_to_index(0, 0, 4),
                        move_to_index(10, 63, 1), move_to_index(20, 40, 0)])
    assert torch.equal(got, exp)

def test_move_accuracy_range_and_perfect():
    logits = torch.full((4, 4352), -10.0)
    target = torch.tensor([5, 5, 5, 5])
    logits[torch.arange(4), target] = 10.0
    assert move_accuracy(logits, target) == 1.0
    assert 0.0 <= move_accuracy(torch.randn(4, 4352), target) <= 1.0

def test_compute_losses_finite_and_backward():
    b = 3
    out = {"move_logits": torch.randn(b, 4352, requires_grad=True),
           "value_logits": torch.randn(b, 3, requires_grad=True),
           "mdn": (torch.randn(b, 3, requires_grad=True),
                   torch.randn(b, 3, requires_grad=True),
                   torch.randn(b, 3, requires_grad=True))}
    batch = {"move_from": torch.tensor([3, 0, 10]), "move_to": torch.tensor([19, 0, 63]),
             "promo": torch.tensor([0, 4, 1]), "result": torch.tensor([2, 1, 0]),
             "think_time": torch.tensor([1.5, 0.05, 12.0])}
    losses = compute_losses(out, batch)
    assert torch.isfinite(losses["total"])
    assert 0.0 <= losses["move_acc"] <= 1.0
    losses["total"].backward()
    assert out["move_logits"].grad is not None

def test_loss_weights_applied():
    b = 2
    out = {"move_logits": torch.randn(b, 4352, requires_grad=True),
           "value_logits": torch.randn(b, 3, requires_grad=True),
           "mdn": (torch.randn(b, 3, requires_grad=True),
                   torch.randn(b, 3, requires_grad=True),
                   torch.randn(b, 3, requires_grad=True))}
    batch = {"move_from": torch.tensor([3, 0]), "move_to": torch.tensor([19, 0]),
             "promo": torch.tensor([0, 4]), "result": torch.tensor([2, 1]),
             "think_time": torch.tensor([1.5, 0.05])}
    l = compute_losses(out, batch, w_policy=1.0, w_value=0.1, w_time=0.2)
    expected = l["policy"] + 0.1 * l["value"] + 0.2 * l["time"]
    assert torch.allclose(l["total"], expected)
