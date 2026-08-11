import torch
import torch.nn.functional as F
from sahformer.model.heads import mdn_nll

def move_target_index(move_from, move_to, promo):
    """Vectorized 4352-class move index (matches heads.move_to_index).
    Non-promo: from*64+to. Promo: 4096 + to*4 + (promo-1)."""
    move_from = move_from.long()
    move_to = move_to.long()
    promo = promo.long()
    non_promo = move_from * 64 + move_to
    promo_idx = 4096 + move_to * 4 + (promo - 1).clamp_min(0)
    return torch.where(promo > 0, promo_idx, non_promo)

def move_accuracy(move_logits, target):
    """Top-1 accuracy. Logged sanity metric only — not a training objective."""
    return (move_logits.argmax(dim=-1) == target).float().mean().item()

def compute_losses(out, batch, w_policy: float = 1.0, w_value: float = 0.1,
                   w_time: float = 0.2):
    target = move_target_index(batch["move_from"], batch["move_to"], batch["promo"])
    policy = F.cross_entropy(out["move_logits"], target)
    value = F.cross_entropy(out["value_logits"], batch["result"].long())
    pi, mu, sigma = out["mdn"]
    time = mdn_nll(pi, mu, sigma, batch["think_time"].float())
    total = w_policy * policy + w_value * value + w_time * time
    return {"policy": policy, "value": value, "time": time, "total": total,
            "move_acc": move_accuracy(out["move_logits"], target)}
