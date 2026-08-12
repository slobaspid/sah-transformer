import torch
import torch.nn.functional as F
from sahformer.model.heads import mdn_nll, mdn_nll_per_sample

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

def per_sample_loss(out, batch, w_policy: float = 1.0, w_value: float = 0.1,
                    w_time: float = 0.2):
    """Combined loss reduced per row -> (B,). Used to weight ponder steps by halt probability."""
    target = move_target_index(batch["move_from"], batch["move_to"], batch["promo"])
    policy = F.cross_entropy(out["move_logits"], target, reduction="none")
    value = F.cross_entropy(out["value_logits"], batch["result"].long(), reduction="none")
    pi, mu, sigma = out["mdn"]
    time = mdn_nll_per_sample(pi, mu, sigma, batch["think_time"].float())
    return w_policy * policy + w_value * value + w_time * time

def ponder_loss(pt_out, batch, prior_lambda: float = 0.4, beta: float = 0.01,
                w_policy: float = 1.0, w_value: float = 0.1, w_time: float = 0.2,
                min_steps: float = 0.0, floor_beta: float = 0.0, uniform: bool = False):
    """PonderNet loss: halt-probability-weighted task loss + KL to a geometric prior.

    Failsafes against halting collapse:
      - `uniform=True` (a warmup): weight all ponder steps equally, so the ponder block learns to
        refine *before* the halting is allowed to skip it.
      - `min_steps`/`floor_beta`: penalize the expected step count when it drops below `min_steps`,
        a hard floor against collapse-to-1.
    """
    steps, p = pt_out["steps"], pt_out["p_halt"]     # p: (B, K)
    K = p.shape[1]
    n_idx = torch.arange(K, device=p.device, dtype=torch.float32)
    step_losses = [per_sample_loss(steps[n], batch, w_policy, w_value, w_time) for n in range(K)]
    if uniform:
        task = torch.stack(step_losses, dim=1).mean()             # train every step equally
    else:
        task = sum(p[:, n] * step_losses[n] for n in range(K)).mean()
    prior = prior_lambda * (1.0 - prior_lambda) ** n_idx
    prior = (prior / prior.sum()).unsqueeze(0)
    kl = (p * (torch.log(p + 1e-9) - torch.log(prior + 1e-9))).sum(dim=1).mean()
    exp_steps = (p * (n_idx + 1)).sum(dim=1).mean()               # tensor (keeps gradient)
    floor = torch.relu(torch.tensor(float(min_steps), device=p.device) - exp_steps)
    total = task + beta * kl + floor_beta * floor
    return {"total": total, "task": task, "kl": kl, "avg_steps": exp_steps.item()}
