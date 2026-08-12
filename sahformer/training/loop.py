import os
import math
from dataclasses import dataclass, asdict
import torch
from torch.utils.data import DataLoader
from sahformer.model.config import ModelConfig
from sahformer.dataset import ShardDataset, StreamingShardDataset
from sahformer.training.build import build_model
from sahformer.training.losses import compute_losses

@dataclass
class TrainConfig:
    mode: str = "full"
    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 128
    max_steps: int = 1000
    warmup_steps: int = 50
    w_policy: float = 1.0
    w_value: float = 0.1
    w_time: float = 0.2
    amp: bool = False
    stream: bool = False          # True: stream shards from disk (for datasets too big for RAM)
    device: str = "cpu"
    seed: int = 0
    out_dir: str = "checkpoints"
    log_every: int = 50
    ckpt_every: int = 200

def _lr_scale(step, warmup, total):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

def save_checkpoint(path, model, cfg, step, best_metric, model_cfg=None):
    torch.save({"model_state": model.state_dict(), "cfg": asdict(cfg),
                "model_cfg": asdict(model_cfg) if model_cfg is not None else None,
                "step": step, "best_metric": best_metric}, path)

def load_checkpoint(path, model):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return ckpt

def load_model(path, mode=None):
    """Rebuild the exact model a checkpoint was trained with (any size) and load its weights.
    Reads the saved model_cfg so it works even if the model was trained bigger than default."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    mc = ckpt.get("model_cfg")
    model_cfg = ModelConfig(**mc) if mc else ModelConfig()
    if mode is None:
        mode = ckpt.get("cfg", {}).get("mode", "full")
    model = build_model(mode, model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, model_cfg

def _infinite(loader):
    while True:
        for batch in loader:
            yield batch

def train(cfg: TrainConfig, shard_paths, model_cfg: ModelConfig = None):
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    model_cfg = model_cfg or ModelConfig()
    model = build_model(cfg.mode, model_cfg).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _lr_scale(s, cfg.warmup_steps, cfg.max_steps))
    dev_type = "cuda" if "cuda" in cfg.device else "cpu"
    scaler = torch.amp.GradScaler(dev_type, enabled=cfg.amp)

    if cfg.stream:
        ds = StreamingShardDataset(shard_paths, shuffle=True, seed=cfg.seed)
        loader = DataLoader(ds, batch_size=cfg.batch_size, drop_last=False)
    else:
        loader = DataLoader(ShardDataset(shard_paths), batch_size=cfg.batch_size,
                            shuffle=True, drop_last=False)
    data = _infinite(loader)
    model.train()
    best = float("inf")
    history = []
    for step in range(cfg.max_steps):
        batch = next(data)
        batch = {k: (v.to(cfg.device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev_type, enabled=cfg.amp):
            out = model(batch)
            losses = compute_losses(out, batch, cfg.w_policy, cfg.w_value, cfg.w_time)
        scaler.scale(losses["total"]).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()

        total = losses["total"].item()
        rec = {"step": step, "lr": sched.get_last_lr()[0], "total": total,
               "policy": losses["policy"].item(), "value": losses["value"].item(),
               "time": losses["time"].item(), "move_acc": losses["move_acc"]}
        history.append(rec)
        if (step + 1) % cfg.log_every == 0:
            print(f"step {step+1}/{cfg.max_steps} total={total:.4f} "
                  f"policy={rec['policy']:.4f} time={rec['time']:.4f} acc={rec['move_acc']:.3f}")
        if total < best:
            best = total
            save_checkpoint(os.path.join(cfg.out_dir, "best.pt"), model, cfg, step, best,
                            model_cfg=model_cfg)
        if (step + 1) % cfg.ckpt_every == 0:
            save_checkpoint(os.path.join(cfg.out_dir, "last.pt"), model, cfg, step, best,
                            model_cfg=model_cfg)

    save_checkpoint(os.path.join(cfg.out_dir, "last.pt"), model, cfg, cfg.max_steps - 1, best,
                    model_cfg=model_cfg)
    return {"history": history, "best": best, "model": model}
