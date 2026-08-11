# Training Pipeline Implementation Plan (Plan 4 of 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A trainable pipeline that fits our model on a tiny real 3+0 shard, logs move-accuracy and think-time NLL, and supports the baseline-vs-clock-aware ablation — all smoke-tested, the same code scaling to a full run later.

**Architecture:** A bounded streaming fetch produces one small real shard (`data/`). A `sahformer/training/` package holds `build_model` (4 ablation modes), the loss/metric functions, and an AdamW + warmup→cosine training loop with checkpointing. Two thin CLI scripts wrap fetch and train. Everything is TDD; the loop's "it learns" guarantee is an overfit test.

**Tech Stack:** Python 3.12 (`.venv`), PyTorch 2.4, `zstandard`, `python-chess`, pytest. Use `./.venv/Scripts/python.exe`.

**Design:** `docs/superpowers/specs/2026-08-11-training-pipeline-design.md`.

**Dependencies (already merged on `master`):** Plan 1 data pipeline (`records.py`, `shards.py`, `dataset.py`, `download.py`, `encoding.py`); Plan 2 v2 / Plan 3 v2 models (`FaithfulChessformer`, `ClockAwareChessformer`, `heads.move_to_index`, `mdn_nll`).

**Note on move-accuracy (from the design):** `move_accuracy` is a **logged sanity metric only**. The best checkpoint is chosen by **lowest total loss**, never by top-1 accuracy. Non-deterministic/sampled play is deferred to the eval plan.

---

## File structure

```
sahformer/
  download.py            # (+) _iter_games_from_binary, stream_games_from_url
  model/clockaware.py    # (+) use_film / use_time_gab flags
  training/
    __init__.py
    build.py             # MODES, build_model(mode, cfg)
    losses.py            # move_target_index, move_accuracy, compute_losses
    loop.py              # TrainConfig, _lr_scale, save/load_checkpoint, train
scripts/
  fetch_sample.py        # bounded streaming fetch -> data/sample.npz
  train.py               # training CLI
tests/
  test_download_stream.py
  training/
    __init__.py
    test_build.py
    test_losses.py
    test_loop.py
    test_scripts.py
```

---

## Task 0: Package scaffold

**Files:** create `sahformer/training/__init__.py`, `tests/training/__init__.py` (both empty).

- [ ] **Step 1: Create empty package markers**

Create `sahformer/training/__init__.py` with a single line:
```python
```
Create `tests/training/__init__.py` with a single line:
```python
```

- [ ] **Step 2: Verify collection**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: the existing 45 tests still pass.

- [ ] **Step 3: Commit**

```bash
git add sahformer/training/__init__.py tests/training/__init__.py
git commit -m "chore: scaffold training package"
```

---

## Task 1: Streaming downloader (shared zst core + URL stream)

**Files:** modify `sahformer/download.py`; create `tests/test_download_stream.py`.

**Behavior:** factor the zstd-stream parsing into one core `_iter_games_from_binary(reader)` used by the existing `iter_games_from_zst(path)` and a new `stream_games_from_url(url)`. No behavior change to the local-file path. The consumer early-stops, so a URL stream only transfers the first chunk.

- [ ] **Step 1: Write the failing test**

`tests/test_download_stream.py`:
```python
import io
import zstandard as zstd
from sahformer.download import _iter_games_from_binary, is_target_game

_PGN = (
    '[Event "A"]\n[TimeControl "180+0"]\n[WhiteElo "1500"]\n[BlackElo "1500"]\n\n'
    '1. e4 { [%clk 0:03:00] } e5 { [%clk 0:03:00] } *\n\n'
    '[Event "B"]\n[TimeControl "600+0"]\n[WhiteElo "1500"]\n[BlackElo "1500"]\n\n'
    '1. d4 d5 *\n\n'
)

def test_iter_games_from_binary_parses_zst_stream():
    comp = zstd.ZstdCompressor().compress(_PGN.encode("utf-8"))
    games = list(_iter_games_from_binary(io.BytesIO(comp)))
    assert len(games) == 2
    # first is a 3+0 game with clocks; second is 10+0 (filtered out)
    assert is_target_game(games[0])
    assert not is_target_game(games[1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_download_stream.py -v`
Expected: FAIL — `cannot import name '_iter_games_from_binary'`.

- [ ] **Step 3: Refactor `download.py`**

Replace the body of `sahformer/download.py` below the `is_target_game` / `iter_games_from_text` definitions. Keep `TARGET_TC`, `is_target_game`, and `iter_games_from_text` exactly as they are; replace `iter_games_from_zst` and add the two new functions:

```python
def _iter_games_from_binary(reader):
    """Decompress a zstd binary stream (file-like with .read) and yield games."""
    import zstandard as zstd
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(reader) as decompressed:
        text_stream = io.TextIOWrapper(decompressed, encoding="utf-8")
        while True:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                return
            yield game

def iter_games_from_zst(path: str):
    """Stream games from a local Lichess .pgn.zst file."""
    with open(path, "rb") as fh:
        yield from _iter_games_from_binary(fh)

def stream_games_from_url(url: str):
    """Stream games directly from a Lichess .pgn.zst URL, decompressing on the fly.
    The consumer should early-stop; only the first chunk of the archive is transferred."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "sahformer/0.1"})
    with urllib.request.urlopen(req) as resp:
        yield from _iter_games_from_binary(resp)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_download_stream.py tests/test_download.py -v`
Expected: PASS (new stream test + the existing download tests unchanged).

- [ ] **Step 5: Commit**

```bash
git add sahformer/download.py tests/test_download_stream.py
git commit -m "feat: shared zst parsing core + stream_games_from_url"
```

---

## Task 2: Ablation flags on ClockAwareChessformer + build_model

**Files:** modify `sahformer/model/clockaware.py`; create `sahformer/training/build.py`, `tests/training/test_build.py`.

**Behavior:** two constructor flags on `ClockAwareChessformer` express the middle ablation modes. `use_time_gab` selects `Encoder(time_conditioned=…)` and whether `t` reaches the encoder; `use_film` gates whether FiLM is applied. Defaults `(True, True)` keep the existing behavior (and existing tests) intact. `build_model` maps a mode string to the right module.

- [ ] **Step 1: Write the failing test**

`tests/training/test_build.py`:
```python
import pytest
import torch
from sahformer.model.config import ModelConfig
from sahformer.training.build import build_model, MODES

def _batch(b=2, fill=0.5):
    return {"board": torch.zeros(b, 8, 8, 12), "history": torch.zeros(b, 7, 8, 8, 12),
            "elo_self": torch.tensor([1500] * b), "elo_opp": torch.tensor([1600] * b),
            "temporal": torch.full((b, 21), fill)}

def test_modes_list():
    assert MODES == ("baseline", "film_only", "gab_only", "full")

def test_all_modes_build_forward_backward():
    c = ModelConfig()
    for m in MODES:
        model = build_model(m, c)
        out = model(_batch(2))
        assert out["move_logits"].shape == (2, 4352)
        assert out["value_logits"].shape == (2, 3)
        assert out["mdn"][0].shape == (2, c.mdn_components)
        out["move_logits"].sum().backward()

def test_baseline_has_no_temporal_params():
    names = [n for n, _ in build_model("baseline", ModelConfig()).named_parameters()]
    assert not any(("temporal_enc" in n) or ("film_gen" in n) for n in names)

def test_gab_only_has_no_film_params():
    names = [n for n, _ in build_model("gab_only", ModelConfig()).named_parameters()]
    assert not any("film_gen" in n for n in names)

def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        build_model("nope", ModelConfig())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/training/test_build.py -v`
Expected: FAIL — `No module named 'sahformer.training.build'`.

- [ ] **Step 3: Add flags to `clockaware.py`**

Replace `sahformer/model/clockaware.py` with:
```python
import torch
import torch.nn as nn
from sahformer.model.config import ModelConfig
from sahformer.model.embedding import InputEmbedding
from sahformer.model.encoder import Encoder
from sahformer.model.heads import PolicyHead, ValueHead, ThinkTimeMDNHead
from sahformer.model.temporal import TemporalEncoder, FiLMGenerator

class ClockAwareChessformer(nn.Module):
    def __init__(self, cfg: ModelConfig, use_film: bool = True, use_time_gab: bool = True):
        super().__init__()
        self.cfg = cfg
        self.use_film = use_film
        self.use_time_gab = use_time_gab
        self.input_emb = InputEmbedding(cfg)
        self.temporal_enc = TemporalEncoder(cfg)
        self.film_gen = FiLMGenerator(cfg) if use_film else None
        self.encoder = Encoder(cfg, time_conditioned=use_time_gab)
        self.policy = PolicyHead(cfg)
        self.value = ValueHead(cfg)
        self.think = ThinkTimeMDNHead(cfg)
        self.t_to_d = nn.Linear(cfg.t_ctx, cfg.dim_vit)

    def forward(self, batch: dict) -> dict:
        tok = self.input_emb(batch["board"].float(), batch["history"].float(),
                             batch["elo_self"], batch["elo_opp"])
        t = self.temporal_enc(batch["temporal"])
        film = self.film_gen(t) if self.use_film else None
        enc = self.encoder(tok, t=(t if self.use_time_gab else None), film=film)
        pooled = enc.mean(dim=1) + self.t_to_d(t)
        return {
            "move_logits": self.policy(enc),
            "value_logits": self.value(enc),
            "mdn": self.think(pooled),
        }
```

- [ ] **Step 4: Create `sahformer/training/build.py`**

```python
from sahformer.model.config import ModelConfig
from sahformer.model.chessformer import FaithfulChessformer
from sahformer.model.clockaware import ClockAwareChessformer

MODES = ("baseline", "film_only", "gab_only", "full")

def build_model(mode: str, cfg: ModelConfig):
    """Return the model for an ablation mode (see the design's ablation table)."""
    if mode == "baseline":
        return FaithfulChessformer(cfg)
    if mode == "film_only":
        return ClockAwareChessformer(cfg, use_film=True, use_time_gab=False)
    if mode == "gab_only":
        return ClockAwareChessformer(cfg, use_film=False, use_time_gab=True)
    if mode == "full":
        return ClockAwareChessformer(cfg, use_film=True, use_time_gab=True)
    raise ValueError(f"unknown mode: {mode!r} (expected one of {MODES})")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/training/test_build.py tests/model/test_clockaware.py -v`
Expected: PASS (new ablation tests + the unchanged clockaware tests — defaults preserve old behavior).

- [ ] **Step 6: Commit**

```bash
git add sahformer/model/clockaware.py sahformer/training/build.py tests/training/test_build.py
git commit -m "feat: ablation flags (use_film/use_time_gab) + build_model"
```

---

## Task 3: Losses & metrics

**Files:** create `sahformer/training/losses.py`, `tests/training/test_losses.py`.

**Behavior:** `move_target_index` is the vectorized form of `heads.move_to_index`. `compute_losses` returns policy CE (4352), value CE (W/D/L), MDN time NLL, their weighted `total`, and the `move_acc` sanity metric.

- [ ] **Step 1: Write the failing test**

`tests/training/test_losses.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/training/test_losses.py -v`
Expected: FAIL — `No module named 'sahformer.training.losses'`.

- [ ] **Step 3: Implement `sahformer/training/losses.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/training/test_losses.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sahformer/training/losses.py tests/training/test_losses.py
git commit -m "feat: training losses, move-index, and move-accuracy metric"
```

---

## Task 4: Training loop + checkpointing

**Files:** create `sahformer/training/loop.py`, `tests/training/test_loop.py`.

**Behavior:** `TrainConfig` holds all knobs. `train` builds the model via `build_model`, runs AdamW with a linear-warmup→cosine schedule over an (infinitely re-iterated) `DataLoader`, records a per-step metric history, saves `best.pt` (lowest total loss) and periodic `last.pt`. AMP is a no-op passthrough when `amp=False`. The overfit test is the "it actually learns" guarantee.

- [ ] **Step 1: Write the failing test**

`tests/training/test_loop.py`:
```python
import numpy as np
import torch
from sahformer.records import PositionRecord
from sahformer.encoding import TEMPORAL_DIM
from sahformer.shards import records_to_arrays, save_shard
from sahformer.model.config import ModelConfig
from sahformer.training.build import build_model
from sahformer.training.loop import TrainConfig, train, save_checkpoint, load_checkpoint

def _rec(seed, mf, mt):
    board = np.zeros((8, 8, 12), np.int8)
    board[seed % 8, 0, 0] = 1  # distinct inputs so the batch is cleanly fittable
    return PositionRecord(
        board=board, history=np.zeros((7, 8, 8, 12), np.int8),
        stm=0, elo_self=1500, elo_opp=1500, temporal=np.zeros(TEMPORAL_DIM, np.float32),
        move_from=mf, move_to=mt, promo=0, result=2, think_time=1.5)

def _shard(tmp_path):
    save_shard(str(tmp_path / "s.npz"), records_to_arrays([_rec(0, 3, 19), _rec(1, 8, 16)]))
    return [str(tmp_path / "s.npz")]

def test_overfit_decreases_loss(tmp_path):
    cfg = TrainConfig(mode="full", max_steps=80, warmup_steps=5, batch_size=2,
                      lr=1e-3, out_dir=str(tmp_path / "ck"))
    res = train(cfg, _shard(tmp_path))
    first = res["history"][0]["total"]
    last = res["history"][-1]["total"]
    assert last < first, (first, last)

def test_train_writes_checkpoints(tmp_path):
    cfg = TrainConfig(mode="baseline", max_steps=10, warmup_steps=2, batch_size=2,
                      ckpt_every=5, out_dir=str(tmp_path / "ck"))
    train(cfg, _shard(tmp_path))
    assert (tmp_path / "ck" / "last.pt").exists()
    assert (tmp_path / "ck" / "best.pt").exists()

def test_checkpoint_roundtrip(tmp_path):
    cfg = TrainConfig(mode="baseline", max_steps=6, warmup_steps=1, batch_size=2,
                      out_dir=str(tmp_path / "ck"))
    res = train(cfg, _shard(tmp_path))
    fresh = build_model("baseline", ModelConfig())
    load_checkpoint(str(tmp_path / "ck" / "last.pt"), fresh)
    for (_, p1), (_, p2) in zip(res["model"].named_parameters(), fresh.named_parameters()):
        assert torch.allclose(p1.detach().cpu(), p2.detach().cpu())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/training/test_loop.py -v`
Expected: FAIL — `No module named 'sahformer.training.loop'`.

- [ ] **Step 3: Implement `sahformer/training/loop.py`**

```python
import os
import math
from dataclasses import dataclass, asdict
import torch
from torch.utils.data import DataLoader
from sahformer.model.config import ModelConfig
from sahformer.dataset import ShardDataset
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

def save_checkpoint(path, model, cfg, step, best_metric):
    torch.save({"model_state": model.state_dict(), "cfg": asdict(cfg),
                "step": step, "best_metric": best_metric}, path)

def load_checkpoint(path, model):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    return ckpt

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
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

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
            save_checkpoint(os.path.join(cfg.out_dir, "best.pt"), model, cfg, step, best)
        if (step + 1) % cfg.ckpt_every == 0:
            save_checkpoint(os.path.join(cfg.out_dir, "last.pt"), model, cfg, step, best)

    save_checkpoint(os.path.join(cfg.out_dir, "last.pt"), model, cfg, cfg.max_steps - 1, best)
    return {"history": history, "best": best, "model": model}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/training/test_loop.py -v`
Expected: PASS. If `test_overfit_decreases_loss` fails, the loop isn't learning — debug (LR, backward, schedule), do not weaken the assertion.

- [ ] **Step 5: Commit**

```bash
git add sahformer/training/loop.py tests/training/test_loop.py
git commit -m "feat: AdamW warmup->cosine training loop with checkpointing"
```

---

## Task 5: CLI scripts (fetch + train)

**Files:** create `scripts/fetch_sample.py`, `scripts/train.py`, `tests/training/test_scripts.py`.

**Behavior:** `fetch_sample.py` streams a URL, keeps up to `--max-games` target games, balances by Elo, and writes `data/sample.npz`. `train.py` globs shards and runs `train`. The test drives `train.py`'s `main()` on a tiny shard (no network).

- [ ] **Step 1: Write the failing test**

`tests/training/test_scripts.py`:
```python
import sys
import importlib.util
import numpy as np
from sahformer.records import PositionRecord
from sahformer.encoding import TEMPORAL_DIM
from sahformer.shards import records_to_arrays, save_shard

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _rec():
    return PositionRecord(
        board=np.zeros((8, 8, 12), np.int8), history=np.zeros((7, 8, 8, 12), np.int8),
        stm=0, elo_self=1500, elo_opp=1500, temporal=np.zeros(TEMPORAL_DIM, np.float32),
        move_from=3, move_to=19, promo=0, result=2, think_time=1.5)

def test_train_cli_runs(tmp_path, monkeypatch):
    save_shard(str(tmp_path / "s.npz"), records_to_arrays([_rec(), _rec()]))
    mod = _load("train_cli", "scripts/train.py")
    monkeypatch.setattr(sys, "argv", [
        "train.py", str(tmp_path / "s.npz"), "--mode", "baseline",
        "--max-steps", "3", "--batch-size", "2", "--out", str(tmp_path / "ck")])
    mod.main()
    assert (tmp_path / "ck" / "last.pt").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/training/test_scripts.py -v`
Expected: FAIL — `scripts/train.py` does not exist.

- [ ] **Step 3: Create `scripts/fetch_sample.py`**

```python
"""Stream a tiny 3+0 sample from a Lichess .pgn.zst URL into one balanced shard.

Only the first chunk of the archive is transferred (we early-stop at --max-games).

Usage:
    python scripts/fetch_sample.py URL data/ --max-games 2000 --seed 0
"""
import argparse
import os
from sahformer.download import stream_games_from_url, is_target_game
from sahformer.records import game_to_records
from sahformer.shards import records_to_arrays, balance_indices, save_shard

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("outdir")
    ap.add_argument("--max-games", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    records = []
    kept = 0
    for game in stream_games_from_url(args.url):
        if not is_target_game(game):
            continue
        records.extend(game_to_records(game))
        kept += 1
        if kept >= args.max_games:
            break

    if kept == 0:
        raise SystemExit("no target (3+0 with clocks) games found — try a 2017-04+ month")
    arr = records_to_arrays(records)
    idx = balance_indices(arr["elo_self"], seed=args.seed)
    balanced = {k: v[idx] for k, v in arr.items()}
    out = os.path.join(args.outdir, "sample.npz")
    save_shard(out, balanced)
    print(f"target_games={kept} positions_kept={len(idx)} -> {out}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create `scripts/train.py`**

```python
"""Train a (clock-aware) Chessformer on shard(s).

Usage:
    python scripts/train.py "data/*.npz" --mode full --max-steps 2000 --out checkpoints/
"""
import argparse
import glob
from sahformer.training.loop import TrainConfig, train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", help="glob for .npz shard(s)")
    ap.add_argument("--mode", default="full",
                    choices=["baseline", "film_only", "gab_only", "full"])
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.shards))
    if not paths:
        raise SystemExit(f"no shards matched: {args.shards}")
    cfg = TrainConfig(mode=args.mode, max_steps=args.max_steps, batch_size=args.batch_size,
                      lr=args.lr, out_dir=args.out, amp=args.amp, device=args.device)
    res = train(cfg, paths)
    print(f"done. best_total={res['best']:.4f} steps={len(res['history'])}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/training/test_scripts.py -v`
Expected: PASS.

- [ ] **Step 6: Full suite + commit**

Run: `./.venv/Scripts/python.exe -m pytest -q` — report total.
```bash
git add scripts/fetch_sample.py scripts/train.py tests/training/test_scripts.py
git commit -m "feat: fetch-sample and train CLI scripts"
```

---

## Task 6: Fetch a tiny real sample + smoke train (manual, needs confirmation)

**Files:** none (produces `data/sample.npz` and `checkpoints/`).

> **Download = explicit-permission action.** Confirm the exact URL and observed size with the user before running Step 1. The streaming fetch early-stops, so only the first chunk transfers, but still ask first.

- [ ] **Step 1: Fetch a small real shard**

Lichess monthly dumps live at `database.lichess.org`. Clocks exist from **2017-04 onward**. Candidate URL:
```
https://database.lichess.org/standard/lichess_db_standard_rated_2017-04.pgn.zst
```
Run (pull ~2000 target games):
```bash
./.venv/Scripts/python.exe scripts/fetch_sample.py https://database.lichess.org/standard/lichess_db_standard_rated_2017-04.pgn.zst data/ --max-games 2000
```
Expected: prints `target_games=2000 positions_kept=<N> -> data/sample.npz`.
If it prints the `no target games` error, the month lacks clocks — retry with a later month (e.g. `2019-01`).

- [ ] **Step 2: Smoke-train each ablation briefly**

```bash
./.venv/Scripts/python.exe scripts/train.py data/sample.npz --mode baseline --max-steps 200 --batch-size 64 --out checkpoints/baseline
./.venv/Scripts/python.exe scripts/train.py data/sample.npz --mode full --max-steps 200 --batch-size 64 --out checkpoints/full
```
Expected: loss trends down over the run; `best.pt`/`last.pt` written under each `checkpoints/<mode>/`.

- [ ] **Step 3: Record the result**

Note the final `best_total` for `baseline` vs `full` and the `move_acc`/`time` trajectory in the run output. (A rigorous baseline-vs-full comparison is the eval plan; this is only a "the machinery trains on real data" check.)

- [ ] **Step 4: Ignore data & checkpoints in git**

Add to `.gitignore` (create if missing):
```
data/
checkpoints/
```
```bash
git add .gitignore
git commit -m "chore: ignore data/ and checkpoints/"
```

---

## Self-review notes

- **Spec coverage:** streaming fetch (Task 1 + fetch_sample in Task 5), 4-mode ablation (Task 2), losses/metrics with move-acc-as-sanity-only (Task 3), AdamW warmup→cosine loop + save/load checkpoint choosing best by total loss (Task 4), CLIs (Task 5), tiny-real-sample proof (Task 6). All design sections map to a task.
- **Non-determinism guard:** `move_accuracy` is only logged; `best.pt` is selected by lowest `total`, never accuracy — consistent with the design's deferred-stochasticity note.
- **Backward-compat:** `ClockAwareChessformer` flag defaults `(True, True)` reproduce the merged behavior, so `tests/model/test_clockaware*.py` keep passing.
- **Type/name consistency:** `build_model(mode, cfg)`, `MODES`, `compute_losses(out, batch, w_policy, w_value, w_time)` returning keys `{policy,value,time,total,move_acc}`, `TrainConfig` fields, and `save_checkpoint(path, model, cfg, step, best_metric)` / `load_checkpoint(path, model)` are used identically across tasks and scripts.
- **Placeholder scan:** none.

## Open items for Plan 5 (evaluation / play)

- Sampled, non-deterministic play (temperature over policy; sample the MDN for timing) — the deferred stochasticity discussion.
- Eval-vs-Maia: move-match rate by Elo bucket, think-time distribution calibration, baseline-vs-full ablation comparison on held-out games.
- Full-scale training run (many shards, GPU/AMP, Colab/Drive checkpoints).
