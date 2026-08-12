# Clock-Gated Pondering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add adaptive computation ("pondering") to the model: it refines its read of the current position a variable number of internal steps, with a PonderNet-style halting gate fed by the clock context, so it thinks deeper with time and cuts thinking short under time pressure.

**Architecture:** A new `PonderChessformer` (built on the clock-aware backbone) runs a shared `PonderBlock` up to `max_ponder` times over the encoder output, with a `HaltHead` producing per-step halt probabilities from the pooled state + clock context. Training uses a PonderNet loss (probability-weighted per-step task loss + KL to a geometric prior). Inference is a drop-in: it halts early and returns the usual single output dict (+ `ponder_steps`), so `play`/`search`/engine work unchanged.

**Tech Stack:** Python 3.12 (`.venv`), PyTorch 2.4, pytest. Use `./.venv/Scripts/python.exe`.

**Design:** `docs/superpowers/specs/2026-08-12-clock-gated-pondering-design.md`.

**Reused facts:** value head order `[P(loss),P(draw),P(win)]`; `policy_difficulty(move_logits)` returns the `(B, think_extra)` difficulty features fed to the MDN head; the clock-aware backbone (`Encoder(time_conditioned=True)`, `TemporalEncoder`, `FiLMGenerator`) is unchanged.

---

## File structure

```
sahformer/model/
  config.py        # (+) max_ponder, ponder_prior
  ponder.py        # PonderBlock, HaltHead, PonderChessformer
sahformer/training/
  losses.py        # (+) mdn_nll_per_sample, per_sample_loss, ponder_loss
  build.py         # (+) mode "ponder"
  loop.py          # (+) ponder training branch
tests/model/
  test_ponder.py
tests/training/
  test_ponder_loss.py
  test_ponder_train.py
```

---

## Task 1: Config fields

**Files:** modify `sahformer/model/config.py`; modify `tests/model/test_config.py`.

- [ ] **Step 1: Write the failing test**

Append to `tests/model/test_config.py`:
```python
def test_ponder_defaults():
    c = ModelConfig()
    assert c.max_ponder == 4
    assert abs(c.ponder_prior - 0.4) < 1e-9
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_config.py::test_ponder_defaults -q`
Expected: FAIL — `ModelConfig has no attribute max_ponder`.

- [ ] **Step 3: Add the fields**

In `sahformer/model/config.py`, add to the dataclass (after `think_extra`):
```python
    max_ponder: int = 4           # max adaptive-computation (ponder) steps
    ponder_prior: float = 0.4     # geometric-prior halting rate for PonderNet
```

- [ ] **Step 4: Run to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sahformer/model/config.py tests/model/test_config.py
git commit -m "feat: config fields for pondering (max_ponder, ponder_prior)"
```

---

## Task 2: PonderBlock + HaltHead

**Files:** create `sahformer/model/ponder.py`, `tests/model/test_ponder.py`.

- [ ] **Step 1: Write the failing test**

`tests/model/test_ponder.py`:
```python
import torch
from sahformer.model.config import ModelConfig
from sahformer.model.ponder import PonderBlock, HaltHead

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
    t_a = torch.zeros(3, c.t_ctx)
    t_b = torch.ones(3, c.t_ctx)
    lam_a = hh(pooled, t_a)
    lam_b = hh(pooled, t_b)
    assert lam_a.shape == (3,)
    assert (lam_a >= 0).all() and (lam_a <= 1).all()
    assert not torch.allclose(lam_a, lam_b)          # halting depends on the clock context
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_ponder.py -q`
Expected: FAIL — `No module named 'sahformer.model.ponder'`.

- [ ] **Step 3: Implement the block + halt head**

`sahformer/model/ponder.py`:
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sahformer.model.config import ModelConfig
from sahformer.model.norm import RMSNorm
from sahformer.model.embedding import InputEmbedding
from sahformer.model.encoder import Encoder
from sahformer.model.temporal import TemporalEncoder, FiLMGenerator
from sahformer.model.heads import PolicyHead, ValueHead, ThinkTimeMDNHead, policy_difficulty

class _PonderMHA(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h = cfg.num_heads
        self.dh = cfg.head_dim
        self.qkv = nn.Linear(cfg.dim_vit, 3 * cfg.dim_vit, bias=False)
        self.out = nn.Linear(cfg.dim_vit, cfg.dim_vit, bias=False)

    def forward(self, x):
        b, s, _ = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.dh), dim=-1)
        ctx = (attn @ v).transpose(1, 2).reshape(b, s, self.h * self.dh)
        return self.out(ctx)

class PonderBlock(nn.Module):
    """One shared recurrent refinement step: pre-norm self-attention + MLP (no GAB)."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim_vit)
        self.attn = _PonderMHA(cfg)
        self.n2 = RMSNorm(cfg.dim_vit)
        hidden = cfg.dim_vit * cfg.mlp_ratio
        self.mlp = nn.Sequential(nn.Linear(cfg.dim_vit, hidden), nn.GELU(),
                                 nn.Linear(hidden, cfg.dim_vit))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x

class HaltHead(nn.Module):
    """P(halt at this step | reached it), from the pooled state + the clock context t."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.dim_vit + cfg.t_ctx, cfg.head_hid_dim), nn.ReLU(),
            nn.Linear(cfg.head_hid_dim, 1))

    def forward(self, pooled, t):
        return torch.sigmoid(self.net(torch.cat([pooled, t], dim=-1))).squeeze(-1)  # (B,)
```

- [ ] **Step 4: Run to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_ponder.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sahformer/model/ponder.py tests/model/test_ponder.py
git commit -m "feat: PonderBlock (recurrent refinement) + clock-fed HaltHead"
```

---

## Task 3: PonderChessformer (ponder_train + drop-in forward)

**Files:** modify `sahformer/model/ponder.py`; modify `tests/model/test_ponder.py`.

- [ ] **Step 1: Write the failing test**

Append to `tests/model/test_ponder.py`:
```python
from sahformer.model.ponder import PonderChessformer

def _batch(b=2, fill=0.0):
    return {"board": torch.zeros(b, 8, 8, 12), "history": torch.zeros(b, 7, 8, 8, 12),
            "elo_self": torch.tensor([1500] * b), "elo_opp": torch.tensor([1600] * b),
            "temporal": torch.full((b, 21), fill)}

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
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_ponder.py -q`
Expected: FAIL — `cannot import name 'PonderChessformer'`.

- [ ] **Step 3: Implement the model**

Append to `sahformer/model/ponder.py`:
```python
class PonderChessformer(nn.Module):
    """Clock-aware backbone + adaptive-computation ponder loop (PonderNet halting)."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_emb = InputEmbedding(cfg)
        self.temporal_enc = TemporalEncoder(cfg)
        self.film_gen = FiLMGenerator(cfg)
        self.encoder = Encoder(cfg, time_conditioned=True)
        self.ponder = PonderBlock(cfg)
        self.halt = HaltHead(cfg)
        self.policy = PolicyHead(cfg)
        self.value = ValueHead(cfg)
        self.think = ThinkTimeMDNHead(cfg)
        self.t_to_d = nn.Linear(cfg.t_ctx, cfg.dim_vit)

    def _encode(self, batch):
        tok = self.input_emb(batch["board"].float(), batch["history"].float(),
                             batch["elo_self"], batch["elo_opp"])
        t = self.temporal_enc(batch["temporal"])
        film = self.film_gen(t)
        enc = self.encoder(tok, t=t, film=film)
        return enc, t

    def _heads(self, h, t):
        pooled = h.mean(dim=1) + self.t_to_d(t)
        move_logits = self.policy(h)
        think_in = torch.cat([pooled, policy_difficulty(move_logits)], dim=-1)
        return {"move_logits": move_logits, "value_logits": self.value(h),
                "mdn": self.think(think_in)}

    def ponder_train(self, batch):
        """Per-step head outputs + halting distribution p_halt (B, K) for the PonderNet loss."""
        enc, t = self._encode(batch)
        K = self.cfg.max_ponder
        h = enc
        steps, lambdas = [], []
        for n in range(K):
            lam = self.halt(h.mean(dim=1), t)
            if n == K - 1:
                lam = torch.ones_like(lam)           # force halt at the last step
            lambdas.append(lam)
            steps.append(self._heads(h, t))
            if n < K - 1:
                h = self.ponder(h)
        p, remain = [], torch.ones_like(lambdas[0])
        for n in range(K):
            p.append(remain * lambdas[n])
            remain = remain * (1 - lambdas[n])
        return {"steps": steps, "p_halt": torch.stack(p, dim=1)}

    @torch.no_grad()
    def forward(self, batch, halt_threshold=0.5, max_steps=None):
        """Drop-in inference: run the ponder loop, halt early, return the usual output dict
        (+ ponder_steps). max_steps caps depth (the clock budget)."""
        enc, t = self._encode(batch)
        K = self.cfg.max_ponder if max_steps is None else min(int(max_steps), self.cfg.max_ponder)
        K = max(1, K)
        h = enc
        cum = torch.zeros(enc.shape[0], device=enc.device)
        out, used = None, 1
        for n in range(K):
            out = self._heads(h, t)
            cum = cum + self.halt(h.mean(dim=1), t)
            used = n + 1
            if float(cum.mean()) >= halt_threshold or n == K - 1:
                break
            h = self.ponder(h)
        out["ponder_steps"] = used
        return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_ponder.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sahformer/model/ponder.py tests/model/test_ponder.py
git commit -m "feat: PonderChessformer - ponder loop, PonderNet halting, drop-in inference"
```

---

## Task 4: PonderNet loss

**Files:** modify `sahformer/training/losses.py`; create `tests/training/test_ponder_loss.py`.

- [ ] **Step 1: Write the failing test**

`tests/training/test_ponder_loss.py`:
```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/training/test_ponder_loss.py -q`
Expected: FAIL — `cannot import name 'per_sample_loss'`.

- [ ] **Step 3: Implement the per-sample + ponder losses**

In `sahformer/training/losses.py`, first refactor `mdn_nll` to reuse a per-sample form. Replace the existing `mdn_nll` function with:
```python
def mdn_nll_per_sample(pi_logits, mu, sigma_param, target_time, eps: float = 1e-6):
    x = torch.log(target_time.clamp_min(eps)).unsqueeze(-1)
    log_pi = F.log_softmax(pi_logits, dim=-1)
    sigma = F.softplus(sigma_param) + 1e-3
    z = (x - mu) / sigma
    comp = -0.5 * z * z - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
    return -(torch.logsumexp(log_pi + comp, dim=-1))          # (B,)

def mdn_nll(pi_logits, mu, sigma_param, target_time, eps: float = 1e-6):
    return mdn_nll_per_sample(pi_logits, mu, sigma_param, target_time, eps).mean()
```
Then append the pondering losses:
```python
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
                w_policy: float = 1.0, w_value: float = 0.1, w_time: float = 0.2):
    """PonderNet loss: halt-probability-weighted task loss + KL to a geometric prior."""
    steps, p = pt_out["steps"], pt_out["p_halt"]     # p: (B, K)
    K = p.shape[1]
    task = 0.0
    for n in range(K):
        task = task + p[:, n] * per_sample_loss(steps[n], batch, w_policy, w_value, w_time)
    task = task.mean()
    n_idx = torch.arange(K, device=p.device, dtype=torch.float32)
    prior = prior_lambda * (1.0 - prior_lambda) ** n_idx
    prior = (prior / prior.sum()).unsqueeze(0)
    kl = (p * (torch.log(p + 1e-9) - torch.log(prior + 1e-9))).sum(dim=1).mean()
    total = task + beta * kl
    avg_steps = (p * (n_idx + 1)).sum(dim=1).mean().item()
    return {"total": total, "task": task, "kl": kl, "avg_steps": avg_steps}
```

- [ ] **Step 4: Run to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/training/test_ponder_loss.py tests/training/test_losses.py -q`
Expected: PASS (new ponder losses + existing loss tests, since `mdn_nll` behavior is unchanged).

- [ ] **Step 5: Commit**

```bash
git add sahformer/training/losses.py tests/training/test_ponder_loss.py
git commit -m "feat: PonderNet loss (per-sample weighting + KL geometric prior)"
```

---

## Task 5: build_model "ponder" mode + training-loop branch

**Files:** modify `sahformer/training/build.py`, `sahformer/training/loop.py`; create `tests/training/test_ponder_train.py`.

- [ ] **Step 1: Write the failing test**

`tests/training/test_ponder_train.py`:
```python
import numpy as np
from sahformer.records import PositionRecord
from sahformer.encoding import TEMPORAL_DIM
from sahformer.shards import records_to_arrays, save_shard
from sahformer.training.build import build_model, MODES
from sahformer.training.loop import TrainConfig, train

def _rec(seed):
    board = np.zeros((8, 8, 12), np.int8); board[seed % 8, 0, 0] = 1
    return PositionRecord(board=board, history=np.zeros((7, 8, 8, 12), np.int8),
        stm=0, elo_self=1500, elo_opp=1500, temporal=np.zeros(TEMPORAL_DIM, np.float32),
        move_from=3, move_to=19, promo=0, result=2, think_time=1.5)

def test_ponder_in_modes_and_builds():
    assert "ponder" in MODES
    from sahformer.model.config import ModelConfig
    m = build_model("ponder", ModelConfig())
    from sahformer.model.ponder import PonderChessformer
    assert isinstance(m, PonderChessformer)

def test_ponder_training_reduces_loss(tmp_path):
    save_shard(str(tmp_path / "s.npz"), records_to_arrays([_rec(0), _rec(1)]))
    cfg = TrainConfig(mode="ponder", max_steps=60, warmup_steps=5, batch_size=2,
                      lr=1e-3, out_dir=str(tmp_path / "ck"))
    res = train(cfg, [str(tmp_path / "s.npz")])
    assert res["history"][-1]["total"] < res["history"][0]["total"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/training/test_ponder_train.py -q`
Expected: FAIL — `"ponder" not in MODES` / build_model raises.

- [ ] **Step 3: Add the mode**

In `sahformer/training/build.py`, add the import and the mode:
```python
from sahformer.model.ponder import PonderChessformer
```
Change `MODES` and add the branch in `build_model`:
```python
MODES = ("baseline", "film_only", "gab_only", "full", "ponder")
```
```python
    if mode == "ponder":
        return PonderChessformer(cfg)
```
(place the `ponder` branch before the final `raise ValueError`.)

- [ ] **Step 4: Add the ponder branch to the training loop**

In `sahformer/training/loop.py`, add the import:
```python
from sahformer.training.losses import compute_losses, ponder_loss
```
(replace the existing `from sahformer.training.losses import compute_losses` line.)

In `train`, replace the forward+loss block:
```python
        with torch.autocast(device_type=dev_type, enabled=cfg.amp):
            out = model(batch)
            losses = compute_losses(out, batch, cfg.w_policy, cfg.w_value, cfg.w_time)
        scaler.scale(losses["total"]).backward()
```
with a mode-aware version:
```python
        with torch.autocast(device_type=dev_type, enabled=cfg.amp):
            if cfg.mode == "ponder":
                pt = model.ponder_train(batch)
                losses = ponder_loss(pt, batch, model_cfg.ponder_prior, 0.01,
                                     cfg.w_policy, cfg.w_value, cfg.w_time)
                losses.setdefault("policy", losses["task"])
                losses.setdefault("value", losses["kl"])
                losses.setdefault("time", torch.tensor(losses["avg_steps"]))
                losses.setdefault("move_acc", 0.0)
            else:
                out = model(batch)
                losses = compute_losses(out, batch, cfg.w_policy, cfg.w_value, cfg.w_time)
        scaler.scale(losses["total"]).backward()
```
(The `setdefault` lines just keep the existing history/logging keys populated; the ponder loss's real signals are `total`/`task`/`kl`/`avg_steps`.)

- [ ] **Step 5: Run to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/training/test_ponder_train.py -q`
Expected: PASS (ponder mode builds; a short ponder run reduces the loss).

- [ ] **Step 6: Commit**

```bash
git add sahformer/training/build.py sahformer/training/loop.py tests/training/test_ponder_train.py
git commit -m "feat: ponder training mode (build_model + PonderNet loop branch)"
```

---

## Task 6: Integration — pondering plays through play/search + full suite

**Files:** create `tests/model/test_ponder_integration.py`.

- [ ] **Step 1: Write the failing test**

`tests/model/test_ponder_integration.py`:
```python
import chess
from sahformer.model.config import ModelConfig
from sahformer.training.build import build_model
from sahformer.play import self_play
from sahformer.search import SearchConfig

def test_ponder_model_self_plays():
    model = build_model("ponder", ModelConfig())
    plies = list(self_play(model, max_plies=6, start_clock=100000.0, seed=0))
    assert len(plies) >= 1
    board = chess.Board()
    for rec in plies:
        assert rec["move"] in board.legal_moves
        board.push(rec["move"])

def test_ponder_model_with_search():
    model = build_model("ponder", ModelConfig())
    board = chess.Board()
    from sahformer.search import mcts_move
    move, _ = mcts_move(model, board, [], sims=6, cfg=SearchConfig(), seed=0)
    assert move in board.legal_moves
```

- [ ] **Step 2: Run to verify it fails or passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_ponder_integration.py -q`
Expected: PASS is likely (the ponder model's `forward` is a drop-in). If it FAILS, fix the real seam (e.g., `self_play`/`search` reading an output key the ponder forward doesn't provide) — do not weaken the test.

- [ ] **Step 3: Full suite**

Run: `./.venv/Scripts/python.exe -m pytest -q` — report total.

- [ ] **Step 4: Commit**

```bash
git add tests/model/test_ponder_integration.py
git commit -m "test: ponder model plays through self-play and search (drop-in)"
```

---

## Task 7: Manual sanity + notebook wiring (optional retrain)

**Files:** none (training on Kaggle) — documentation step.

- [ ] **Step 1: Note the training command**

In the Kaggle/Colab notebook, `mode="ponder"` trains a ponder model:
```python
cfg = TrainConfig(mode="ponder", max_steps=30000, warmup_steps=1500, batch_size=512,
                  lr=3e-4, amp=True, stream=True, device="cuda", out_dir=f"{OUT}/ponder")
res = train(cfg, DATA_SHARDS, model_cfg=MODEL)
```
Its checkpoint is self-describing, so the local viewer/engine load it automatically.

- [ ] **Step 2: Watch it play + read ponder depth**

After training, self-play with a ponder checkpoint; log `out["ponder_steps"]` across moves to see
whether it ponders more on complex positions and fewer under low clock. This is the qualitative
check; the rigorous **thought-depth-vs-Elo** measurement is the eval plan.

---

## Self-review notes

- **Spec coverage:** PonderBlock + HaltHead (Task 2); PonderChessformer with ponder loop, PonderNet halting, clock-fed halting, drop-in forward + `ponder_steps`, clock step-cap (Task 3); PonderNet loss with per-sample weighting + KL geometric prior (Task 4); `max_ponder`/`ponder_prior` config (Task 1); build/train integration (Task 5); play/search drop-in (Task 6). All design sections map to a task.
- **Halting sums to 1:** last-step `lambda` forced to 1, `p_n = lambda_n · Π_{m<n}(1-lambda_m)` — telescopes to 1 (Task 3 test).
- **Clock gate:** `HaltHead` consumes `t`; the behavioral "low clock → halt sooner" is a *learned* property (verified in eval), so Task 2 only asserts the halting *depends on* `t`.
- **Drop-in interface:** `PonderChessformer.forward` returns `move_logits`/`value_logits`/`mdn` (+ `ponder_steps`), matching what `play`/`search`/engine read (Task 6).
- **Type consistency:** `ponder_train`→`{steps, p_halt}`, `ponder_loss(pt_out, batch, prior_lambda, beta, w_*)`→`{total,task,kl,avg_steps}`, `per_sample_loss(out,batch,w_*)`→`(B,)` used identically across tasks.
- **Placeholder scan:** none.

## Open items for the eval plan

- Does pondering improve human-realism (move-match / timing on unseen games), or is it a wash?
- Correlate `ponder_steps` with position difficulty and with **Elo** (thought-depth-vs-Elo), alongside the search's sims-to-converge.
- Tune `max_ponder`, `ponder_prior`, `beta`, and the inference `halt_threshold` against human-alignment metrics.
