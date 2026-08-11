# Time Modulation on Faithful Backbone (Plan 3 v2 of 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

> **SUPERSEDES** `2026-08-11-time-modulation.md`. Builds the time layer on top of the faithful backbone from `2026-08-11-model-backbone-v2-faithful.md`, which already wired the `time_conditioned` flag through GAB and the `t`/`film` hooks through the encoder.

**Goal:** Make the faithful backbone clock-aware — feed the 21-dim temporal block through a `TemporalEncoder`, condition each layer's GAB on `t`, apply FiLM per block, and fuse `t` into the value/MDN heads. Prove the clock moves the outputs.

**Tech Stack:** Python 3.12 (`.venv`), PyTorch 2.4, pytest. Use `./.venv/Scripts/python.exe`.

**Depends on:** faithful backbone (Plan 2 v2) merged: `config.py` (has `t_ctx`, `temporal_dim`), `gab.py` (has `time_conditioned` flag + `t` arg), `encoder.py` (`Encoder(cfg, time_conditioned=)`, `forward(x, t=, film=)`), `heads.py`, `chessformer.py`.

**New files:**
```
sahformer/model/temporal.py     # TemporalEncoder, FiLMGenerator
sahformer/model/clockaware.py   # ClockAwareChessformer
tests/model/test_temporal.py
tests/model/test_clockaware.py
tests/model/test_clockaware_integration.py
```

---

## Task 1: TemporalEncoder + FiLMGenerator

**Files:** create `sahformer/model/temporal.py`, `tests/model/test_temporal.py`.

- [ ] **Step 1: Write test**

`tests/model/test_temporal.py`:
```python
import torch
from sahformer.model.config import ModelConfig
from sahformer.model.temporal import TemporalEncoder, FiLMGenerator

def test_temporal_encoder_shape():
    c = ModelConfig()
    out = TemporalEncoder(c)(torch.zeros(5, c.temporal_dim))
    assert out.shape == (5, c.t_ctx)

def test_temporal_varies():
    c = ModelConfig()
    enc = TemporalEncoder(c)
    assert not torch.allclose(enc(torch.zeros(1, c.temporal_dim)),
                              enc(torch.ones(1, c.temporal_dim)))

def test_film_shapes_and_identity():
    c = ModelConfig()
    gen = FiLMGenerator(c)
    gamma, beta = gen(torch.zeros(2, c.t_ctx))
    assert gamma.shape == (2, c.num_blocks, c.dim_vit)
    assert beta.shape == (2, c.num_blocks, c.dim_vit)
    assert torch.allclose(gamma, torch.ones_like(gamma), atol=0.2)
    assert torch.allclose(beta, torch.zeros_like(beta), atol=0.2)
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement** `sahformer/model/temporal.py`:
```python
import torch
import torch.nn as nn
from sahformer.model.config import ModelConfig

class TemporalEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.temporal_dim, cfg.t_ctx), nn.GELU(),
            nn.Linear(cfg.t_ctx, cfg.t_ctx),
        )
        self.ln = nn.LayerNorm(cfg.t_ctx)

    def forward(self, temporal):
        return self.ln(self.net(temporal.float()))

class FiLMGenerator(nn.Module):
    """Per-block per-channel (gamma, beta) from t; near identity at init."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n = cfg.num_blocks
        self.d = cfg.dim_vit
        self.gen = nn.Linear(cfg.t_ctx, 2 * cfg.num_blocks * cfg.dim_vit)
        nn.init.normal_(self.gen.weight, std=0.02)
        nn.init.zeros_(self.gen.bias)

    def forward(self, t):
        b = t.shape[0]
        raw = self.gen(t).view(b, self.n, 2, self.d)
        return 1.0 + raw[:, :, 0, :], raw[:, :, 1, :]
```

- [ ] **Step 4: Run, confirm pass. Step 5: Commit**

```bash
git add sahformer/model/temporal.py tests/model/test_temporal.py
git commit -m "feat: temporal encoder and FiLM generator on faithful backbone"
```

---

## Task 2: ClockAwareChessformer

**Files:** create `sahformer/model/clockaware.py`, `tests/model/test_clockaware.py`.

**Behavior:** like `FaithfulChessformer` but `Encoder(cfg, time_conditioned=True)`, encodes `t` from `batch["temporal"]`, passes `t` (GAB conditioning) and FiLM `(γ,β)` into `encoder.forward`, and fuses `t` into the pooled vector for the value and MDN heads via `Linear(t_ctx, dim_vit)`.

- [ ] **Step 1: Write test**

`tests/model/test_clockaware.py`:
```python
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
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement** `sahformer/model/clockaware.py`:
```python
import torch
import torch.nn as nn
from sahformer.model.config import ModelConfig
from sahformer.model.embedding import InputEmbedding
from sahformer.model.encoder import Encoder
from sahformer.model.heads import PolicyHead, ValueHead, ThinkTimeMDNHead
from sahformer.model.temporal import TemporalEncoder, FiLMGenerator

class ClockAwareChessformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_emb = InputEmbedding(cfg)
        self.temporal_enc = TemporalEncoder(cfg)
        self.film_gen = FiLMGenerator(cfg)
        self.encoder = Encoder(cfg, time_conditioned=True)
        self.policy = PolicyHead(cfg)
        self.value = ValueHead(cfg)
        self.think = ThinkTimeMDNHead(cfg)
        self.t_to_d = nn.Linear(cfg.t_ctx, cfg.dim_vit)

    def forward(self, batch: dict) -> dict:
        tok = self.input_emb(batch["board"].float(), batch["history"].float(),
                             batch["elo_self"], batch["elo_opp"])
        t = self.temporal_enc(batch["temporal"])
        film = self.film_gen(t)
        enc = self.encoder(tok, t=t, film=film)        # (B,64,dim_vit)
        pooled = enc.mean(dim=1) + self.t_to_d(t)
        return {
            "move_logits": self.policy(enc),
            "value_logits": self.value(enc),
            "mdn": self.think(pooled),
        }
```

Note: the value head applies its own norm+pool over `enc`; the MDN head consumes the
clock-fused `pooled`. This keeps the value head faithful while making think-time clock-aware.

- [ ] **Step 4: Run, confirm pass (report param count). If `test_time_changes_output` fails, the time path isn't reaching outputs — debug, don't weaken. Step 5: Commit**

```bash
git add sahformer/model/clockaware.py tests/model/test_clockaware.py
git commit -m "feat: ClockAwareChessformer on faithful backbone (time-conditioned GAB + FiLM)"
```

---

## Task 3: End-to-end + behavioral sanity

**Files:** create `tests/model/test_clockaware_integration.py`.

- [ ] **Step 1: Write test**

`tests/model/test_clockaware_integration.py`:
```python
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sahformer.records import PositionRecord
from sahformer.encoding import build_temporal
from sahformer.shards import records_to_arrays, save_shard
from sahformer.dataset import ShardDataset
from sahformer.model.config import ModelConfig
from sahformer.model.clockaware import ClockAwareChessformer
from sahformer.model.heads import move_to_index, mdn_nll

def _rec(temporal):
    return PositionRecord(
        board=np.zeros((8, 8, 12), np.int8), history=np.zeros((7, 8, 8, 12), np.int8),
        stm=0, elo_self=1500, elo_opp=1500, temporal=temporal.astype(np.float32),
        move_from=3, move_to=19, promo=0, result=2, think_time=1.5)

def test_end_to_end_and_time_sensitivity(tmp_path):
    calm = build_temporal(150.0, 150.0, [4.0], 8)
    panic = build_temporal(1.5, 80.0, [0.05, 0.05], 50)
    save_shard(str(tmp_path / "s.npz"), records_to_arrays([_rec(calm), _rec(panic)]))
    batch = next(iter(DataLoader(ShardDataset([str(tmp_path / "s.npz")]), batch_size=2)))
    model = ClockAwareChessformer(ModelConfig())
    out = model(batch)
    b = batch["move_from"].shape[0]
    target = torch.tensor([move_to_index(int(batch["move_from"][i]), int(batch["move_to"][i]),
                                         int(batch["promo"][i])) for i in range(b)])
    total = (F.cross_entropy(out["move_logits"], target)
             + 0.1 * F.cross_entropy(out["value_logits"], batch["result"])
             + 0.2 * mdn_nll(*out["mdn"], batch["think_time"]))
    assert torch.isfinite(total)
    total.backward()
    # behavioral: the two rows (calm vs panic clock) get different predicted think-time means
    model.eval()
    with torch.no_grad():
        mu = model(batch)["mdn"][1]
    assert not torch.allclose(mu[0], mu[1], atol=1e-6)
```

- [ ] **Step 2: Run it.** Fix real seam issues (don't weaken).

- [ ] **Step 3: Full suite** `./.venv/Scripts/python.exe -m pytest -q` — report total.

- [ ] **Step 4: Commit**

```bash
git add tests/model/test_clockaware_integration.py
git commit -m "test: clock-aware faithful model end-to-end + time sensitivity"
```

---

## Self-review notes

- **Reuses the faithful hooks:** GAB `time_conditioned=True` concatenates `t` into each layer's generator; encoder threads FiLM per block; MDN head consumes the clock-fused pooled vector. No backbone rewrite needed.
- **Kept faithful:** the value head still norms+pools `enc` itself (clock enters value only through the time-modulated trunk, not a head hack); policy stays the 4352 faithful head.
- **Time sensitivity is enforced** by `test_time_changes_output` and the integration behavioral check.
- **Placeholder scan:** none. **Consistency:** `FiLMGenerator` returns `(B,num_blocks,dim_vit)` matching `Encoder.forward`'s per-block slicing; `temporal` batch key matches `ShardDataset`.

## Open items for Plan 4 (training)

- Ablation flags: baseline = `FaithfulChessformer`; FiLM-only / GAB-only toggles for the curriculum.
- Loss weights policy 1.0 / value 0.1 / time 0.2–0.3; guard on move-accuracy vs baseline.
- AdamW + cyclic cosine, fp16, batch 128, checkpoint to Drive; needs real Lichess data (Plan 1 Task 7).
