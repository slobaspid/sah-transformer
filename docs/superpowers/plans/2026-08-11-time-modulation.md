# Time Modulation: FiLM + Time-Conditioned GAB (Plan 3 of 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Plan 2 Chessformer baseline clock-aware — inject the `temporal` feature vector so the clock reshapes how the board is read (FiLM on every block + a time-conditioned Geometric Attention Bias), and route time into the value and think-time heads.

**Architecture:** A `TemporalEncoder` maps the 21-dim temporal vector to a context vector `t`. `t` drives two modulation paths: (1) a `FiLMGenerator` produces per-layer per-channel `(γ, β)` applied to each transformer block's output, and (2) a low-rank `GeometricAttentionBias` builds a per-example `(B, n_heads, 66, 66)` additive attention bias from a board summary + `t` + the skill embeddings, fed through the encoder's existing `attn_bias` hook. `t` is also fused into the pooled vector so the value and MDN think-time heads are clock-aware. A new `ClockAwareChessformer` assembles this; the Plan 2 `Chessformer` stays as the clock-blind baseline for ablation.

**Tech Stack:** Python 3.12 (`.venv`), PyTorch 2.4, pytest. All commands use `./.venv/Scripts/python.exe`.

**Why low-rank GAB:** the paper's dense GAB projects a conditioning vector to `n_heads × 64 × 64 = 32768` outputs — from a 128-dim input that alone is ~4.2M params, the entire 5M budget. We instead produce rank-`r` per-square vectors `u, v ∈ (B, h, 64, r)` and form the bias as `bias[b,h,i,j] = (u_i · v_j)/√r`. With `r=4` this is ~0.5M params and still per-example, board- and time-conditioned.

**Scope boundaries:**
- No training loop, no real data download, no evaluation (Plans 4-5).
- Modifies two Plan 2 files: `config.py` (adds `t_ctx`, `gab_rank`) and `encoder.py` (threads optional `film` through blocks). Both changes are backward-compatible — Plan 2's 36 tests MUST still pass (film defaults to `None`).

---

## New/changed dimensions

| Name | Value | Meaning |
|---|---|---|
| `t_ctx` | 128 | temporal context vector width |
| `gab_rank` | 4 | low-rank factorization rank for GAB bias |

**New file structure:**
```
sahformer/model/
  temporal.py       # TemporalEncoder
  modulation.py     # FiLMGenerator, GeometricAttentionBias
  clockaware.py     # ClockAwareChessformer
  config.py         # MODIFY: add t_ctx, gab_rank
  encoder.py        # MODIFY: thread optional per-layer film through blocks
tests/model/
  test_temporal.py
  test_modulation.py
  test_clockaware.py
```

---

## Task 1: Config additions

**Files:**
- Modify: `sahformer/model/config.py`
- Modify: `tests/model/test_config.py` (append)

- [ ] **Step 1: Append the failing test**

Append to `tests/model/test_config.py`:
```python
def test_modulation_dims():
    c = ModelConfig()
    assert c.t_ctx == 128
    assert c.gab_rank == 4
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_config.py::test_modulation_dims -v`
Expected: FAIL (`AttributeError: ... t_ctx`)

- [ ] **Step 3: Add the fields**

In `sahformer/model/config.py`, add these two fields to the `ModelConfig` dataclass (place them after `mdn_components`):
```python
    t_ctx: int = 128
    gab_rank: int = 4
```

- [ ] **Step 4: Run to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_config.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add sahformer/model/config.py tests/model/test_config.py
git commit -m "feat: add t_ctx and gab_rank config for time modulation"
```

---

## Task 2: TemporalEncoder

**Files:**
- Create: `sahformer/model/temporal.py`
- Create: `tests/model/test_temporal.py`

**Behavior:** `temporal (B,21) → Linear(21,t_ctx) → GELU → Linear(t_ctx,t_ctx) → LayerNorm → t (B,t_ctx)`.

- [ ] **Step 1: Write the failing test**

`tests/model/test_temporal.py`:
```python
import torch
from sahformer.model.config import ModelConfig
from sahformer.model.temporal import TemporalEncoder

def test_temporal_encoder_shape():
    c = ModelConfig()
    enc = TemporalEncoder(c)
    out = enc(torch.zeros(5, c.temporal_dim))
    assert out.shape == (5, c.t_ctx)

def test_temporal_encoder_varies_with_input():
    c = ModelConfig()
    enc = TemporalEncoder(c)
    a = enc(torch.zeros(1, c.temporal_dim))
    b = enc(torch.ones(1, c.temporal_dim))
    assert not torch.allclose(a, b)

def test_temporal_encoder_gradients():
    c = ModelConfig()
    enc = TemporalEncoder(c)
    x = torch.randn(3, c.temporal_dim, requires_grad=True)
    enc(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_temporal.py -v`
Expected: FAIL (import error)

- [ ] **Step 3: Implement**

`sahformer/model/temporal.py`:
```python
import torch
import torch.nn as nn
from sahformer.model.config import ModelConfig

class TemporalEncoder(nn.Module):
    """Map the 21-dim in-game temporal feature vector to a context vector t."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.temporal_dim, cfg.t_ctx),
            nn.GELU(),
            nn.Linear(cfg.t_ctx, cfg.t_ctx),
        )
        self.ln = nn.LayerNorm(cfg.t_ctx)

    def forward(self, temporal: torch.Tensor) -> torch.Tensor:
        return self.ln(self.net(temporal.float()))
```

- [ ] **Step 4: Run to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_temporal.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add sahformer/model/temporal.py tests/model/test_temporal.py
git commit -m "feat: temporal encoder mapping clock features to context vector"
```

---

## Task 3: FiLM generator + thread film through the encoder

**Files:**
- Create: `sahformer/model/modulation.py` (FiLMGenerator; GAB added in Task 4)
- Modify: `sahformer/model/encoder.py` (block + encoder accept optional `film`)
- Create: `tests/model/test_modulation.py`

**Behavior:**
- `FiLMGenerator`: `t (B,t_ctx) → Linear → (γ, β)` each `(B, n_layers, d_model)`. `γ = 1 + raw`, `β = raw2`, with small init so it starts near identity.
- `TransformerBlock.forward(x, attn_bias=None, film=None)`: if `film=(γ_l, β_l)` (each `(B, d_model)`) is given, apply `x = γ_l.unsqueeze(1) * x + β_l.unsqueeze(1)` after the two sublayers.
- `TransformerEncoder.forward(x, attn_bias=None, film=None)`: if `film=(γ_all, β_all)` (each `(B, n_layers, d_model)`), pass layer `i`'s slice to block `i`.

- [ ] **Step 1: Write the failing test**

`tests/model/test_modulation.py`:
```python
import torch
from sahformer.model.config import ModelConfig
from sahformer.model.modulation import FiLMGenerator
from sahformer.model.encoder import TransformerEncoder

def test_film_generator_shapes():
    c = ModelConfig()
    gen = FiLMGenerator(c)
    gamma, beta = gen(torch.randn(4, c.t_ctx))
    assert gamma.shape == (4, c.n_layers, c.d_model)
    assert beta.shape == (4, c.n_layers, c.d_model)

def test_film_starts_near_identity():
    c = ModelConfig()
    gen = FiLMGenerator(c)
    gamma, beta = gen(torch.zeros(2, c.t_ctx))
    # near identity: gamma ~ 1, beta ~ 0
    assert torch.allclose(gamma, torch.ones_like(gamma), atol=0.2)
    assert torch.allclose(beta, torch.zeros_like(beta), atol=0.2)

def test_encoder_film_changes_output():
    c = ModelConfig()
    enc = TransformerEncoder(c)
    x = torch.randn(2, c.seq_len, c.d_model)
    base = enc(x)
    gamma = torch.full((2, c.n_layers, c.d_model), 1.5)
    beta = torch.full((2, c.n_layers, c.d_model), 0.3)
    modulated = enc(x, film=(gamma, beta))
    assert not torch.allclose(base, modulated)

def test_encoder_no_film_matches_baseline():
    # film=None must behave exactly like the Plan 2 encoder
    c = ModelConfig()
    enc = TransformerEncoder(c)
    x = torch.randn(2, c.seq_len, c.d_model)
    assert torch.allclose(enc(x), enc(x, film=None))
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_modulation.py -v`
Expected: FAIL (import error for FiLMGenerator)

- [ ] **Step 3: Implement FiLMGenerator and modify the encoder**

`sahformer/model/modulation.py`:
```python
import math
import torch
import torch.nn as nn
from sahformer.model.config import ModelConfig

class FiLMGenerator(nn.Module):
    """Produce per-layer, per-channel (gamma, beta) from the temporal context t.
    Initialized so gamma~1, beta~0 (near identity at start)."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_layers = cfg.n_layers
        self.d_model = cfg.d_model
        self.gen = nn.Linear(cfg.t_ctx, 2 * cfg.n_layers * cfg.d_model)
        nn.init.normal_(self.gen.weight, std=0.02)
        nn.init.zeros_(self.gen.bias)

    def forward(self, t: torch.Tensor):
        b = t.shape[0]
        raw = self.gen(t).view(b, self.n_layers, 2, self.d_model)
        gamma = 1.0 + raw[:, :, 0, :]
        beta = raw[:, :, 1, :]
        return gamma, beta
```

Replace `sahformer/model/encoder.py` entirely with (adds the `film` plumbing; attention unchanged):
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sahformer.model.config import ModelConfig

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h = cfg.n_heads
        self.dh = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x, attn_bias=None):
        b, s, _ = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        if attn_bias is not None:
            logits = logits + attn_bias
        attn = F.softmax(logits, dim=-1)
        ctx = attn @ v
        ctx = ctx.transpose(1, 2).reshape(b, s, self.h * self.dh)
        return self.out(ctx)

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        hidden = cfg.d_model * cfg.mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, hidden), nn.GELU(), nn.Linear(hidden, cfg.d_model)
        )

    def forward(self, x, attn_bias=None, film=None):
        x = x + self.attn(self.ln1(x), attn_bias=attn_bias)
        x = x + self.mlp(self.ln2(x))
        if film is not None:
            gamma, beta = film                      # each (B, d_model)
            x = gamma.unsqueeze(1) * x + beta.unsqueeze(1)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

    def forward(self, x, attn_bias=None, film=None):
        for i, blk in enumerate(self.blocks):
            layer_film = None
            if film is not None:
                gamma_all, beta_all = film          # each (B, n_layers, d_model)
                layer_film = (gamma_all[:, i, :], beta_all[:, i, :])
            x = blk(x, attn_bias=attn_bias, film=layer_film)
        return self.ln_f(x)
```

- [ ] **Step 4: Run to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_modulation.py tests/model/test_encoder.py -v`
Expected: PASS (4 modulation + 4 encoder). The Plan 2 encoder tests MUST still pass (film defaults to None).

- [ ] **Step 5: Commit**

```bash
git add sahformer/model/modulation.py sahformer/model/encoder.py tests/model/test_modulation.py
git commit -m "feat: FiLM generator and per-layer film plumbing in encoder"
```

---

## Task 4: Time-conditioned Geometric Attention Bias (low-rank)

**Files:**
- Modify: `sahformer/model/modulation.py` (append `GeometricAttentionBias`)
- Modify: `tests/model/test_modulation.py` (append)

**Behavior:** `GeometricAttentionBias.forward(board_tokens, t, skill_self, skill_opp)` where `board_tokens (B,64,d_model)`, `t (B,t_ctx)`, `skill_self/opp (B,d_model)`. It:
1. compresses each board token to 32 dims, flattens → `(B, 64*32)`, projects to a 128-dim board context;
2. fuses `[board_ctx, t, skill_self, skill_opp]` → 128-dim `c`;
3. produces `u, v ∈ (B, n_heads, 64, gab_rank)` and forms `board_bias[b,h,i,j] = (u_i·v_j)/√rank`;
4. pads into a `(B, n_heads, 66, 66)` bias with zeros for the 2 skill-token rows/cols.

Small init so the bias starts near zero (baseline-preserving).

- [ ] **Step 1: Append the failing test**

Append to `tests/model/test_modulation.py`:
```python
from sahformer.model.modulation import GeometricAttentionBias

def test_gab_bias_shape():
    c = ModelConfig()
    gab = GeometricAttentionBias(c)
    board = torch.randn(3, 64, c.d_model)
    t = torch.randn(3, c.t_ctx)
    ss = torch.randn(3, c.d_model)
    so = torch.randn(3, c.d_model)
    bias = gab(board, t, ss, so)
    assert bias.shape == (3, c.n_heads, c.seq_len, c.seq_len)

def test_gab_skill_token_rows_are_zero():
    c = ModelConfig()
    gab = GeometricAttentionBias(c)
    bias = gab(torch.randn(2, 64, c.d_model), torch.randn(2, c.t_ctx),
               torch.randn(2, c.d_model), torch.randn(2, c.d_model))
    nsk = c.n_skill_tokens
    # skill-token rows and columns carry no GAB bias
    assert torch.count_nonzero(bias[:, :, :nsk, :]) == 0
    assert torch.count_nonzero(bias[:, :, :, :nsk]) == 0

def test_gab_bias_depends_on_time():
    c = ModelConfig()
    gab = GeometricAttentionBias(c)
    board = torch.randn(2, 64, c.d_model)
    ss = torch.randn(2, c.d_model)
    so = torch.randn(2, c.d_model)
    b_calm = gab(board, torch.zeros(2, c.t_ctx), ss, so)
    b_panic = gab(board, torch.ones(2, c.t_ctx) * 3.0, ss, so)
    # same board, different clock context -> different attention bias
    assert not torch.allclose(b_calm, b_panic)

def test_gab_gradients():
    c = ModelConfig()
    gab = GeometricAttentionBias(c)
    t = torch.randn(2, c.t_ctx, requires_grad=True)
    gab(torch.randn(2, 64, c.d_model), t, torch.randn(2, c.d_model),
        torch.randn(2, c.d_model)).sum().backward()
    assert t.grad is not None and torch.isfinite(t.grad).all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_modulation.py -k gab -v`
Expected: FAIL (import error for GeometricAttentionBias)

- [ ] **Step 3: Append the implementation**

Append to `sahformer/model/modulation.py`:
```python
class GeometricAttentionBias(nn.Module):
    """Low-rank, time-conditioned attention bias over the 64 board tokens.
    bias[b,h,i,j] = (u_i . v_j) / sqrt(rank), padded with zeros for skill tokens."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.h = cfg.n_heads
        self.r = cfg.gab_rank
        self.nsq = cfg.n_squares
        self.nsk = cfg.n_skill_tokens
        self.token_compress = nn.Linear(cfg.d_model, 32)
        self.board_ctx = nn.Sequential(
            nn.Linear(cfg.n_squares * 32, 128), nn.GELU(), nn.LayerNorm(128)
        )
        cin = 128 + cfg.t_ctx + 2 * cfg.d_model
        self.fuse = nn.Sequential(nn.Linear(cin, 128), nn.GELU())
        self.to_u = nn.Linear(128, self.h * self.nsq * self.r)
        self.to_v = nn.Linear(128, self.h * self.nsq * self.r)
        for lin in (self.to_u, self.to_v):
            nn.init.normal_(lin.weight, std=0.02)
            nn.init.zeros_(lin.bias)

    def forward(self, board_tokens, t, skill_self, skill_opp):
        b = board_tokens.shape[0]
        comp = self.token_compress(board_tokens).reshape(b, -1)     # (B, 64*32)
        bctx = self.board_ctx(comp)                                 # (B, 128)
        c = torch.cat([bctx, t, skill_self, skill_opp], dim=-1)
        c = self.fuse(c)                                            # (B, 128)
        u = self.to_u(c).view(b, self.h, self.nsq, self.r)
        v = self.to_v(c).view(b, self.h, self.nsq, self.r)
        board_bias = torch.einsum("bhir,bhjr->bhij", u, v) / math.sqrt(self.r)
        s = self.cfg.seq_len
        bias = torch.zeros(b, self.h, s, s, device=board_bias.device, dtype=board_bias.dtype)
        bias[:, :, self.nsk:, self.nsk:] = board_bias
        return bias
```

- [ ] **Step 4: Run to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_modulation.py -v`
Expected: PASS (4 film + 4 gab = 8 passed)

- [ ] **Step 5: Commit**

```bash
git add sahformer/model/modulation.py tests/model/test_modulation.py
git commit -m "feat: low-rank time-conditioned geometric attention bias"
```

---

## Task 5: Assemble ClockAwareChessformer

**Files:**
- Create: `sahformer/model/clockaware.py`
- Create: `tests/model/test_clockaware.py`

**Behavior:** `ClockAwareChessformer.forward(batch)` consumes the same batch as `Chessformer` PLUS `batch["temporal"] (B,21)`. It encodes `t`, builds FiLM `(γ,β)` and the GAB bias, runs the encoder with both, and fuses `t` into the pooled vector (`pooled = board_out.mean(1) + Linear(t_ctx,d_model)(t)`) before the value and MDN heads. Returns the same dict keys as `Chessformer`.

- [ ] **Step 1: Write the failing test**

`tests/model/test_clockaware.py`:
```python
import torch
from sahformer.model.config import ModelConfig
from sahformer.model.clockaware import ClockAwareChessformer

def _batch(b=3, temporal_fill=0.0):
    return {
        "board": torch.zeros(b, 8, 8, 12),
        "history": torch.zeros(b, 7, 8, 8, 12),
        "elo_self": torch.tensor([1500] * b),
        "elo_opp": torch.tensor([1600] * b),
        "temporal": torch.full((b, 21), temporal_fill),
    }

def test_forward_shapes():
    c = ModelConfig()
    model = ClockAwareChessformer(c)
    out = model(_batch(3))
    assert out["move_logits"].shape == (3, 64, 64)
    assert out["promo_logits"].shape == (3, 64, 4)
    assert out["value_logits"].shape == (3, 3)
    assert out["mdn"][0].shape == (3, c.mdn_components)

def test_param_budget():
    c = ModelConfig()
    n = sum(p.numel() for p in ClockAwareChessformer(c).parameters())
    # bigger than the 4.49M baseline, still well under 12M
    assert 4_500_000 < n < 12_000_000, n

def test_backward_runs():
    c = ModelConfig()
    model = ClockAwareChessformer(c)
    out = model(_batch(2))
    (out["move_logits"].sum() + out["value_logits"].sum()).backward()
    assert any(p.grad is not None for p in model.parameters())

def test_time_actually_changes_output():
    # THE key test: same board/elo, different clock -> different predictions.
    c = ModelConfig()
    model = ClockAwareChessformer(c)
    model.eval()
    with torch.no_grad():
        calm = model(_batch(2, temporal_fill=0.0))
        panic = model(_batch(2, temporal_fill=1.0))
    assert not torch.allclose(calm["move_logits"], panic["move_logits"], atol=1e-6)
    assert not torch.allclose(calm["mdn"][1], panic["mdn"][1], atol=1e-6)  # mu differs
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_clockaware.py -v`
Expected: FAIL (import error)

- [ ] **Step 3: Implement**

`sahformer/model/clockaware.py`:
```python
import torch
import torch.nn as nn
from sahformer.model.config import ModelConfig
from sahformer.model.embedding import InputEmbedding, SkillEmbedding
from sahformer.model.encoder import TransformerEncoder
from sahformer.model.heads import PolicyHead, ValueHead, ThinkTimeMDNHead
from sahformer.model.temporal import TemporalEncoder
from sahformer.model.modulation import FiLMGenerator, GeometricAttentionBias

class ClockAwareChessformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_emb = InputEmbedding(cfg)
        self.skill_emb = SkillEmbedding(cfg)
        self.temporal_enc = TemporalEncoder(cfg)
        self.film_gen = FiLMGenerator(cfg)
        self.gab = GeometricAttentionBias(cfg)
        self.encoder = TransformerEncoder(cfg)
        self.policy = PolicyHead(cfg)
        self.value = ValueHead(cfg)
        self.think = ThinkTimeMDNHead(cfg)
        self.t_to_d = nn.Linear(cfg.t_ctx, cfg.d_model)

    def forward(self, batch: dict) -> dict:
        board = batch["board"].float()
        history = batch["history"].float()
        board_tok = self.input_emb(board, history)              # (B,64,d)
        self_tok = self.skill_emb(batch["elo_self"])            # (B,d)
        opp_tok = self.skill_emb(batch["elo_opp"])              # (B,d)
        t = self.temporal_enc(batch["temporal"])               # (B,t_ctx)

        gamma, beta = self.film_gen(t)                         # each (B,L,d)
        bias = self.gab(board_tok, t, self_tok, opp_tok)       # (B,h,66,66)

        seq = torch.cat([self_tok.unsqueeze(1), opp_tok.unsqueeze(1), board_tok], dim=1)
        enc = self.encoder(seq, attn_bias=bias, film=(gamma, beta))
        board_out = enc[:, self.cfg.n_skill_tokens:, :]        # (B,64,d)
        pooled = board_out.mean(dim=1) + self.t_to_d(t)        # (B,d), clock-aware

        move_logits, promo_logits = self.policy(board_out)
        value_logits = self.value(pooled)
        mdn = self.think(pooled)
        return {
            "move_logits": move_logits,
            "promo_logits": promo_logits,
            "value_logits": value_logits,
            "mdn": mdn,
        }
```

- [ ] **Step 4: Run to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_clockaware.py -v`
Expected: PASS (4 passed). If `test_time_actually_changes_output` fails, the modulation isn't wired to the outputs — debug the wiring, do NOT weaken the test. Report the actual parameter count from `test_param_budget`.

- [ ] **Step 5: Commit**

```bash
git add sahformer/model/clockaware.py tests/model/test_clockaware.py
git commit -m "feat: assemble ClockAwareChessformer (FiLM + GAB + time-aware heads)"
```

---

## Task 6: End-to-end integration + behavioral sanity

**Files:**
- Create: `tests/model/test_clockaware_integration.py`

**Behavior:** Confirm the clock-aware model consumes a real `ShardDataset` batch (which includes `temporal`), all three losses backprop, AND a behavioral sanity check: two identical positions differing only in the temporal vector produce different think-time distributions.

- [ ] **Step 1: Write the test**

`tests/model/test_clockaware_integration.py`:
```python
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sahformer.records import PositionRecord
from sahformer.encoding import TEMPORAL_DIM, build_temporal
from sahformer.shards import records_to_arrays, save_shard
from sahformer.dataset import ShardDataset
from sahformer.model.config import ModelConfig
from sahformer.model.clockaware import ClockAwareChessformer
from sahformer.model.heads import mdn_nll

def _rec(temporal):
    return PositionRecord(
        board=np.zeros((8, 8, 12), np.int8),
        history=np.zeros((7, 8, 8, 12), np.int8),
        stm=0, elo_self=1500, elo_opp=1500,
        temporal=temporal.astype(np.float32),
        move_from=3, move_to=19, promo=0, result=2, think_time=1.5,
    )

def test_end_to_end_with_temporal(tmp_path):
    calm = build_temporal(my_clock=120.0, opp_clock=120.0, own_think_history=[3.0], ply=10)
    panic = build_temporal(my_clock=2.0, opp_clock=90.0, own_think_history=[0.05], ply=40)
    save_shard(str(tmp_path / "s.npz"), records_to_arrays([_rec(calm), _rec(panic)]))
    ds = ShardDataset([str(tmp_path / "s.npz")])
    batch = next(iter(DataLoader(ds, batch_size=2)))

    model = ClockAwareChessformer(ModelConfig())
    out = model(batch)

    b = batch["move_from"].shape[0]
    policy_loss = F.cross_entropy(out["move_logits"].reshape(b, -1),
                                  batch["move_from"] * 64 + batch["move_to"])
    value_loss = F.cross_entropy(out["value_logits"], batch["result"])
    pi, mu, sigma_p = out["mdn"]
    time_loss = mdn_nll(pi, mu, sigma_p, batch["think_time"])
    total = policy_loss + 0.1 * value_loss + 0.2 * time_loss
    assert torch.isfinite(total)
    total.backward()
    assert any(p.grad is not None for p in model.parameters())

def test_time_pressure_shifts_think_time_distribution(tmp_path):
    calm = build_temporal(my_clock=150.0, opp_clock=150.0, own_think_history=[4.0], ply=8)
    panic = build_temporal(my_clock=1.5, opp_clock=80.0, own_think_history=[0.05, 0.05], ply=50)
    save_shard(str(tmp_path / "s.npz"), records_to_arrays([_rec(calm), _rec(panic)]))
    ds = ShardDataset([str(tmp_path / "s.npz")])
    batch = next(iter(DataLoader(ds, batch_size=2)))
    model = ClockAwareChessformer(ModelConfig())
    model.eval()
    with torch.no_grad():
        out = model(batch)
    mu = out["mdn"][1]
    # the two rows have different clock context -> different predicted log-time means
    assert not torch.allclose(mu[0], mu[1], atol=1e-6)
```

- [ ] **Step 2: Run it**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_clockaware_integration.py -v`
Expected: PASS (2 passed). If a seam mismatch appears (dtype/shape from DataLoader collation of `temporal`), fix the real cause and explain.

- [ ] **Step 3: Run the FULL suite**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: all tests pass (Plan 1's 17 + Plan 2's 19 + Plan 3's new tests). Report the total.

- [ ] **Step 4: Commit**

```bash
git add tests/model/test_clockaware_integration.py
git commit -m "test: clock-aware end-to-end with temporal features and behavioral sanity"
```

---

## Self-review notes

- **Spec coverage (Section 2 time modulation):** temporal→context vector (Task 2), FiLM per-layer per-channel modulation applied to every block (Task 3), time-conditioned GAB fed via the encoder's `attn_bias` hook (Task 4), `t` routed into value + MDN heads via pooled fusion (Task 5), full assembly with the baseline preserved for ablation (Task 5), real-dataset seam + behavioral sanity that time shifts the think-time distribution (Task 6). The spec's low-rank concern for GAB is addressed by the rank-`r` factorization.
- **Backward compatibility:** `encoder.py` changes default `film=None`, so Plan 2's 36 tests remain valid; Task 3 Step 4 explicitly re-runs them. The Plan 2 `Chessformer` is untouched and remains the clock-blind ablation baseline.
- **Placeholder scan:** none — every step has runnable code/commands.
- **Type/name consistency:** `FiLMGenerator` returns `(gamma, beta)` each `(B, n_layers, d_model)`; encoder consumes exactly that; `GeometricAttentionBias.forward(board_tokens, t, skill_self, skill_opp)` signature matches its call in `clockaware.py`; config fields `t_ctx`/`gab_rank` used consistently; head-fusion uses `t_to_d: Linear(t_ctx, d_model)`.

## Open items for Plan 4 (training)

- Curriculum from the spec: (a) baseline `Chessformer`, (b) +think-time head [already in baseline], (c) FiLM-only, (d) full `ClockAwareChessformer`. Provide a flag to disable GAB and/or FiLM for the ablation runs (e.g. pass `attn_bias=None` / `film=None` selectively, or a config toggle).
- Loss weights: policy 1.0, value 0.1, think-time 0.2–0.3 (tune). Guard metric: move-accuracy vs the clock-blind baseline.
- AdamW + cyclic cosine, fp16, batch 128 (→64 if T4-tight), checkpoint to Google Drive across Colab disconnects.
- The clock-aware model must be compared against the Plan 2 baseline on time-pressure slices (Plan 5 eval).
