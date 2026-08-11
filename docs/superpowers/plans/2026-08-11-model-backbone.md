> ⚠️ **SUPERSEDED by `2026-08-11-model-backbone-v2-faithful.md`** (2026-08-11). After reading Maia-3's actual source we rebuilt the backbone faithful to their real 5M config. This plan (and the code it produced) is historical.

# Model Backbone + Heads Implementation Plan (Plan 2 of 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a trainable Chessformer-style baseline model that consumes the Plan 1 dataset and outputs a move-policy, a Win/Draw/Loss value, and a Mixture-Density think-time distribution — the clock-blind foundation the time-modulation plan (Plan 3) extends.

**Architecture:** 64 board-square tokens (current + 7 history planes → 96 channels each) embedded to `d_model`, plus 2 prepended skill tokens (interpolated Elo embeddings for self & opponent). A standard pre-LN transformer encoder with learned absolute positional embeddings (GAB replaces these in Plan 3). Three heads read the encoder output: a source→destination attention policy head (+ promotion logits), a mean-pooled WDL value head, and an MDN think-time head predicting a mixture of Gaussians over log-time.

**Tech Stack:** Python 3.12 (in `.venv`), PyTorch 2.4, pytest. All commands use the venv interpreter: `./.venv/Scripts/python.exe`.

**Scope boundaries:**
- This plan does NOT consume the `temporal` feature vector — the baseline is clock-blind. FiLM + time-conditioned GAB (which inject `temporal`) are Plan 3.
- This plan does NOT include a training loop, data loading beyond a smoke forward pass, or evaluation — those are Plans 4 and 5.
- Time-conditioned GAB is Plan 3. This plan uses learned absolute positional embeddings as the baseline positional signal.

---

## Model dimensions (locked here, referenced by Plan 3)

| Name | Value | Meaning |
|---|---|---|
| `N_SQUARES` | 64 | board tokens |
| `IN_CHANNELS` | 96 | 12 piece planes × (1 current + 7 history) |
| `D_MODEL` | 256 | token embedding width (5M-class) |
| `N_LAYERS` | 8 | transformer blocks |
| `N_HEADS` | 8 | attention heads (head_dim 32) |
| `MLP_RATIO` | 2 | FFN expansion |
| `SKILL_EMB` | 128 | skill embedding width before projection to `D_MODEL` |
| `TEMPORAL_DIM` | 21 | from `sahformer.encoding` (unused in Plan 2; wired in Plan 3) |
| `MDN_COMPONENTS` | 3 | Gaussian components in the think-time mixture |
| `SEQ_LEN` | 66 | 2 skill tokens + 64 board tokens |

**Token order in the sequence:** index 0 = self-skill token, index 1 = opponent-skill token, indices 2..65 = the 64 board squares (row-major, stm-oriented, matching `encode_board`). Heads that operate on board squares slice `[:, 2:, :]`.

**File structure:**
```
sahformer/model/
  __init__.py
  config.py        # ModelConfig dataclass
  embedding.py     # InputEmbedding, SkillEmbedding
  encoder.py       # TransformerBlock, TransformerEncoder (standard pre-LN)
  heads.py         # PolicyHead, ValueHead, ThinkTimeMDNHead, mdn_nll
  chessformer.py   # Chessformer: assembles everything, forward() -> dict
tests/model/
  __init__.py
  test_config.py
  test_embedding.py
  test_encoder.py
  test_heads.py
  test_chessformer.py
```

---

## Task 1: Model config

**Files:**
- Create: `sahformer/model/__init__.py`, `sahformer/model/config.py`
- Create: `tests/model/__init__.py`, `tests/model/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/model/__init__.py`: empty file.

`tests/model/test_config.py`:
```python
from sahformer.model.config import ModelConfig

def test_defaults_match_5m_class():
    c = ModelConfig()
    assert c.n_squares == 64
    assert c.in_channels == 96
    assert c.d_model == 256
    assert c.n_layers == 8
    assert c.n_heads == 8
    assert c.d_model % c.n_heads == 0
    assert c.skill_emb == 128
    assert c.mdn_components == 3
    assert c.seq_len == 66  # 2 skill + 64 board

def test_head_dim_derived():
    c = ModelConfig()
    assert c.head_dim == 32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_config.py -v`
Expected: FAIL (import error)

- [ ] **Step 3: Write minimal implementation**

`sahformer/model/__init__.py`:
```python
```

`sahformer/model/config.py`:
```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    n_squares: int = 64
    in_channels: int = 96          # 12 planes x (1 current + 7 history)
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    mlp_ratio: int = 2
    skill_emb: int = 128
    temporal_dim: int = 21
    mdn_components: int = 3
    n_skill_tokens: int = 2        # self + opponent

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def seq_len(self) -> int:
        return self.n_skill_tokens + self.n_squares
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add sahformer/model/__init__.py sahformer/model/config.py tests/model/__init__.py tests/model/test_config.py
git commit -m "feat: model config with 5M-class dimensions"
```

---

## Task 2: Input & skill embeddings

**Files:**
- Create: `sahformer/model/embedding.py`
- Create: `tests/model/test_embedding.py`

**Behavior:**
- `InputEmbedding` takes `board (B,8,8,12)` and `history (B,7,8,8,12)` floats, concatenates per-square channels into `(B,64,96)`, projects to `(B,64,d_model)`, and adds a learned `(64,d_model)` positional embedding.
- `SkillEmbedding` holds two learnable endpoint vectors `weak` (Elo 0) and `strong` (Elo 5000) of width `skill_emb`. For an Elo tensor it computes `gamma = clip((5000 - elo)/5000, 0, 1)` and returns `gamma*weak + (1-gamma)*strong`, then projects to `d_model`. Used for both self and opponent tokens.

- [ ] **Step 1: Write the failing test**

`tests/model/test_embedding.py`:
```python
import torch
from sahformer.model.config import ModelConfig
from sahformer.model.embedding import InputEmbedding, SkillEmbedding

def test_input_embedding_shape():
    c = ModelConfig()
    emb = InputEmbedding(c)
    board = torch.zeros(4, 8, 8, 12)
    history = torch.zeros(4, 7, 8, 8, 12)
    out = emb(board, history)
    assert out.shape == (4, 64, c.d_model)

def test_skill_embedding_shape_and_interpolation():
    c = ModelConfig()
    skill = SkillEmbedding(c)
    elo = torch.tensor([0, 5000, 2500])
    out = skill(elo)
    assert out.shape == (3, c.d_model)

def test_skill_endpoints_differ():
    c = ModelConfig()
    skill = SkillEmbedding(c)
    weak = skill(torch.tensor([0]))
    strong = skill(torch.tensor([5000]))
    # endpoints should not be identical after random init
    assert not torch.allclose(weak, strong)

def test_skill_clamps_out_of_range():
    c = ModelConfig()
    skill = SkillEmbedding(c)
    # elo above 5000 clamps to the strong endpoint projection
    hi = skill(torch.tensor([9000]))
    strong = skill(torch.tensor([5000]))
    assert torch.allclose(hi, strong, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_embedding.py -v`
Expected: FAIL (import error)

- [ ] **Step 3: Write minimal implementation**

`sahformer/model/embedding.py`:
```python
import torch
import torch.nn as nn
from sahformer.model.config import ModelConfig

class InputEmbedding(nn.Module):
    """Board + 7 history planes -> 64 tokens of width d_model, with learned
    absolute positional embeddings."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.in_channels, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(cfg.n_squares, cfg.d_model))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, board: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        b = board.shape[0]
        cur = board.reshape(b, 64, 12)                       # (B,64,12)
        # history (B,7,8,8,12) -> per square 7*12 channels
        hist = history.permute(0, 2, 3, 1, 4).reshape(b, 64, 7 * 12)  # (B,64,84)
        x = torch.cat([cur, hist], dim=-1)                   # (B,64,96)
        x = self.proj(x)                                     # (B,64,d_model)
        return x + self.pos.unsqueeze(0)

class SkillEmbedding(nn.Module):
    """Interpolated Elo embedding between a weak (0) and strong (5000) endpoint,
    projected to d_model."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.weak = nn.Parameter(torch.randn(cfg.skill_emb) * 0.02)
        self.strong = nn.Parameter(torch.randn(cfg.skill_emb) * 0.02)
        self.proj = nn.Linear(cfg.skill_emb, cfg.d_model)

    def forward(self, elo: torch.Tensor) -> torch.Tensor:
        elo = elo.float()
        gamma = torch.clamp((5000.0 - elo) / 5000.0, 0.0, 1.0).unsqueeze(-1)  # (B,1)
        emb = gamma * self.weak + (1.0 - gamma) * self.strong                 # (B,skill_emb)
        return self.proj(emb)                                                 # (B,d_model)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_embedding.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add sahformer/model/embedding.py tests/model/test_embedding.py
git commit -m "feat: input and interpolated skill embeddings"
```

---

## Task 3: Transformer encoder

**Files:**
- Create: `sahformer/model/encoder.py`
- Create: `tests/model/test_encoder.py`

**Behavior:** Standard pre-LN transformer. `TransformerBlock` = LN → multi-head self-attention → residual, then LN → MLP (GELU, expansion `mlp_ratio`) → residual. `TransformerEncoder` stacks `n_layers` of them. Attention accepts an optional additive bias of shape `(B, n_heads, S, S)` added to the attention logits before softmax — Plan 3's GAB will use it; in Plan 2 it is always `None`.

- [ ] **Step 1: Write the failing test**

`tests/model/test_encoder.py`:
```python
import torch
from sahformer.model.config import ModelConfig
from sahformer.model.encoder import TransformerBlock, TransformerEncoder

def test_block_preserves_shape():
    c = ModelConfig()
    blk = TransformerBlock(c)
    x = torch.randn(2, c.seq_len, c.d_model)
    out = blk(x)
    assert out.shape == x.shape

def test_encoder_preserves_shape():
    c = ModelConfig()
    enc = TransformerEncoder(c)
    x = torch.randn(2, c.seq_len, c.d_model)
    out = enc(x)
    assert out.shape == x.shape

def test_attention_bias_accepted():
    c = ModelConfig()
    blk = TransformerBlock(c)
    x = torch.randn(2, c.seq_len, c.d_model)
    bias = torch.zeros(2, c.n_heads, c.seq_len, c.seq_len)
    out = blk(x, attn_bias=bias)
    assert out.shape == x.shape

def test_gradients_flow():
    c = ModelConfig()
    enc = TransformerEncoder(c)
    x = torch.randn(2, c.seq_len, c.d_model, requires_grad=True)
    enc(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_encoder.py -v`
Expected: FAIL (import error)

- [ ] **Step 3: Write minimal implementation**

`sahformer/model/encoder.py`:
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
        q, k, v = qkv[0], qkv[1], qkv[2]              # (B,h,S,dh)
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)  # (B,h,S,S)
        if attn_bias is not None:
            logits = logits + attn_bias
        attn = F.softmax(logits, dim=-1)
        ctx = attn @ v                                # (B,h,S,dh)
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

    def forward(self, x, attn_bias=None):
        x = x + self.attn(self.ln1(x), attn_bias=attn_bias)
        x = x + self.mlp(self.ln2(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

    def forward(self, x, attn_bias=None):
        for blk in self.blocks:
            x = blk(x, attn_bias=attn_bias)
        return self.ln_f(x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_encoder.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add sahformer/model/encoder.py tests/model/test_encoder.py
git commit -m "feat: pre-LN transformer encoder with optional attention bias hook"
```

---

## Task 4: Heads (policy, value, MDN think-time) + MDN loss

**Files:**
- Create: `sahformer/model/heads.py`
- Create: `tests/model/test_heads.py`

**Behavior:**
- `PolicyHead`: from board tokens `(B,64,d_model)` produce move logits `(B,64,64)` via source-query · destination-key dot product (scaled), plus promotion logits `(B,64,4)` from a linear on destination keys.
- `ValueHead`: from a pooled vector `(B,d_model)` → `Linear(d_model,128)→ReLU→Linear(128,3)` → `(B,3)`.
- `ThinkTimeMDNHead`: from a pooled vector `(B,d_model)` → mixture params `pi_logits (B,M)`, `mu (B,M)`, `sigma_param (B,M)`. `sigma = softplus(sigma_param) + 1e-3`.
- `mdn_nll(pi_logits, mu, sigma_param, target_time)`: NLL of `x = log(target_time)` under the Gaussian mixture in log-time space, via log-sum-exp. Returns a scalar (mean over batch).

- [ ] **Step 1: Write the failing test**

`tests/model/test_heads.py`:
```python
import math
import torch
from sahformer.model.config import ModelConfig
from sahformer.model.heads import PolicyHead, ValueHead, ThinkTimeMDNHead, mdn_nll

def test_policy_head_shapes():
    c = ModelConfig()
    head = PolicyHead(c)
    board = torch.randn(2, 64, c.d_model)
    move_logits, promo_logits = head(board)
    assert move_logits.shape == (2, 64, 64)
    assert promo_logits.shape == (2, 64, 4)

def test_value_head_shape():
    c = ModelConfig()
    head = ValueHead(c)
    pooled = torch.randn(2, c.d_model)
    assert head(pooled).shape == (2, 3)

def test_mdn_head_shapes():
    c = ModelConfig()
    head = ThinkTimeMDNHead(c)
    pooled = torch.randn(2, c.d_model)
    pi, mu, sigma_p = head(pooled)
    assert pi.shape == (2, c.mdn_components)
    assert mu.shape == (2, c.mdn_components)
    assert sigma_p.shape == (2, c.mdn_components)

def test_mdn_nll_single_component_matches_normal():
    # One component, mu=0, sigma=1: NLL of x=log(1)=0 is 0.5*log(2*pi)
    pi_logits = torch.zeros(1, 1)
    mu = torch.zeros(1, 1)
    # choose sigma_param so softplus(sigma_param)+1e-3 == 1.0
    target_sigma = 1.0 - 1e-3
    sigma_param = torch.log(torch.expm1(torch.tensor([[target_sigma]])))  # inverse softplus
    target_time = torch.tensor([1.0])
    nll = mdn_nll(pi_logits, mu, sigma_param, target_time)
    assert abs(nll.item() - 0.5 * math.log(2 * math.pi)) < 1e-3

def test_mdn_nll_finite_and_differentiable():
    c = ModelConfig()
    head = ThinkTimeMDNHead(c)
    pooled = torch.randn(4, c.d_model, requires_grad=True)
    pi, mu, sigma_p = head(pooled)
    target = torch.tensor([0.05, 2.0, 12.0, 0.5])
    loss = mdn_nll(pi, mu, sigma_p, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert pooled.grad is not None and torch.isfinite(pooled.grad).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_heads.py -v`
Expected: FAIL (import error)

- [ ] **Step 3: Write minimal implementation**

`sahformer/model/heads.py`:
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sahformer.model.config import ModelConfig

class PolicyHead(nn.Module):
    """Source-destination attention policy. move_logits[b, from, to]."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.q = nn.Linear(cfg.d_model, cfg.d_model)   # source (from) queries
        self.k = nn.Linear(cfg.d_model, cfg.d_model)   # destination (to) keys
        self.scale = 1.0 / math.sqrt(cfg.d_model)
        self.promo = nn.Linear(cfg.d_model, 4)         # from destination keys

    def forward(self, board_tokens: torch.Tensor):
        q = self.q(board_tokens)                       # (B,64,d)
        k = self.k(board_tokens)                       # (B,64,d)
        move_logits = (q @ k.transpose(-2, -1)) * self.scale  # (B,64,64)
        promo_logits = self.promo(k)                   # (B,64,4)
        return move_logits, promo_logits

class ValueHead(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 128), nn.ReLU(), nn.Linear(128, 3)
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)

class ThinkTimeMDNHead(nn.Module):
    """Predict a mixture of Gaussians over log think-time."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        m = cfg.mdn_components
        self.trunk = nn.Sequential(nn.Linear(cfg.d_model, 128), nn.ReLU())
        self.pi = nn.Linear(128, m)
        self.mu = nn.Linear(128, m)
        self.sigma = nn.Linear(128, m)

    def forward(self, pooled: torch.Tensor):
        h = self.trunk(pooled)
        return self.pi(h), self.mu(h), self.sigma(h)

def mdn_nll(pi_logits, mu, sigma_param, target_time, eps: float = 1e-6):
    """Negative log-likelihood of log(target_time) under the Gaussian mixture."""
    x = torch.log(target_time.clamp_min(eps)).unsqueeze(-1)   # (B,1)
    log_pi = F.log_softmax(pi_logits, dim=-1)                 # (B,M)
    sigma = F.softplus(sigma_param) + 1e-3                    # (B,M)
    z = (x - mu) / sigma
    comp_logpdf = -0.5 * z * z - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
    log_prob = torch.logsumexp(log_pi + comp_logpdf, dim=-1)  # (B,)
    return -log_prob.mean()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_heads.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add sahformer/model/heads.py tests/model/test_heads.py
git commit -m "feat: policy, value, and MDN think-time heads with MDN NLL loss"
```

---

## Task 5: Assemble the full Chessformer model

**Files:**
- Create: `sahformer/model/chessformer.py`
- Create: `tests/model/test_chessformer.py`

**Behavior:** `Chessformer.forward(batch)` takes a dict of tensors matching the Plan 1 `ShardDataset` sample (batched): `board (B,8,8,12)`, `history (B,7,8,8,12)`, `elo_self (B,)`, `elo_opp (B,)` (and ignores `temporal`, `stm`, targets). It builds the 66-token sequence (2 skill + 64 board), runs the encoder, and returns a dict:
- `move_logits (B,64,64)`, `promo_logits (B,64,4)` from the board tokens
- `value_logits (B,3)` from mean-pooled board tokens
- `mdn (pi_logits, mu, sigma_param)` each `(B,M)` from mean-pooled board tokens

- [ ] **Step 1: Write the failing test**

`tests/model/test_chessformer.py`:
```python
import torch
from sahformer.model.config import ModelConfig
from sahformer.model.chessformer import Chessformer

def _batch(b=3):
    return {
        "board": torch.zeros(b, 8, 8, 12),
        "history": torch.zeros(b, 7, 8, 8, 12),
        "elo_self": torch.tensor([1500] * b),
        "elo_opp": torch.tensor([1600] * b),
    }

def test_forward_output_shapes():
    c = ModelConfig()
    model = Chessformer(c)
    out = model(_batch(3))
    assert out["move_logits"].shape == (3, 64, 64)
    assert out["promo_logits"].shape == (3, 64, 4)
    assert out["value_logits"].shape == (3, 3)
    pi, mu, sigma_p = out["mdn"]
    assert pi.shape == (3, c.mdn_components)

def test_param_count_is_5m_class():
    c = ModelConfig()
    model = Chessformer(c)
    n = sum(p.numel() for p in model.parameters())
    # 5M-class: comfortably between 2M and 12M
    assert 2_000_000 < n < 12_000_000, n

def test_backward_runs():
    c = ModelConfig()
    model = Chessformer(c)
    out = model(_batch(2))
    loss = out["move_logits"].sum() + out["value_logits"].sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_chessformer.py -v`
Expected: FAIL (import error)

- [ ] **Step 3: Write minimal implementation**

`sahformer/model/chessformer.py`:
```python
import torch
import torch.nn as nn
from sahformer.model.config import ModelConfig
from sahformer.model.embedding import InputEmbedding, SkillEmbedding
from sahformer.model.encoder import TransformerEncoder
from sahformer.model.heads import PolicyHead, ValueHead, ThinkTimeMDNHead

class Chessformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_emb = InputEmbedding(cfg)
        self.skill_emb = SkillEmbedding(cfg)
        self.encoder = TransformerEncoder(cfg)
        self.policy = PolicyHead(cfg)
        self.value = ValueHead(cfg)
        self.think = ThinkTimeMDNHead(cfg)

    def forward(self, batch: dict) -> dict:
        board = batch["board"].float()
        history = batch["history"].float()
        board_tok = self.input_emb(board, history)          # (B,64,d)
        self_tok = self.skill_emb(batch["elo_self"]).unsqueeze(1)   # (B,1,d)
        opp_tok = self.skill_emb(batch["elo_opp"]).unsqueeze(1)     # (B,1,d)
        seq = torch.cat([self_tok, opp_tok, board_tok], dim=1)      # (B,66,d)
        enc = self.encoder(seq)                             # (B,66,d)
        board_out = enc[:, self.cfg.n_skill_tokens:, :]     # (B,64,d)
        pooled = board_out.mean(dim=1)                      # (B,d)
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

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_chessformer.py -v`
Expected: PASS (3 passed). If `test_param_count_is_5m_class` fails, report the actual count — do NOT weaken the test without confirming the count is deliberate; adjust `d_model`/`n_layers` in a follow-up only if the coordinator approves.

- [ ] **Step 5: Commit**

```bash
git add sahformer/model/chessformer.py tests/model/test_chessformer.py
git commit -m "feat: assemble full Chessformer baseline (policy/value/MDN heads)"
```

---

## Task 6: End-to-end integration with the Plan 1 dataset

**Files:**
- Create: `tests/model/test_integration.py`

**Behavior:** Confirm the model consumes a real `ShardDataset` batch (via `torch.utils.data.DataLoader` default collation) and that all three losses compute and backprop together — proving Plan 1 and Plan 2 fit at the seams.

- [ ] **Step 1: Write the failing test**

`tests/model/test_integration.py`:
```python
import numpy as np
import torch
from torch.utils.data import DataLoader
from sahformer.records import PositionRecord
from sahformer.encoding import TEMPORAL_DIM
from sahformer.shards import records_to_arrays, save_shard
from sahformer.dataset import ShardDataset
from sahformer.model.config import ModelConfig
from sahformer.model.chessformer import Chessformer
from sahformer.model.heads import mdn_nll
import torch.nn.functional as F

def _rec(elo, mf, mt, promo, result, tt):
    return PositionRecord(
        board=np.zeros((8, 8, 12), np.int8),
        history=np.zeros((7, 8, 8, 12), np.int8),
        stm=0, elo_self=elo, elo_opp=elo,
        temporal=np.zeros(TEMPORAL_DIM, np.float32),
        move_from=mf, move_to=mt, promo=promo, result=result, think_time=tt,
    )

def test_end_to_end_batch_and_losses(tmp_path):
    recs = [_rec(1500, 3, 19, 0, 2, 1.5), _rec(1200, 8, 16, 0, 1, 0.05)]
    save_shard(str(tmp_path / "s.npz"), records_to_arrays(recs))
    ds = ShardDataset([str(tmp_path / "s.npz")])
    loader = DataLoader(ds, batch_size=2)
    batch = next(iter(loader))

    model = Chessformer(ModelConfig())
    out = model(batch)

    # policy loss: flatten 64x64 to 4096 classes, target = from*64+to
    b = batch["move_from"].shape[0]
    move_target = batch["move_from"] * 64 + batch["move_to"]
    policy_loss = F.cross_entropy(out["move_logits"].reshape(b, -1), move_target)
    value_loss = F.cross_entropy(out["value_logits"], batch["result"])
    pi, mu, sigma_p = out["mdn"]
    time_loss = mdn_nll(pi, mu, sigma_p, batch["think_time"])

    total = policy_loss + 0.1 * value_loss + 0.2 * time_loss
    assert torch.isfinite(total)
    total.backward()
    assert any(p.grad is not None for p in model.parameters())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_integration.py -v`
Expected: FAIL initially only if code has issues; if all prior tasks are correct it should pass once written. (Write the test, run it, and fix any real integration mismatch it surfaces — e.g. dtype of `elo_self` from the DataLoader.)

- [ ] **Step 3: Make it pass**

No new implementation expected. If the test fails due to a dtype/shape mismatch at the Plan1↔Plan2 seam, fix the real cause in the model (e.g. ensure `batch["elo_self"]` is handled as a long/int tensor in `SkillEmbedding.forward`, which already calls `.float()`). Report any change made.

- [ ] **Step 4: Run the FULL suite**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: all tests pass (Plan 1's 17 + Plan 2's new tests).

- [ ] **Step 5: Commit**

```bash
git add tests/model/test_integration.py
git commit -m "test: end-to-end Plan1 dataset -> model -> combined losses"
```

---

## Self-review notes

- **Spec coverage (Section 2 heads + backbone):** board-as-tokens + 7 history (Task 2), interpolated skill embeddings prepended as tokens (Task 2, Task 5), transformer encoder (Task 3), source→destination policy head + promotion (Task 4), mean-pooled WDL value head (Task 4), MDN think-time head over log-time with NLL loss (Task 4), full assembly + param-budget guard (Task 5), and a real dataset seam test with all three losses (Task 6).
- **Deferred to Plan 3 (correctly out of scope):** temporal feature consumption, FiLM modulation, time-conditioned GAB. The encoder already exposes an `attn_bias` hook (Task 3) so GAB slots in without refactoring; the model pools/`pooled` vector is where FiLM and the temporal context will attach.
- **Placeholder scan:** none — every step has runnable code/commands.
- **Type/name consistency:** `ModelConfig` field names identical across all tasks; `mdn_nll` signature `(pi_logits, mu, sigma_param, target_time)` identical in Task 4 definition and Task 6 use; head output tuple order `(pi_logits, mu, sigma_param)` consistent; `n_skill_tokens=2` slice `[:, 2:, :]` matches token order defined at top.

## Open items for Plan 3 (time modulation)

- Feed `batch["temporal"]` through a temporal-context MLP → `t` (dim ~128).
- FiLM: `t` → per-layer `(gamma, beta)`; apply inside each `TransformerBlock` (extend `forward` to accept/consume FiLM params).
- Time-conditioned GAB: build `(B, n_heads, 66, 66)` bias from board summary + `t` + skill, feed via the existing `attn_bias` hook; watch the parameter budget (a naive `d→h*4096` projection is millions of params — use the paper's pooling/low-rank reduction).
- Route `t` into the MDN and value heads so think-time prediction becomes clock-aware.
