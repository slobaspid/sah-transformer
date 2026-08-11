# Faithful Maia-3 Backbone Implementation Plan (Plan 2 v2 of 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **SUPERSEDES** `2026-08-11-model-backbone.md`. After reading Maia-3's real source, we rebuild the backbone faithful to their exact 5M config. This plan REPLACES the code under `sahformer/model/` (config, embedding, encoder, heads, chessformer) with faithful versions. Old model tests are removed/replaced. The Plan 1 data pipeline is untouched.

**Goal:** A trainable model whose backbone matches Maia-3's real 5M architecture (skill-concat input, per-layer avg-pool GAB inside attention, RMSNorm post-norm trunk, 4352 from-to+promotion policy head, LN-before-pool value head), plus our MDN think-time head. Clock-blind here; the time layer is Plan 3 v2.

**Architecture (their exact numbers, from `maia3/model_registry.py`):** 64 square tokens, each `12×history(8) + 2×dim_emb(128) = 352` channels (board planes + both skill embeddings concatenated per square) → `Linear(352→256)`, no absolute PE. Trunk: 8 blocks, `dim_vit=256`, 8 heads, `mlp_ratio=2`, RMSNorm post-norm, GELU, no qkv bias. GAB per block: avg-pool tokens → `Linear(256→64)`+GELU+LN → `Linear(64→heads·64)`+GELU+LN → reshape `(B,8,64)` → shared `gab_weight(4096,64)` einsum → `(B,8,64,64)` added to attention logits. Heads: policy 4352, value 3, MDN think-time.

**Tech Stack:** Python 3.12 (`.venv`), PyTorch 2.4, pytest. Use `./.venv/Scripts/python.exe`.

**License note:** Maia-3 is AGPL-3.0. Reimplement from the spec below — do NOT copy their source verbatim.

---

## Config (locked; `sahformer/model/config.py`)

| field | value |
|---|---|
| `history` | 8 (current + 7 past) |
| `in_channels` | `12*8 = 96` board planes |
| `dim_emb` | 128 (skill embedding) |
| `token_in` | `96 + 2*128 = 352` |
| `dim_vit` (d_model) | 256 |
| `num_blocks` | 8 |
| `num_heads` | 8 |
| `head_dim` | 32 |
| `mlp_ratio` | 2 (FFN 512) |
| `gab_gen_size` (d3) | 64 |
| `gab_intermediate_dim` (d2) | 64 |
| `gab_per_square_dim` (d1) | 0 (avg-pool variant) |
| `head_hid_dim` | 256 |
| `mdn_components` | 3 |
| `temporal_dim` | 21 (used in Plan 3) |
| `t_ctx` | 128 (used in Plan 3) |
| `n_squares` | 64 |

**Move index scheme (locked; consumed by training):** a move maps to one of **4352** classes.
Non-promotion: `index = from*64 + to` (0..4095). Promotion: `index = 4096 + to*4 + (promo-1)`
where `promo ∈ {1:N,2:B,3:R,4:Q}` and `to` is the destination square 0..63. Helper
`move_to_index(from_sq, to_sq, promo)` lives in `heads.py`.

**File structure (replacing the old model files):**
```
sahformer/model/
  config.py       # ModelConfig (faithful fields)
  norm.py         # RMSNorm
  embedding.py    # SkillEmbedding, InputEmbedding (skill concat onto tokens)
  gab.py          # GeometricAttentionBias (per-layer, avg-pool, optional time ctx)
  encoder.py      # MHA (+GAB), EncoderBlock (post-norm RMS), Encoder (shared gab_weight)
  heads.py        # move_to_index, PolicyHead(4352), ValueHead, ThinkTimeMDNHead, mdn_nll
  chessformer.py  # FaithfulChessformer (clock-blind baseline)
```

---

## Task 0: Remove superseded model code

**Files:** delete old `sahformer/model/{embedding,encoder,heads,chessformer}.py` and old `tests/model/test_{embedding,encoder,heads,chessformer,integration}.py`. Keep `config.py` (will be overwritten) and `__init__.py`.

- [ ] **Step 1: Delete superseded files**

```bash
git rm sahformer/model/embedding.py sahformer/model/encoder.py sahformer/model/heads.py sahformer/model/chessformer.py
git rm tests/model/test_embedding.py tests/model/test_encoder.py tests/model/test_heads.py tests/model/test_chessformer.py tests/model/test_integration.py
```

- [ ] **Step 2: Verify remaining suite still imports**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_encoding.py tests/test_records.py tests/test_shards.py tests/test_dataset.py tests/test_download.py -q`
Expected: Plan 1's 17 tests pass (model tests are gone).

- [ ] **Step 3: Commit**

```bash
git commit -m "chore: remove superseded model code for faithful Maia-3 rebuild"
```

---

## Task 1: Config + RMSNorm

**Files:** overwrite `sahformer/model/config.py`; create `sahformer/model/norm.py`; overwrite `tests/model/test_config.py`; create `tests/model/test_norm.py`.

- [ ] **Step 1: Write tests**

`tests/model/test_config.py`:
```python
from sahformer.model.config import ModelConfig

def test_faithful_dims():
    c = ModelConfig()
    assert c.history == 8
    assert c.in_channels == 96
    assert c.dim_emb == 128
    assert c.token_in == 352
    assert c.dim_vit == 256
    assert c.num_blocks == 8
    assert c.num_heads == 8
    assert c.head_dim == 32
    assert c.gab_gen_size == 64
    assert c.gab_intermediate_dim == 64
    assert c.gab_per_square_dim == 0
    assert c.head_hid_dim == 256
    assert c.mdn_components == 3
    assert c.n_squares == 64
```

`tests/model/test_norm.py`:
```python
import torch
from sahformer.model.config import ModelConfig
from sahformer.model.norm import RMSNorm

def test_rmsnorm_shape_and_scale():
    n = RMSNorm(256)
    x = torch.randn(2, 64, 256)
    out = n(x)
    assert out.shape == x.shape
    # RMS of output along last dim ~ 1 at init (weight=1)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)
```

- [ ] **Step 2: Run, confirm fail.**

Run: `./.venv/Scripts/python.exe -m pytest tests/model/test_config.py tests/model/test_norm.py -v`

- [ ] **Step 3: Implement**

`sahformer/model/config.py`:
```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    history: int = 8              # current + 7 past
    dim_emb: int = 128            # skill embedding width
    dim_vit: int = 256            # d_model
    num_blocks: int = 8
    num_heads: int = 8
    mlp_ratio: int = 2
    gab_gen_size: int = 64        # d3
    gab_intermediate_dim: int = 64  # d2
    gab_per_square_dim: int = 0   # d1 (0 => avg-pool variant)
    head_hid_dim: int = 256
    mdn_components: int = 3
    n_squares: int = 64
    temporal_dim: int = 21        # Plan 3
    t_ctx: int = 128              # Plan 3

    @property
    def in_channels(self) -> int:
        return 12 * self.history

    @property
    def token_in(self) -> int:
        return self.in_channels + 2 * self.dim_emb

    @property
    def head_dim(self) -> int:
        return self.dim_vit // self.num_heads
```

`sahformer/model/norm.py`:
```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)
```

- [ ] **Step 4: Run, confirm pass. Step 5: Commit**

```bash
git add sahformer/model/config.py sahformer/model/norm.py tests/model/test_config.py tests/model/test_norm.py
git commit -m "feat: faithful Maia-3 config and RMSNorm"
```

---

## Task 2: Skill embedding + skill-concat input embedding

**Files:** create `sahformer/model/embedding.py`, `tests/model/test_embedding.py`.

**Behavior:** `SkillEmbedding` interpolates weak/strong endpoints (dim_emb) with `γ=(5000−elo)/5000`. `InputEmbedding` builds 64 tokens: per square, concat `[12*history board channels, skill_self(128), skill_opp(128)]` → `token_in=352` → `Linear(352→dim_vit)`. No positional embedding (GAB handles position).

- [ ] **Step 1: Write test**

`tests/model/test_embedding.py`:
```python
import torch
from sahformer.model.config import ModelConfig
from sahformer.model.embedding import SkillEmbedding, InputEmbedding

def test_skill_interp_shape():
    c = ModelConfig()
    s = SkillEmbedding(c)
    out = s(torch.tensor([0, 5000, 2500]))
    assert out.shape == (3, c.dim_emb)

def test_skill_clamps():
    c = ModelConfig()
    s = SkillEmbedding(c)
    assert torch.allclose(s(torch.tensor([9000])), s(torch.tensor([5000])), atol=1e-6)

def test_input_embedding_shape():
    c = ModelConfig()
    emb = InputEmbedding(c)
    board = torch.zeros(4, 8, 8, 12)
    history = torch.zeros(4, 7, 8, 8, 12)
    elo_self = torch.tensor([1500, 1500, 1500, 1500])
    elo_opp = torch.tensor([1400, 1400, 1400, 1400])
    out = emb(board, history, elo_self, elo_opp)
    assert out.shape == (4, 64, c.dim_vit)
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement**

`sahformer/model/embedding.py`:
```python
import torch
import torch.nn as nn
from sahformer.model.config import ModelConfig

class SkillEmbedding(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.weak = nn.Parameter(torch.randn(cfg.dim_emb) * 0.02)
        self.strong = nn.Parameter(torch.randn(cfg.dim_emb) * 0.02)

    def forward(self, elo: torch.Tensor) -> torch.Tensor:
        gamma = torch.clamp((5000.0 - elo.float()) / 5000.0, 0.0, 1.0).unsqueeze(-1)
        return gamma * self.weak + (1.0 - gamma) * self.strong

class InputEmbedding(nn.Module):
    """64 tokens: board planes (current+7 history) concatenated per square with the
    two skill embeddings, projected to dim_vit. No absolute positional embedding."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.skill = SkillEmbedding(cfg)
        self.proj = nn.Linear(cfg.token_in, cfg.dim_vit)

    def forward(self, board, history, elo_self, elo_opp):
        b = board.shape[0]
        cur = board.reshape(b, 64, 12)                                   # (B,64,12)
        hist = history.permute(0, 2, 3, 1, 4).reshape(b, 64, 7 * 12)     # (B,64,84)
        planes = torch.cat([cur, hist], dim=-1)                          # (B,64,96)
        ss = self.skill(elo_self).unsqueeze(1).expand(b, 64, -1)         # (B,64,128)
        so = self.skill(elo_opp).unsqueeze(1).expand(b, 64, -1)          # (B,64,128)
        tok = torch.cat([planes, ss, so], dim=-1)                        # (B,64,352)
        return self.proj(tok)                                            # (B,64,dim_vit)
```

- [ ] **Step 4: Run, confirm pass. Step 5: Commit**

```bash
git add sahformer/model/embedding.py tests/model/test_embedding.py
git commit -m "feat: skill embedding and skill-concat input embedding (faithful)"
```

---

## Task 3: Per-layer Geometric Attention Bias (avg-pool variant)

**Files:** create `sahformer/model/gab.py`, `tests/model/test_gab.py`.

**Behavior:** `GeometricAttentionBias.forward(tokens, gab_weight, t=None)` where `tokens (B,64,dim_vit)`, `gab_weight (64*64, gab_gen_size)` shared param, optional `t (B,t_ctx)` (Plan 3; None here). Avg-pool tokens → optional concat `t` → `Linear(→gab_intermediate)`+GELU+LN → `Linear(→heads*gab_gen_size)`+GELU+LN → reshape `(B,heads,gab_gen_size)` → einsum with `gab_weight` → `(B,heads,64,64)`.

The `Linear` input dim is `dim_vit + (t_ctx if time_conditioned else 0)`. Construct with a `time_conditioned: bool` flag so Plan 3 flips it on without changing this file.

- [ ] **Step 1: Write test**

`tests/model/test_gab.py`:
```python
import torch
from sahformer.model.config import ModelConfig
from sahformer.model.gab import GeometricAttentionBias

def _gw(c):
    return torch.randn(64 * 64, c.gab_gen_size)

def test_gab_shape():
    c = ModelConfig()
    gab = GeometricAttentionBias(c)
    bias = gab(torch.randn(3, 64, c.dim_vit), _gw(c))
    assert bias.shape == (3, c.num_heads, 64, 64)

def test_gab_depends_on_board():
    c = ModelConfig()
    gab = GeometricAttentionBias(c)
    gw = _gw(c)
    a = gab(torch.zeros(2, 64, c.dim_vit), gw)
    b = gab(torch.randn(2, 64, c.dim_vit), gw)
    assert not torch.allclose(a, b)

def test_gab_gradients():
    c = ModelConfig()
    gab = GeometricAttentionBias(c)
    x = torch.randn(2, 64, c.dim_vit, requires_grad=True)
    gab(x, _gw(c)).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

def test_gab_time_conditioned_shape():
    c = ModelConfig()
    gab = GeometricAttentionBias(c, time_conditioned=True)
    bias = gab(torch.randn(2, 64, c.dim_vit), _gw(c), t=torch.randn(2, c.t_ctx))
    assert bias.shape == (2, c.num_heads, 64, 64)
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement**

`sahformer/model/gab.py`:
```python
import torch
import torch.nn as nn
from sahformer.model.config import ModelConfig

class GeometricAttentionBias(nn.Module):
    """Per-layer GAB, 5M avg-pool variant. Optionally conditioned on temporal ctx t."""
    def __init__(self, cfg: ModelConfig, time_conditioned: bool = False):
        super().__init__()
        self.cfg = cfg
        self.h = cfg.num_heads
        self.time_conditioned = time_conditioned
        in_dim = cfg.dim_vit + (cfg.t_ctx if time_conditioned else 0)
        self.lin1 = nn.Linear(in_dim, cfg.gab_intermediate_dim)
        self.ln1 = nn.LayerNorm(cfg.gab_intermediate_dim)
        self.lin2 = nn.Linear(cfg.gab_intermediate_dim, self.h * cfg.gab_gen_size)
        self.ln2 = nn.LayerNorm(self.h * cfg.gab_gen_size)
        self.act = nn.GELU()

    def forward(self, tokens, gab_weight, t=None):
        b = tokens.shape[0]
        pooled = tokens.mean(dim=1)                      # (B, dim_vit) avg-pool variant
        if self.time_conditioned:
            assert t is not None, "time_conditioned GAB requires t"
            pooled = torch.cat([pooled, t], dim=-1)
        y = self.ln1(self.act(self.lin1(pooled)))
        y = self.ln2(self.act(self.lin2(y)))
        y = y.view(b, self.h, self.cfg.gab_gen_size)     # (B, h, d3)
        # shared final projection: templates -> 4096 -> 64x64
        flat = torch.einsum("bhi,oi->bho", y, gab_weight)  # (B, h, 4096)
        return flat.view(b, self.h, 64, 64)
```

- [ ] **Step 4: Run, confirm pass. Step 5: Commit**

```bash
git add sahformer/model/gab.py tests/model/test_gab.py
git commit -m "feat: per-layer avg-pool geometric attention bias (faithful)"
```

---

## Task 4: Attention + post-norm RMS block + encoder

**Files:** create `sahformer/model/encoder.py`, `tests/model/test_encoder.py`.

**Behavior:** `MHA` = qkv `Linear(dim_vit, 3*dim_vit, bias=False)`, adds the GAB bias `(B,h,64,64)` to attention logits. `EncoderBlock` = **post-norm** with RMSNorm: `x = norm1(x + attn(x, bias)); x = norm2(x + mlp(x))`. `Encoder` owns the **shared** `gab_weight` parameter and one GAB module per block, and threads optional per-block time (`t`) and FiLM (Plan 3; here `t=None`, `film=None`).

- [ ] **Step 1: Write test**

`tests/model/test_encoder.py`:
```python
import torch
from sahformer.model.config import ModelConfig
from sahformer.model.encoder import Encoder

def test_encoder_shape():
    c = ModelConfig()
    enc = Encoder(c)
    x = torch.randn(2, 64, c.dim_vit)
    assert enc(x).shape == (2, 64, c.dim_vit)

def test_encoder_gradients():
    c = ModelConfig()
    enc = Encoder(c)
    x = torch.randn(2, 64, c.dim_vit, requires_grad=True)
    enc(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

def test_shared_gab_weight_is_single_param():
    c = ModelConfig()
    enc = Encoder(c)
    # exactly one parameter named gab_weight, shape (4096, gab_gen_size)
    gw = [p for n, p in enc.named_parameters() if n.endswith("gab_weight")]
    assert len(gw) == 1 and tuple(gw[0].shape) == (4096, c.gab_gen_size)
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement**

`sahformer/model/encoder.py`:
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sahformer.model.config import ModelConfig
from sahformer.model.norm import RMSNorm
from sahformer.model.gab import GeometricAttentionBias

class MHA(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h = cfg.num_heads
        self.dh = cfg.head_dim
        self.qkv = nn.Linear(cfg.dim_vit, 3 * cfg.dim_vit, bias=False)
        self.out = nn.Linear(cfg.dim_vit, cfg.dim_vit, bias=False)

    def forward(self, x, gab_bias):
        b, s, _ = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh) + gab_bias
        attn = F.softmax(logits, dim=-1)
        ctx = (attn @ v).transpose(1, 2).reshape(b, s, self.h * self.dh)
        return self.out(ctx)

class EncoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, time_conditioned: bool = False):
        super().__init__()
        self.attn = MHA(cfg)
        self.gab = GeometricAttentionBias(cfg, time_conditioned=time_conditioned)
        self.norm1 = RMSNorm(cfg.dim_vit)
        self.norm2 = RMSNorm(cfg.dim_vit)
        hidden = cfg.dim_vit * cfg.mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(cfg.dim_vit, hidden), nn.GELU(), nn.Linear(hidden, cfg.dim_vit)
        )

    def forward(self, x, gab_weight, t=None, film=None):
        bias = self.gab(x, gab_weight, t=t)
        x = self.norm1(x + self.attn(x, bias))          # post-norm
        x = self.norm2(x + self.mlp(x))
        if film is not None:
            gamma, beta = film
            x = gamma.unsqueeze(1) * x + beta.unsqueeze(1)
        return x

class Encoder(nn.Module):
    def __init__(self, cfg: ModelConfig, time_conditioned: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList(
            [EncoderBlock(cfg, time_conditioned=time_conditioned) for _ in range(cfg.num_blocks)]
        )
        # shared-across-layers final GAB projection
        self.gab_weight = nn.Parameter(torch.randn(64 * 64, cfg.gab_gen_size) * 0.02)

    def forward(self, x, t=None, film=None):
        for i, blk in enumerate(self.blocks):
            layer_film = None
            if film is not None:
                g, be = film
                layer_film = (g[:, i, :], be[:, i, :])
            x = blk(x, self.gab_weight, t=t, film=layer_film)
        return x
```

- [ ] **Step 4: Run, confirm pass. Step 5: Commit**

```bash
git add sahformer/model/encoder.py tests/model/test_encoder.py
git commit -m "feat: post-norm RMS encoder with per-layer GAB and shared gab_weight"
```

---

## Task 5: Heads (policy 4352, value LN-pool, MDN) + move index

**Files:** create `sahformer/model/heads.py`, `tests/model/test_heads.py`.

- [ ] **Step 1: Write test**

`tests/model/test_heads.py`:
```python
import math
import torch
from sahformer.model.config import ModelConfig
from sahformer.model.heads import (move_to_index, PolicyHead, ValueHead,
                                   ThinkTimeMDNHead, mdn_nll)

def test_move_index_scheme():
    assert move_to_index(3, 19, 0) == 3 * 64 + 19          # non-promo
    assert move_to_index(0, 0, 4) == 4096 + 0 * 4 + 3      # queen promo to sq 0
    assert move_to_index(10, 63, 1) == 4096 + 63 * 4 + 0   # knight promo to sq 63

def test_policy_head_4352():
    c = ModelConfig()
    head = PolicyHead(c)
    logits = head(torch.randn(2, 64, c.dim_vit))
    assert logits.shape == (2, 4352)

def test_value_head():
    c = ModelConfig()
    head = ValueHead(c)
    assert head(torch.randn(2, 64, c.dim_vit)).shape == (2, 3)

def test_mdn_head_and_loss():
    c = ModelConfig()
    head = ThinkTimeMDNHead(c)
    pooled = torch.randn(4, c.dim_vit, requires_grad=True)
    pi, mu, sig = head(pooled)
    assert pi.shape == (4, c.mdn_components)
    loss = mdn_nll(pi, mu, sig, torch.tensor([0.05, 2.0, 12.0, 0.5]))
    assert torch.isfinite(loss)
    loss.backward()
    assert pooled.grad is not None

def test_mdn_single_component_matches_normal():
    pi = torch.zeros(1, 1); mu = torch.zeros(1, 1)
    sigp = torch.log(torch.expm1(torch.tensor([[1.0 - 1e-3]])))
    nll = mdn_nll(pi, mu, sigp, torch.tensor([1.0]))
    assert abs(nll.item() - 0.5 * math.log(2 * math.pi)) < 1e-3
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement**

`sahformer/model/heads.py`:
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sahformer.model.config import ModelConfig
from sahformer.model.norm import RMSNorm

def move_to_index(from_sq: int, to_sq: int, promo: int) -> int:
    """4352-class move index. Non-promo: from*64+to. Promo: 4096 + to*4 + (promo-1)."""
    if promo == 0:
        return from_sq * 64 + to_sq
    return 4096 + to_sq * 4 + (promo - 1)

class PolicyHead(nn.Module):
    """From-to bilinear (4096) + promotion logits (64 dest x 4 pieces = 256) = 4352."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.proj_from = nn.Linear(cfg.dim_vit, cfg.head_hid_dim, bias=False)
        self.proj_to = nn.Linear(cfg.dim_vit, cfg.head_hid_dim, bias=False)
        self.scale = 1.0 / math.sqrt(cfg.head_hid_dim)
        self.promo = nn.Linear(cfg.dim_vit, 4)

    def forward(self, tokens):
        b = tokens.shape[0]
        qf = self.proj_from(tokens)                       # (B,64,hid)
        kt = self.proj_to(tokens)                         # (B,64,hid)
        moves = torch.einsum("bid,bjd->bij", qf, kt) * self.scale  # (B,64,64)
        moves = moves.reshape(b, 4096)
        promo = self.promo(tokens).reshape(b, 64 * 4)     # (B,256): dest*4 + piece
        return torch.cat([moves, promo], dim=-1)          # (B,4352)

class ValueHead(nn.Module):
    """Norm -> mean-pool -> Linear+ReLU -> 3 (W/D/L)."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.dim_vit)
        self.hid = nn.Linear(cfg.dim_vit, cfg.head_hid_dim)
        self.out = nn.Linear(cfg.head_hid_dim, 3)

    def forward(self, tokens):
        pooled = self.norm(tokens).mean(dim=1)
        return self.out(F.relu(self.hid(pooled)))

class ThinkTimeMDNHead(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        m = cfg.mdn_components
        self.trunk = nn.Sequential(nn.Linear(cfg.dim_vit, cfg.head_hid_dim), nn.ReLU())
        self.pi = nn.Linear(cfg.head_hid_dim, m)
        self.mu = nn.Linear(cfg.head_hid_dim, m)
        self.sigma = nn.Linear(cfg.head_hid_dim, m)

    def forward(self, pooled):
        h = self.trunk(pooled)
        return self.pi(h), self.mu(h), self.sigma(h)

def mdn_nll(pi_logits, mu, sigma_param, target_time, eps: float = 1e-6):
    x = torch.log(target_time.clamp_min(eps)).unsqueeze(-1)
    log_pi = F.log_softmax(pi_logits, dim=-1)
    sigma = F.softplus(sigma_param) + 1e-3
    z = (x - mu) / sigma
    comp = -0.5 * z * z - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
    return -(torch.logsumexp(log_pi + comp, dim=-1)).mean()
```

- [ ] **Step 4: Run, confirm pass. Step 5: Commit**

```bash
git add sahformer/model/heads.py tests/model/test_heads.py
git commit -m "feat: 4352 policy head, LN-pool value head, MDN think-time head"
```

---

## Task 6: Assemble FaithfulChessformer (clock-blind baseline)

**Files:** create `sahformer/model/chessformer.py`, `tests/model/test_chessformer.py`.

- [ ] **Step 1: Write test**

`tests/model/test_chessformer.py`:
```python
import torch
from sahformer.model.config import ModelConfig
from sahformer.model.chessformer import FaithfulChessformer

def _batch(b=3):
    return {"board": torch.zeros(b, 8, 8, 12), "history": torch.zeros(b, 7, 8, 8, 12),
            "elo_self": torch.tensor([1500] * b), "elo_opp": torch.tensor([1600] * b)}

def test_forward_shapes():
    c = ModelConfig()
    out = FaithfulChessformer(c)(_batch(3))
    assert out["move_logits"].shape == (3, 4352)
    assert out["value_logits"].shape == (3, 3)
    assert out["mdn"][0].shape == (3, c.mdn_components)

def test_param_count_5m_class():
    c = ModelConfig()
    n = sum(p.numel() for p in FaithfulChessformer(c).parameters())
    assert 3_000_000 < n < 9_000_000, n

def test_backward():
    c = ModelConfig()
    out = FaithfulChessformer(c)(_batch(2))
    out["move_logits"].sum().backward()
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement**

`sahformer/model/chessformer.py`:
```python
import torch
import torch.nn as nn
from sahformer.model.config import ModelConfig
from sahformer.model.embedding import InputEmbedding
from sahformer.model.encoder import Encoder
from sahformer.model.heads import PolicyHead, ValueHead, ThinkTimeMDNHead

class FaithfulChessformer(nn.Module):
    """Clock-blind Maia-3-faithful backbone + our MDN think-time head."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_emb = InputEmbedding(cfg)
        self.encoder = Encoder(cfg)
        self.policy = PolicyHead(cfg)
        self.value = ValueHead(cfg)
        self.think = ThinkTimeMDNHead(cfg)

    def forward(self, batch: dict) -> dict:
        tok = self.input_emb(batch["board"].float(), batch["history"].float(),
                             batch["elo_self"], batch["elo_opp"])
        enc = self.encoder(tok)                       # (B,64,dim_vit)
        pooled = enc.mean(dim=1)
        return {
            "move_logits": self.policy(enc),
            "value_logits": self.value(enc),
            "mdn": self.think(pooled),
        }
```

- [ ] **Step 4: Run, confirm pass (report actual param count). Step 5: Commit**

```bash
git add sahformer/model/chessformer.py tests/model/test_chessformer.py
git commit -m "feat: assemble faithful clock-blind Chessformer baseline"
```

---

## Task 7: Integration with the Plan 1 dataset

**Files:** create `tests/model/test_integration.py`.

- [ ] **Step 1: Write test**

`tests/model/test_integration.py`:
```python
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sahformer.records import PositionRecord
from sahformer.encoding import TEMPORAL_DIM
from sahformer.shards import records_to_arrays, save_shard
from sahformer.dataset import ShardDataset
from sahformer.model.config import ModelConfig
from sahformer.model.chessformer import FaithfulChessformer
from sahformer.model.heads import move_to_index, mdn_nll

def _rec(mf, mt, promo, result, tt):
    return PositionRecord(
        board=np.zeros((8, 8, 12), np.int8), history=np.zeros((7, 8, 8, 12), np.int8),
        stm=0, elo_self=1500, elo_opp=1500, temporal=np.zeros(TEMPORAL_DIM, np.float32),
        move_from=mf, move_to=mt, promo=promo, result=result, think_time=tt)

def test_end_to_end(tmp_path):
    recs = [_rec(3, 19, 0, 2, 1.5), _rec(8, 16, 0, 1, 0.05)]
    save_shard(str(tmp_path / "s.npz"), records_to_arrays(recs))
    batch = next(iter(DataLoader(ShardDataset([str(tmp_path / "s.npz")]), batch_size=2)))
    model = FaithfulChessformer(ModelConfig())
    out = model(batch)
    b = batch["move_from"].shape[0]
    target = torch.tensor([move_to_index(int(batch["move_from"][i]), int(batch["move_to"][i]),
                                         int(batch["promo"][i])) for i in range(b)])
    policy_loss = F.cross_entropy(out["move_logits"], target)
    value_loss = F.cross_entropy(out["value_logits"], batch["result"])
    pi, mu, sig = out["mdn"]
    time_loss = mdn_nll(pi, mu, sig, batch["think_time"])
    total = policy_loss + 0.1 * value_loss + 0.2 * time_loss
    assert torch.isfinite(total)
    total.backward()
    assert any(p.grad is not None for p in model.parameters())
```

- [ ] **Step 2: Run it.** Fix any real seam mismatch (do not weaken).

- [ ] **Step 3: Full suite** `./.venv/Scripts/python.exe -m pytest -q` — report total.

- [ ] **Step 4: Commit**

```bash
git add tests/model/test_integration.py
git commit -m "test: faithful backbone end-to-end with Plan 1 dataset"
```

---

## Self-review notes

- **Faithful to Maia-3 5M:** skill-concat input (352 depth), no absolute PE, per-layer avg-pool GAB with shared gab_weight, RMSNorm post-norm trunk, 8×256×8, from-to bilinear policy + 256 promotion logits (4352), LN-before-pool value head, dim_emb 128, γ interpolation. Numbers taken from `model_registry.py`.
- **Deviations (documented):** promotion index layout (`4096 + to*4 + promo-1`) is ours to match the Plan 1 dataset targets; their internal 8×8×4 layout is equivalent in count (256) and mechanism (separate promotion logits). MDN think-time head replaces their scalar ponder head (intentional, per our design).
- **Deferred to Plan 3 v2 (correctly out of scope):** temporal encoder, FiLM, time-conditioned GAB (the `time_conditioned` flag and `t`/`film` hooks are already wired through GAB/encoder), MDN routing of `t`, and the `ClockAwareChessformer` assembly.
- **Placeholder scan:** none. **Type consistency:** `ModelConfig` fields, `move_to_index` scheme, and head signatures consistent across tasks.

## Open items for Plan 3 v2

- `TemporalEncoder` (21 → t_ctx). Build `ClockAwareChessformer` = `FaithfulChessformer` with `Encoder(cfg, time_conditioned=True)`, a `FiLMGenerator`, `t` passed to encoder, and `t` fused into the MDN/value pooled vector.
- Time-sensitivity test: same board+elo, different `temporal` ⇒ different move logits AND different MDN μ.
