# Time-Adaptive Search ("ALLIE-lite") Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a search step that reuses the trained model's policy + value heads and spends a search budget derived from the predicted think-time — so the bot *calculates more on hard positions* (sacrifices, sharp tactics) and plays a better-checked, still-human move there.

**Architecture:** A single `sahformer/search.py` module implements a budgeted AlphaZero-style PUCT Monte-Carlo Tree Search: the **policy head** is the move prior, the **value head** (W/D/L → `P(win)−P(loss)`) is the leaf evaluator, and the number of simulations comes from the predicted think-time (`time_to_sims`). It's an on/off option in `self_play` (viewer) and the UCI engine; pure-policy play stays the default baseline. Search budgets are capped small to keep play **human, not superhuman** (the project's north star: human realism over strength).

**Tech Stack:** Python 3.12 (`.venv`), PyTorch 2.4, `python-chess`, pytest. Use `./.venv/Scripts/python.exe`.

**Design:** `docs/superpowers/specs/2026-08-12-time-adaptive-search-design.md`.

**Key facts the code relies on (already in the repo):**
- Value head output order is **`[P(loss), P(draw), P(win)]`** (result is stm-relative 0/1/2). So a position's scalar value (mover POV) is `wdl[2] - wdl[0] ∈ [−1, 1]`.
- `move_to_index(*encode_move(board, m))` gives the 4352-way policy index of a legal move `m` (same trick `self_play` uses to gather per-move priors).
- `_stack_history(plane_hist, cur)` builds the 7-ply history tensor from a list of prior encoded boards + the current one.

---

## File structure

```
sahformer/
  search.py                 # SearchConfig, time_to_sims, mcts_move (+ private MCTS helpers)
  play.py                   # (+) self_play/selfplay_frames gain a `search` option
scripts/
  uci_engine.py             # (+) a "Search" UCI option routes moves through mcts_move
tests/
  test_search.py            # time_to_sims + MCTS behavior (real untrained model + a mock)
  test_search_integration.py  # self_play(search=True) plays a legal game
```

---

## Task 1: SearchConfig + time_to_sims

**Files:** create `sahformer/search.py`, `tests/test_search.py`.

- [ ] **Step 1: Write the failing test**

`tests/test_search.py`:
```python
from sahformer.search import SearchConfig, time_to_sims

def test_time_to_sims_scales_and_clamps():
    c = SearchConfig(sims_per_second=16.0, min_sims=1, max_sims=256)
    assert time_to_sims(0.0, c) == 1                 # clamped up to the floor
    assert time_to_sims(0.2, c) in (3, 4)            # ~0.2s obvious move -> a few sims
    assert time_to_sims(8.0, c) == 128               # hard position -> real calculation
    assert time_to_sims(10_000.0, c) == 256          # clamped to the human-realism cap
    # monotonic non-decreasing
    xs = [time_to_sims(s, c) for s in (0.1, 1, 3, 8, 20)]
    assert xs == sorted(xs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_search.py -q`
Expected: FAIL — `No module named 'sahformer.search'`.

- [ ] **Step 3: Implement the config + mapping**

`sahformer/search.py`:
```python
from dataclasses import dataclass

@dataclass
class SearchConfig:
    c_puct: float = 1.5           # exploration constant in PUCT
    sims_per_second: float = 16.0  # predicted seconds -> simulation budget
    min_sims: int = 1             # instant moves fall back to (near) raw policy
    max_sims: int = 256           # human-realism cap: never calculate like an engine
    temperature: float = 0.0      # 0 = most-visited move; >0 = sample by visit counts
    elo: int = 1500               # skill to imitate during search

def time_to_sims(seconds: float, cfg: SearchConfig) -> int:
    """Turn a predicted human think-time (seconds) into a simulation budget."""
    n = round(float(seconds) * cfg.sims_per_second)
    return int(min(max(n, cfg.min_sims), cfg.max_sims))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_search.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sahformer/search.py tests/test_search.py
git commit -m "feat: search config + think-time -> simulation-budget mapping"
```

---

## Task 2: Budgeted PUCT MCTS (`mcts_move`)

**Files:** modify `sahformer/search.py`; modify `tests/test_search.py`.

**Behavior:** `mcts_move(model, board, plane_hist, sims, cfg, ...)` runs `sims` PUCT simulations from `board` and returns `(chosen_move, root_node)`. Each simulation: descend by PUCT to a leaf, evaluate the leaf with the model (policy prior on its legal moves + scalar value), back the value up flipping sign each ply. Move is chosen by visit counts (argmax, or visit-weighted sample when `temperature > 0`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_search.py`:
```python
import numpy as np
import torch
import chess
from sahformer.model.config import ModelConfig
from sahformer.training.build import build_model
from sahformer.search import mcts_move
from sahformer.model.heads import move_to_index
from sahformer.encoding import encode_move

def test_mcts_returns_legal_move_and_visits():
    model = build_model("full", ModelConfig()); model.eval()
    board = chess.Board()
    cfg = SearchConfig(elo=1500, temperature=0.0)
    move, root = mcts_move(model, board, [], sims=16, cfg=cfg, seed=0)
    assert move in board.legal_moves
    total_visits = sum(ch.N for ch in root.children.values())
    assert total_visits >= 16                         # every sim visited some child

def test_mcts_deterministic_with_temp0():
    model = build_model("full", ModelConfig()); model.eval()
    board = chess.Board()
    cfg = SearchConfig(temperature=0.0)
    m1, _ = mcts_move(model, board, [], sims=24, cfg=cfg, seed=0)
    m2, _ = mcts_move(model, board, [], sims=24, cfg=cfg, seed=0)
    assert m1 == m2

def test_mcts_prior_drives_visits():
    # a mock model that strongly favors one specific legal move (neutral value everywhere)
    board = chess.Board()
    favored = chess.Move.from_uci("e2e4")
    favored_idx = move_to_index(*encode_move(board, favored))

    class _MockModel:
        def eval(self): return self
        def __call__(self, batch):
            logits = torch.full((1, 4352), -10.0)
            logits[0, favored_idx] = 10.0             # huge prior on e2e4 at the root
            return {"move_logits": logits,
                    "value_logits": torch.zeros(1, 3),  # neutral value
                    "mdn": None}

    cfg = SearchConfig(temperature=0.0)
    move, root = mcts_move(_MockModel(), board, [], sims=40, cfg=cfg, seed=0)
    assert move == favored                            # search concentrates on the high-prior move
    assert root.children[favored].N == max(ch.N for ch in root.children.values())

def test_mcts_works_midgame():
    model = build_model("full", ModelConfig()); model.eval()
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    move, _ = mcts_move(model, board, [], sims=8, cfg=SearchConfig(), seed=1)
    assert move in board.legal_moves
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_search.py -q`
Expected: FAIL — `cannot import name 'mcts_move'`.

- [ ] **Step 3: Implement the MCTS**

Append to `sahformer/search.py`:
```python
import math
import numpy as np
import chess
import torch
import torch.nn.functional as F
from sahformer.encoding import encode_board, encode_move, build_temporal
from sahformer.records import _stack_history
from sahformer.model.heads import move_to_index

class _Node:
    __slots__ = ("board", "plane_hist", "prior", "children", "N", "W", "expanded")
    def __init__(self, board, plane_hist, prior=0.0):
        self.board = board
        self.plane_hist = plane_hist   # list of encoded boards before this position
        self.prior = prior
        self.children = {}             # chess.Move -> _Node
        self.N = 0                     # visit count
        self.W = 0.0                   # summed value, this node's mover POV
        self.expanded = False

    @property
    def Q(self):
        return self.W / self.N if self.N > 0 else 0.0

def _terminal_value(board):
    # from the side-to-move's POV: checkmated = loss (-1); any other game-over = draw (0)
    return -1.0 if board.is_checkmate() else 0.0

def _evaluate(model, node, cfg, my_clock, opp_clock, ply, device):
    """Run the model on node.board: set priors on its children, return value (mover POV)."""
    b = node.board
    cur = encode_board(b)
    hist = _stack_history(node.plane_hist, cur)
    temporal = build_temporal(my_clock, opp_clock, [], ply)
    batch = {
        "board": torch.from_numpy(cur).float().unsqueeze(0).to(device),
        "history": torch.from_numpy(hist).float().unsqueeze(0).to(device),
        "elo_self": torch.tensor([cfg.elo]).to(device),
        "elo_opp": torch.tensor([cfg.elo]).to(device),
        "temporal": torch.from_numpy(temporal).float().unsqueeze(0).to(device),
    }
    with torch.no_grad():
        out = model(batch)
    logits = out["move_logits"][0]
    legal = list(b.legal_moves)
    idxs = [move_to_index(*encode_move(b, m)) for m in legal]
    priors = F.softmax(logits[idxs], dim=-1).cpu().numpy()
    wdl = F.softmax(out["value_logits"][0], dim=-1).cpu().numpy()
    value = float(wdl[2] - wdl[0])                 # P(win) - P(loss), mover POV
    next_hist = (node.plane_hist + [cur])[-7:]
    for m, p in zip(legal, priors):
        child_board = b.copy(stack=False)
        child_board.push(m)
        node.children[m] = _Node(child_board, next_hist, prior=float(p))
    node.expanded = True
    return value

def _select_child(node, cfg):
    total_N = sum(c.N for c in node.children.values())
    sqrt_total = math.sqrt(total_N) if total_N > 0 else 1.0
    best_move, best_child, best_score = None, None, -1e18
    for m, c in node.children.items():
        u = cfg.c_puct * c.prior * sqrt_total / (1 + c.N)
        score = -c.Q + u                            # child Q is opponent POV -> negate for us
        if score > best_score:
            best_score, best_move, best_child = score, m, c
    return best_move, best_child

def _simulate(model, root, cfg, my_clock, opp_clock, ply, device):
    path = [root]
    node = root
    while node.expanded and node.children and not node.board.is_game_over():
        _, node = _select_child(node, cfg)
        path.append(node)
    if node.board.is_game_over():
        value = _terminal_value(node.board)
    else:
        value = _evaluate(model, node, cfg, my_clock, opp_clock, ply, device)
    for n in reversed(path):                        # back up, flipping perspective each ply
        n.N += 1
        n.W += value
        value = -value

def mcts_move(model, board, plane_hist, sims, cfg, my_clock=180.0, opp_clock=180.0,
              ply=0, device="cpu", seed=None):
    """Run `sims` PUCT simulations from `board`; return (chosen_move, root_node)."""
    rng = np.random.default_rng(seed)
    root = _Node(board.copy(stack=False), list(plane_hist))
    if board.is_game_over():
        return None, root
    _evaluate(model, root, cfg, my_clock, opp_clock, ply, device)   # expand root
    for _ in range(sims):
        _simulate(model, root, cfg, my_clock, opp_clock, ply, device)
    moves = list(root.children.keys())
    visits = np.array([root.children[m].N for m in moves], dtype=np.float64)
    if visits.sum() == 0:
        priors = np.array([root.children[m].prior for m in moves])
        return moves[int(priors.argmax())], root
    if cfg.temperature <= 1e-6:
        return moves[int(visits.argmax())], root
    p = visits ** (1.0 / cfg.temperature)
    p = p / p.sum()
    return moves[int(rng.choice(len(moves), p=p))], root
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_search.py -q`
Expected: PASS (all five: time_to_sims + four MCTS tests).

- [ ] **Step 5: Commit**

```bash
git add sahformer/search.py tests/test_search.py
git commit -m "feat: budgeted PUCT MCTS using policy prior + value-head leaf eval"
```

---

## Task 3: Route self-play through search (viewer)

**Files:** modify `sahformer/play.py`; create `tests/test_search_integration.py`.

**Behavior:** `self_play` (and therefore `selfplay_frames` and the viewer) gains a `search` flag and a `search_cfg`. When `search=True`, the move is chosen by `mcts_move` with a budget from `time_to_sims(predicted_think, search_cfg)` — so the *same* predicted think-time now sets both the on-screen pause and the search depth. When `search=False` (default), behavior is unchanged.

- [ ] **Step 1: Write the failing test**

`tests/test_search_integration.py`:
```python
import chess
from sahformer.model.config import ModelConfig
from sahformer.training.build import build_model
from sahformer.play import self_play
from sahformer.search import SearchConfig

def test_self_play_with_search_is_legal():
    model = build_model("full", ModelConfig())
    board = chess.Board()
    scfg = SearchConfig(max_sims=8, elo=1500, temperature=0.0)
    plies = list(self_play(model, max_plies=8, start_clock=100000.0, seed=0,
                           search=True, search_cfg=scfg))
    assert len(plies) >= 1
    for rec in plies:
        assert rec["move"] in board.legal_moves
        board.push(rec["move"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_search_integration.py -q`
Expected: FAIL — `self_play() got an unexpected keyword argument 'search'`.

- [ ] **Step 3: Add the search path to `self_play`**

In `sahformer/play.py`, add imports near the top (after the existing imports):
```python
from sahformer.search import SearchConfig, time_to_sims, mcts_move
```

Change the `self_play` signature to add the two args:
```python
@torch.no_grad()
def self_play(model, max_plies=200, elo=1500, temperature=1.0,
              start_clock=180.0, device="cpu", seed=0, think_temp=1.0,
              search=False, search_cfg=None):
```

Inside the loop, the current move-selection block is:
```python
        out = model(batch)
        logits = out["move_logits"][0]
        legal = list(board.legal_moves)
        idxs = [move_to_index(*encode_move(board, m)) for m in legal]
        scores = logits[idxs]
        if temperature <= 1e-6:
            move = legal[int(scores.argmax().item())]
        else:
            probs = F.softmax(scores / temperature, dim=-1).detach().cpu().numpy()
            probs = probs / probs.sum()
            move = legal[int(rng.choice(len(legal), p=probs))]
        think = _sample_think_time(out["mdn"], rng, think_temp=think_temp)
```
Replace it with (compute think first, then either search or sample the policy):
```python
        out = model(batch)
        think = _sample_think_time(out["mdn"], rng, think_temp=think_temp)
        if search:
            scfg = search_cfg or SearchConfig(elo=elo, temperature=temperature)
            sims = time_to_sims(think, scfg)
            move, _ = mcts_move(model, board, plane_hist, sims, scfg,
                                my_clock=clock[mover], opp_clock=clock[not mover],
                                ply=ply, device=device, seed=int(rng.integers(1 << 30)))
        else:
            logits = out["move_logits"][0]
            legal = list(board.legal_moves)
            idxs = [move_to_index(*encode_move(board, m)) for m in legal]
            scores = logits[idxs]
            if temperature <= 1e-6:
                move = legal[int(scores.argmax().item())]
            else:
                probs = F.softmax(scores / temperature, dim=-1).detach().cpu().numpy()
                probs = probs / probs.sum()
                move = legal[int(rng.choice(len(legal), p=probs))]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_search_integration.py tests/test_play.py -q`
Expected: PASS (new search game + the existing play tests unchanged).

- [ ] **Step 5: Add a `--search` flag to the viewer**

In `scripts/watch_selfplay.py`, add the arg and thread it through. After the existing `--think-temp` arg line, add:
```python
    ap.add_argument("--search", action="store_true", help="use time-adaptive MCTS for moves")
    ap.add_argument("--max-sims", type=int, default=256, help="cap on search simulations")
```
Change the `selfplay_frames(...)` call to pass a search config through (see Task 3 note): update `selfplay_frames` in `sahformer/play.py` to accept and forward `search`/`search_cfg`:
```python
def selfplay_frames(model, max_plies=120, elo=1500, temperature=1.0,
                    start_clock=180.0, seed=0, size=400, think_temp=1.0,
                    search=False, search_cfg=None):
```
and forward them in its internal `self_play(...)` call:
```python
    for rec in self_play(model, max_plies=max_plies, elo=elo, temperature=temperature,
                         start_clock=start_clock, device="cpu", seed=seed,
                         think_temp=think_temp, search=search, search_cfg=search_cfg):
```
Then in `scripts/watch_selfplay.py`'s `main`, build the config and pass it:
```python
    from sahformer.search import SearchConfig
    scfg = SearchConfig(elo=args.elo, temperature=args.temperature, max_sims=args.max_sims)
    frames, caps = selfplay_frames(model, max_plies=args.plies, elo=args.elo,
                                   temperature=args.temperature, seed=args.seed,
                                   think_temp=args.think_temp,
                                   search=args.search, search_cfg=scfg)
```

- [ ] **Step 6: Verify the viewer still builds a game (no search + with search)**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_selfplay_frames.py -q`
Expected: PASS (default path unchanged).

- [ ] **Step 7: Commit**

```bash
git add sahformer/play.py scripts/watch_selfplay.py tests/test_search_integration.py
git commit -m "feat: time-adaptive search option in self-play + viewer (--search)"
```

---

## Task 4: Route the UCI engine through search

**Files:** modify `scripts/uci_engine.py`; modify `tests/test_uci_engine.py`.

**Behavior:** a `Search` UCI option (0 = off/pure policy, >0 = on with that `max_sims` cap). When on, the engine picks its move with `mcts_move`, budgeting sims from the model's predicted think-time for the position; `Pace` still governs the visible/clock time.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_uci_engine.py`:
```python
def test_uci_search_option_plays_legal(monkeypatch, capsys):
    mod = _load("uci_engine", "scripts/uci_engine.py")
    cmds = ("uci\nisready\nsetoption name Search value 8\n"
            "position startpos moves e2e4 e7e5\ngo wtime 60000 btime 60000\nquit\n")
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(cmds))
    mod.main()
    out = capsys.readouterr().out
    assert "option name Search" in out
    bm = [l for l in out.splitlines() if l.startswith("bestmove ")]
    assert bm and len(bm[0].split()[1]) >= 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_uci_engine.py -q`
Expected: FAIL — no `option name Search` emitted / option not handled.

- [ ] **Step 3: Wire search into the engine**

In `scripts/uci_engine.py`, add the import near the others:
```python
from sahformer.search import SearchConfig, time_to_sims, mcts_move
from sahformer.play import _sample_think_time
```
Add a `search` entry to `opts` (initialised from env/arg like the others):
```python
    opts = {"elo": 1500, "temperature": 0.3,
            "pace": _initial("--pace", "SAHFORMER_PACE", 0.0),
            "think_temp": _initial("--think-temp", "SAHFORMER_THINK_TEMP", 0.7),
            "search": _initial("--search", "SAHFORMER_SEARCH", 0.0)}
```
Emit the option in the `uci` handler (after the `ThinkTemp` line):
```python
            emit("option name Search type string default 0")
```
Handle it in `setoption` (after the `thinktemp` branch):
```python
                    elif name == "search":
                        opts["search"] = float(val)
```
In `choose(...)`, after `out = model(batch)` and computing `logits`/`legal`, replace the move-selection block with a search branch. The current block is:
```python
        scores = logits[idxs]
        t = opts["temperature"]
        if t <= 1e-6:
            move = legal[int(scores.argmax().item())]
        else:
            probs = F.softmax(scores / t, dim=-1).cpu().numpy()
            probs = probs / probs.sum()
            move = legal[int(rng.choice(len(legal), p=probs))]
```
Replace with:
```python
        if opts["search"] > 0:
            scfg = SearchConfig(elo=opts["elo"], temperature=opts["temperature"],
                                max_sims=int(opts["search"]))
            think = _sample_think_time(out["mdn"], rng, think_temp=0.5)
            sims = time_to_sims(think, scfg)
            move, _ = mcts_move(model, b, plane_hist, sims, scfg,
                                my_clock=my, opp_clock=opp, ply=len(moves), device="cpu",
                                seed=int(rng.integers(1 << 30)))
        else:
            scores = logits[idxs]
            t = opts["temperature"]
            if t <= 1e-6:
                move = legal[int(scores.argmax().item())]
            else:
                probs = F.softmax(scores / t, dim=-1).cpu().numpy()
                probs = probs / probs.sum()
                move = legal[int(rng.choice(len(legal), p=probs))]
```
(`b`, `plane_hist`, `my`, `opp`, `moves` are all already in scope inside `choose`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_uci_engine.py -q`
Expected: PASS (protocol test + the new search-option test).

- [ ] **Step 5: Full suite + commit**

Run: `./.venv/Scripts/python.exe -m pytest -q` — report total.
```bash
git add scripts/uci_engine.py tests/test_uci_engine.py
git commit -m "feat: UCI Search option routes moves through time-adaptive MCTS"
```

---

## Task 5: Manual sanity — watch it search on a real model

**Files:** none (uses the trained `best.pt`).

- [ ] **Step 1: Self-play with search on the trained model**

```bash
.\.venv\Scripts\python.exe scripts\watch_selfplay.py --ckpt "C:\Users\sloba\Downloads\model_full\best.pt" --search --elo 2000 --temperature 0.2 --plies 100
```
Expected: a legal, complete game; it should be noticeably slower to generate (search runs many forward passes on the harder positions). Compare against the same command **without** `--search`.

- [ ] **Step 2: Note the qualitative difference**

Look at whether the sharp/critical positions (captures, sacrifices) get better-checked replies with search on, while quiet positions look the same. This is a qualitative human-realism check, **not** a strength benchmark — if it starts playing like a cold engine, lower `--max-sims`.

---

## Self-review notes

- **Spec coverage:** budgeted PUCT MCTS with value-head leaf eval + policy prior (Task 2); `time_to_sims` budget mapping (Task 1); human-realism cap via `max_sims` + human priors/value + visit-temperature (Tasks 1–2, `SearchConfig`); on/off switch with pure-policy default in both viewer and engine (Tasks 3–4); reuse of `move_to_index`/`_stack_history`/value convention (Task 2). All design sections map to a task.
- **Value convention:** `value = wdl[2] - wdl[0]` matches the repo's `[P(loss),P(draw),P(win)]` order (verified against `records.py` result encoding and `ValueHead`).
- **Backup sign:** value is leaf-mover POV; flipped each ply on the way up; `_select_child` negates child `Q` for the parent's POV. Consistent.
- **Type/name consistency:** `SearchConfig`, `time_to_sims(seconds, cfg)`, `mcts_move(model, board, plane_hist, sims, cfg, ...)` used identically across tasks and both call sites.
- **Placeholder scan:** none.
- **Guardrail honored:** search is opt-in, budget-capped, and driven by the human policy/value — pure-policy play stays the default and the honest baseline.

## Open items for the eval plan

- Compare pure-policy vs search on **move-match** and on the **time-pressure / sacrifice subset** on unseen games — does search actually improve human-realism (not just strength)?
- Tune `c_puct`, `sims_per_second`, `max_sims`, and the visit-temperature against human-alignment metrics.
- Speed: batch simulations / add a transposition table if live-blitz search is too slow.
