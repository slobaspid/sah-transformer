# Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn raw Lichess 3+0 blitz PGNs (with clock annotations) into a rating-balanced tensor dataset of position records — the input contract every later plan (model, training, eval) consumes.

**Architecture:** Pure-Python extraction pipeline built on `python-chess`. A single game → a list of `PositionRecord`s. Records are encoded into compact NumPy arrays and written to sharded `.npz` files, then rebalanced across 22 Elo bins. A thin PyTorch `Dataset` reads shards. This plan builds NO model — it is fully testable on its own with tiny inline PGNs.

**Tech Stack:** Python 3.12, `python-chess`, `numpy`, `zstandard` (Lichess `.pgn.zst`), `pytest`, PyTorch (only for the `Dataset` wrapper at the end).

---

## Dataset contract (locked here, consumed by Plans 2–4)

A **PositionRecord** describes one position a human moved from. Fields:

| Field | Shape / type | Meaning |
|---|---|---|
| `board` | `int8[8, 8, 12]` | current side-to-move-relative board, 12 piece planes |
| `history` | `int8[7, 8, 8, 12]` | previous 7 plies (same encoding), earliest repeated if missing |
| `stm` | `int8` | side to move: 0 = white, 1 = black (board already flipped to stm) |
| `elo_self` | `int16` | Elo of the player to move |
| `elo_opp` | `int16` | Elo of the opponent |
| `temporal` | `float32[TEMPORAL_DIM]` | in-game time feature vector (layout below) |
| `move_from` | `int8` | source square 0..63 (stm-relative) |
| `move_to` | `int8` | destination square 0..63 (stm-relative) |
| `promo` | `int8` | 0 none, 1 N, 2 B, 3 R, 4 Q |
| `result` | `int8` | game result from stm view: 0 loss, 1 draw, 2 win |
| `think_time` | `float32` | seconds spent on THIS move (the MDN target); raw, ≥ 0 |

`TEMPORAL_DIM = 21`. **Temporal vector layout** (all in-game, no leakage — only moves 1..n-1):

| idx | feature |
|---|---|
| 0 | `my_clock / 180` |
| 1 | `opp_clock / 180` |
| 2 | `(my_clock - opp_clock) / 180` |
| 3 | `log1p(my_clock) / log1p(180)` |
| 4 | `log1p(opp_clock) / log1p(180)` |
| 5..9 | last-5 own think-times, `log1p(t)/log1p(30)`, most-recent first, 0-padded |
| 10..14 | last-5 own premove flags (1 if think-time < 0.1s else 0), most-recent first, 0-padded |
| 15 | `my_clock < 2` (binary) |
| 16 | `my_clock < 5` (binary) |
| 17 | `my_clock < 10` (binary) |
| 18 | `my_clock > 30` (binary) |
| 19 | own burn rate = mean of available last-5 think-times, `/30` |
| 20 | `ply / 80` (game progress proxy) |

**Think-time derivation** (increment 0): for a player's move at ply p, `think_time = prev_same_player_clock - clock_after_this_move`. White's first move: `prev = 180`. Clamp negatives to 0 (clock parsing jitter).

**Piece plane order** (index 0..11): `[P, N, B, R, Q, K, p, n, b, r, q, k]` where uppercase = side-to-move's pieces, lowercase = opponent's (because the board is flipped to stm).

---

## File structure

```
sahformer/
  __init__.py
  encoding.py        # board/move/temporal encoders — pure functions
  records.py         # PositionRecord dataclass + game -> records
  download.py        # fetch + decompress + filter Lichess shard to 3+0
  shards.py          # records -> .npz shards; balance across Elo bins
  dataset.py         # torch Dataset reading shards
tests/
  test_encoding.py
  test_records.py
  test_shards.py
  test_dataset.py
requirements.txt
```

---

## Task 0: Project scaffold

**Files:**
- Create: `requirements.txt`
- Create: `sahformer/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Write requirements.txt**

```
python-chess==1.999
numpy==2.1.0
zstandard==0.23.0
pytest==8.3.0
torch==2.4.0
```

- [ ] **Step 2: Create empty package markers**

`sahformer/__init__.py`:
```python
"""Clock-aware Chessformer for 3+0 blitz."""
```

`tests/__init__.py`:
```python
```

- [ ] **Step 3: Install and verify**

Run: `pip install -r requirements.txt && python -c "import chess, numpy, zstandard, torch; print('ok')"`
Expected: prints `ok`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt sahformer/__init__.py tests/__init__.py
git commit -m "chore: project scaffold and deps"
```

---

## Task 1: Board & move encoding

**Files:**
- Create: `sahformer/encoding.py`
- Test: `tests/test_encoding.py`

- [ ] **Step 1: Write the failing test**

`tests/test_encoding.py`:
```python
import chess
import numpy as np
from sahformer.encoding import encode_board, encode_move, PIECE_ORDER

def test_encode_board_startpos_white():
    board = chess.Board()  # white to move
    planes = encode_board(board)
    assert planes.shape == (8, 8, 12)
    assert planes.dtype == np.int8
    # 8 own pawns on rank 2 (stm=white, own pawns plane 0)
    assert planes[:, :, 0].sum() == 8
    # own king (plane 5) present exactly once
    assert planes[:, :, 5].sum() == 1
    # opponent king (plane 11) present exactly once
    assert planes[:, :, 11].sum() == 1

def test_encode_board_flips_for_black():
    board = chess.Board()
    board.push_san("e4")  # now black to move
    planes = encode_board(board)
    # after flip, side-to-move (black) own pawns still counted in plane 0
    assert planes[:, :, 0].sum() == 8
    # the just-moved white pawn is an opponent pawn (plane 6)
    assert planes[:, :, 6].sum() == 8

def test_encode_move_simple():
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    frm, to, promo = encode_move(board, move)
    assert 0 <= frm < 64 and 0 <= to < 64
    assert promo == 0

def test_encode_move_promotion():
    board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
    move = chess.Move.from_uci("a7a8q")
    frm, to, promo = encode_move(board, move)
    assert promo == 4  # queen
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_encoding.py -v`
Expected: FAIL with `ModuleNotFoundError` / `cannot import name`

- [ ] **Step 3: Write minimal implementation**

`sahformer/encoding.py`:
```python
import chess
import numpy as np

# 0..11: own P N B R Q K, then opponent p n b r q k (board is flipped to stm)
PIECE_ORDER = [
    chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING
]
_PROMO_ID = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}

def _oriented_square(sq: int, white_to_move: bool) -> int:
    """Return square index oriented so the side to move is 'at the bottom'."""
    return sq if white_to_move else chess.square_mirror(sq)

def encode_board(board: chess.Board) -> np.ndarray:
    """8x8x12 int8 planes, oriented to the side to move. Own pieces planes 0-5,
    opponent pieces planes 6-11."""
    stm_white = board.turn == chess.WHITE
    planes = np.zeros((8, 8, 12), dtype=np.int8)
    for sq, piece in board.piece_map().items():
        osq = _oriented_square(sq, stm_white)
        rank, file = divmod(osq, 8)
        own = piece.color == board.turn
        base = 0 if own else 6
        plane = base + PIECE_ORDER.index(piece.piece_type)
        planes[rank, file, plane] = 1
    return planes

def encode_move(board: chess.Board, move: chess.Move):
    """Return (from_sq, to_sq, promo_id) oriented to side to move."""
    stm_white = board.turn == chess.WHITE
    frm = _oriented_square(move.from_square, stm_white)
    to = _oriented_square(move.to_square, stm_white)
    promo = _PROMO_ID[move.promotion]
    return frm, to, promo
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_encoding.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add sahformer/encoding.py tests/test_encoding.py
git commit -m "feat: board and move encoding oriented to side-to-move"
```

---

## Task 2: Temporal feature vector

**Files:**
- Modify: `sahformer/encoding.py` (append)
- Test: `tests/test_encoding.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_encoding.py`:
```python
from sahformer.encoding import build_temporal, TEMPORAL_DIM

def test_temporal_shape_and_clocks():
    v = build_temporal(my_clock=90.0, opp_clock=120.0,
                        own_think_history=[2.0, 0.05, 5.0], ply=10)
    assert v.shape == (TEMPORAL_DIM,)
    assert v.dtype == np.float32
    assert abs(v[0] - 90.0 / 180.0) < 1e-6      # my_clock/180
    assert abs(v[1] - 120.0 / 180.0) < 1e-6     # opp_clock/180
    assert abs(v[2] - (90.0 - 120.0) / 180.0) < 1e-6

def test_temporal_premove_flag():
    # most-recent think-time first: 0.05s is a premove
    v = build_temporal(my_clock=10.0, opp_clock=10.0,
                        own_think_history=[0.05, 4.0], ply=20)
    assert v[10] == 1.0   # last move was a premove
    assert v[11] == 0.0   # the one before was not

def test_temporal_pressure_buckets():
    v = build_temporal(my_clock=1.5, opp_clock=40.0, own_think_history=[], ply=30)
    assert v[15] == 1.0   # <2s
    assert v[16] == 1.0   # <5s
    assert v[17] == 1.0   # <10s
    assert v[18] == 0.0   # >30s is about my_clock, which is 1.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_encoding.py -k temporal -v`
Expected: FAIL with `cannot import name 'build_temporal'`

- [ ] **Step 3: Write minimal implementation**

Append to `sahformer/encoding.py`:
```python
TEMPORAL_DIM = 21
_LOG180 = float(np.log1p(180.0))
_LOG30 = float(np.log1p(30.0))

def build_temporal(my_clock: float, opp_clock: float,
                   own_think_history, ply: int) -> np.ndarray:
    """own_think_history: this player's think-times for PRIOR moves, most-recent first.
    Never includes the current move (no leakage)."""
    v = np.zeros(TEMPORAL_DIM, dtype=np.float32)
    v[0] = my_clock / 180.0
    v[1] = opp_clock / 180.0
    v[2] = (my_clock - opp_clock) / 180.0
    v[3] = np.log1p(max(my_clock, 0.0)) / _LOG180
    v[4] = np.log1p(max(opp_clock, 0.0)) / _LOG180
    last5 = list(own_think_history[:5])
    for i, t in enumerate(last5):
        v[5 + i] = np.log1p(max(t, 0.0)) / _LOG30
        v[10 + i] = 1.0 if t < 0.1 else 0.0
    v[15] = 1.0 if my_clock < 2.0 else 0.0
    v[16] = 1.0 if my_clock < 5.0 else 0.0
    v[17] = 1.0 if my_clock < 10.0 else 0.0
    v[18] = 1.0 if my_clock > 30.0 else 0.0
    v[19] = (float(np.mean(last5)) / 30.0) if last5 else 0.0
    v[20] = ply / 80.0
    return v
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_encoding.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add sahformer/encoding.py tests/test_encoding.py
git commit -m "feat: in-game temporal feature vector (no leakage)"
```

---

## Task 3: Game → PositionRecords

**Files:**
- Create: `sahformer/records.py`
- Test: `tests/test_records.py`

- [ ] **Step 1: Write the failing test**

`tests/test_records.py`:
```python
import io
import chess.pgn
import numpy as np
from sahformer.records import game_to_records, PositionRecord

# 3+0 game with %clk after every move. White wins.
PGN = """[Event "Rated Blitz game"]
[White "a"]
[Black "b"]
[Result "1-0"]
[WhiteElo "1500"]
[BlackElo "1480"]
[TimeControl "180+0"]

1. e4 { [%clk 0:02:58] } e5 { [%clk 0:02:57] } 2. Nf3 { [%clk 0:02:55] } Nc6 { [%clk 0:02:50] } 3. Bb5 { [%clk 0:02:52] } a6 { [%clk 0:02:40] } 1-0
"""

def _first_game():
    return chess.pgn.read_game(io.StringIO(PGN))

def test_yields_one_record_per_move():
    recs = list(game_to_records(_first_game()))
    assert len(recs) == 6  # 3 full moves = 6 plies

def test_first_move_think_time():
    recs = list(game_to_records(_first_game()))
    # white spent 180 - 178 = 2s on move 1 (clock shows 2:58 = 178s)
    assert abs(recs[0].think_time - 2.0) < 1e-6
    assert recs[0].elo_self == 1500
    assert recs[0].elo_opp == 1480

def test_result_is_stm_relative():
    recs = list(game_to_records(_first_game()))
    # white wins -> white-to-move records have result 2 (win)
    assert recs[0].result == 2   # white's 1st move
    assert recs[1].result == 0   # black's reply -> loss from black view

def test_temporal_has_no_future_leak():
    recs = list(game_to_records(_first_game()))
    # move 1 white: no prior own think times -> last5 slots zero
    assert recs[0].temporal[5] == 0.0
    # move 2 white (index 2): prior own think time = 2.0s present
    assert recs[2].temporal[5] > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_records.py -v`
Expected: FAIL with import error

- [ ] **Step 3: Write minimal implementation**

`sahformer/records.py`:
```python
from dataclasses import dataclass
import numpy as np
import chess
from sahformer.encoding import encode_board, encode_move, build_temporal, TEMPORAL_DIM

BASE_SECONDS = 180.0  # 3+0

@dataclass
class PositionRecord:
    board: np.ndarray        # int8[8,8,12]
    history: np.ndarray      # int8[7,8,8,12]
    stm: int
    elo_self: int
    elo_opp: int
    temporal: np.ndarray     # float32[TEMPORAL_DIM]
    move_from: int
    move_to: int
    promo: int
    result: int              # stm-relative: 0 loss, 1 draw, 2 win
    think_time: float

_RESULT_WHITE = {"1-0": 2, "0-1": 0, "1/2-1/2": 1}

def _result_for_stm(result_str: str, white_to_move: bool) -> int:
    w = _RESULT_WHITE.get(result_str, 1)
    if white_to_move:
        return w
    return {0: 2, 1: 1, 2: 0}[w]  # mirror for black

def game_to_records(game):
    """Yield a PositionRecord per ply that has a clock annotation."""
    result_str = game.headers.get("Result", "1/2-1/2")
    white_elo = int(game.headers.get("WhiteElo", 0) or 0)
    black_elo = int(game.headers.get("BlackElo", 0) or 0)

    board = game.board()
    prev_clock = {chess.WHITE: BASE_SECONDS, chess.BLACK: BASE_SECONDS}
    think_hist = {chess.WHITE: [], chess.BLACK: []}  # most-recent first
    plane_hist = []  # list of int8[8,8,12], newest last

    node = game
    ply = 0
    while node.variations:
        node = node.variation(0)
        move = node.move
        mover = board.turn                      # who is about to move
        clock_after = node.clock()              # seconds left AFTER this move
        if clock_after is None:
            board.push(move)
            continue
        think = max(prev_clock[mover] - clock_after, 0.0)

        my_clock = prev_clock[mover]
        opp_clock = prev_clock[not mover]
        temporal = build_temporal(
            my_clock=my_clock, opp_clock=opp_clock,
            own_think_history=think_hist[mover], ply=ply,
        )

        cur = encode_board(board)
        hist = _stack_history(plane_hist, cur)
        frm, to, promo = encode_move(board, move)

        yield PositionRecord(
            board=cur, history=hist, stm=0 if mover == chess.WHITE else 1,
            elo_self=white_elo if mover == chess.WHITE else black_elo,
            elo_opp=black_elo if mover == chess.WHITE else white_elo,
            temporal=temporal, move_from=frm, move_to=to, promo=promo,
            result=_result_for_stm(result_str, mover == chess.WHITE),
            think_time=think,
        )

        # advance bookkeeping
        prev_clock[mover] = clock_after
        think_hist[mover] = [think] + think_hist[mover]
        plane_hist.append(cur)
        board.push(move)
        ply += 1

def _stack_history(plane_hist, current):
    """Return int8[7,8,8,12]: the 7 plies before `current`, newest first,
    earliest repeated if fewer than 7 exist."""
    out = np.zeros((7, 8, 8, 12), dtype=np.int8)
    prev = list(reversed(plane_hist[-7:]))  # newest first
    for i in range(7):
        if i < len(prev):
            out[i] = prev[i]
        elif prev:
            out[i] = prev[-1]  # repeat earliest available
        else:
            out[i] = current   # very first ply: repeat current
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_records.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add sahformer/records.py tests/test_records.py
git commit -m "feat: convert a game to per-ply position records with clocks"
```

---

## Task 4: Download & filter a Lichess shard

**Files:**
- Create: `sahformer/download.py`
- Test: `tests/test_download.py`

- [ ] **Step 1: Write the failing test** (filtering logic is unit-tested; network is not)

`tests/test_download.py`:
```python
import io
import chess.pgn
from sahformer.download import is_target_game, iter_games_from_text

MIXED_PGN = """[Event "Rated Blitz game"]
[TimeControl "180+0"]
[WhiteElo "1500"]
[BlackElo "1500"]
[Result "1-0"]

1. e4 { [%clk 0:02:58] } 1-0

[Event "Rated Bullet game"]
[TimeControl "60+0"]
[WhiteElo "1500"]
[BlackElo "1500"]
[Result "0-1"]

1. e4 e5 0-1

[Event "Rated Blitz game"]
[TimeControl "180+2"]
[WhiteElo "1500"]
[BlackElo "1500"]
[Result "1-0"]

1. e4 { [%clk 0:02:58] } 1-0
"""

def test_only_3plus0_with_clocks_pass():
    games = list(iter_games_from_text(MIXED_PGN))
    keep = [g for g in games if is_target_game(g)]
    assert len(keep) == 1
    assert keep[0].headers["TimeControl"] == "180+0"

def test_rejects_missing_clock():
    pgn = ('[TimeControl "180+0"]\n[WhiteElo "1"]\n[BlackElo "1"]\n'
           '[Result "1-0"]\n\n1. e4 1-0\n')
    g = chess.pgn.read_game(io.StringIO(pgn))
    assert is_target_game(g) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_download.py -v`
Expected: FAIL with import error

- [ ] **Step 3: Write minimal implementation**

`sahformer/download.py`:
```python
import io
import chess.pgn

TARGET_TC = "180+0"  # 3+0 blitz

def is_target_game(game) -> bool:
    if game is None:
        return False
    if game.headers.get("TimeControl") != TARGET_TC:
        return False
    if not game.headers.get("WhiteElo") or not game.headers.get("BlackElo"):
        return False
    # require at least one clock annotation in the mainline
    node = game
    while node.variations:
        node = node.variation(0)
        if node.clock() is not None:
            return True
    return False

def iter_games_from_text(text: str):
    stream = io.StringIO(text)
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            return
        yield game

def iter_games_from_zst(path: str):
    """Stream games from a Lichess .pgn.zst file without full decompression."""
    import zstandard as zstd
    with open(path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            while True:
                game = chess.pgn.read_game(text_stream)
                if game is None:
                    return
                yield game
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_download.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add sahformer/download.py tests/test_download.py
git commit -m "feat: filter Lichess games to 3+0 with clock annotations"
```

---

## Task 5: Write & balance shards

**Files:**
- Create: `sahformer/shards.py`
- Test: `tests/test_shards.py`

**Elo binning:** 22 bins. Bin index = `clip((elo_self - 600) // 100, 0, 21)` (i.e. <700→0, 700-799→1, …, ≥2600→21). Balancing caps each bin to the count of the smallest non-empty bin (`min_count`), so all present bins are equally represented.

- [ ] **Step 1: Write the failing test**

`tests/test_shards.py`:
```python
import numpy as np
from sahformer.shards import elo_bin, records_to_arrays, balance_indices
from sahformer.records import PositionRecord
from sahformer.encoding import TEMPORAL_DIM

def _rec(elo):
    return PositionRecord(
        board=np.zeros((8, 8, 12), np.int8),
        history=np.zeros((7, 8, 8, 12), np.int8),
        stm=0, elo_self=elo, elo_opp=elo,
        temporal=np.zeros(TEMPORAL_DIM, np.float32),
        move_from=0, move_to=1, promo=0, result=1, think_time=1.0,
    )

def test_elo_bin_edges():
    assert elo_bin(650) == 0
    assert elo_bin(700) == 1
    assert elo_bin(1550) == 9
    assert elo_bin(3000) == 21

def test_records_to_arrays_shapes():
    arr = records_to_arrays([_rec(1500), _rec(1500)])
    assert arr["board"].shape == (2, 8, 8, 12)
    assert arr["temporal"].shape == (2, TEMPORAL_DIM)
    assert arr["think_time"].shape == (2,)
    assert arr["think_time"].dtype == np.float32

def test_balance_caps_to_smallest_bin():
    elos = np.array([1500, 1500, 1500, 800])  # bin9 x3, bin1 x1
    idx = balance_indices(elos, seed=0)
    kept = elos[idx]
    # each present bin capped to min_count = 1
    assert (kept == 1500).sum() == 1
    assert (kept == 800).sum() == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_shards.py -v`
Expected: FAIL with import error

- [ ] **Step 3: Write minimal implementation**

`sahformer/shards.py`:
```python
import numpy as np

N_BINS = 22

def elo_bin(elo: int) -> int:
    return int(np.clip((elo - 600) // 100, 0, N_BINS - 1))

def records_to_arrays(records):
    """Stack a list of PositionRecord into a dict of batched numpy arrays."""
    n = len(records)
    out = {
        "board": np.zeros((n, 8, 8, 12), np.int8),
        "history": np.zeros((n, 7, 8, 8, 12), np.int8),
        "stm": np.zeros(n, np.int8),
        "elo_self": np.zeros(n, np.int16),
        "elo_opp": np.zeros(n, np.int16),
        "temporal": np.zeros((n, records[0].temporal.shape[0]), np.float32),
        "move_from": np.zeros(n, np.int8),
        "move_to": np.zeros(n, np.int8),
        "promo": np.zeros(n, np.int8),
        "result": np.zeros(n, np.int8),
        "think_time": np.zeros(n, np.float32),
    }
    for i, r in enumerate(records):
        out["board"][i] = r.board
        out["history"][i] = r.history
        out["stm"][i] = r.stm
        out["elo_self"][i] = r.elo_self
        out["elo_opp"][i] = r.elo_opp
        out["temporal"][i] = r.temporal
        out["move_from"][i] = r.move_from
        out["move_to"][i] = r.move_to
        out["promo"][i] = r.promo
        out["result"][i] = r.result
        out["think_time"][i] = r.think_time
    return out

def balance_indices(elo_self: np.ndarray, seed: int = 0) -> np.ndarray:
    """Return indices that equalize all present Elo bins to the smallest bin size."""
    rng = np.random.default_rng(seed)
    bins = np.array([elo_bin(int(e)) for e in elo_self])
    present = [b for b in range(N_BINS) if (bins == b).any()]
    min_count = min((bins == b).sum() for b in present)
    keep = []
    for b in present:
        idx = np.where(bins == b)[0]
        rng.shuffle(idx)
        keep.extend(idx[:min_count].tolist())
    keep = np.array(sorted(keep))
    return keep

def save_shard(path: str, arrays: dict):
    np.savez_compressed(path, **arrays)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_shards.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add sahformer/shards.py tests/test_shards.py
git commit -m "feat: shard writing and Elo-bin balancing"
```

---

## Task 6: End-to-end build script + torch Dataset

**Files:**
- Create: `sahformer/dataset.py`
- Create: `scripts/build_dataset.py`
- Test: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing test**

`tests/test_dataset.py`:
```python
import numpy as np
from sahformer.shards import records_to_arrays, save_shard
from sahformer.records import PositionRecord
from sahformer.encoding import TEMPORAL_DIM
from sahformer.dataset import ShardDataset

def _rec():
    return PositionRecord(
        board=np.zeros((8, 8, 12), np.int8),
        history=np.zeros((7, 8, 8, 12), np.int8),
        stm=0, elo_self=1500, elo_opp=1500,
        temporal=np.arange(TEMPORAL_DIM, dtype=np.float32),
        move_from=3, move_to=19, promo=0, result=2, think_time=2.5,
    )

def test_dataset_roundtrip(tmp_path):
    arr = records_to_arrays([_rec(), _rec()])
    p = tmp_path / "shard0.npz"
    save_shard(str(p), arr)
    ds = ShardDataset([str(p)])
    assert len(ds) == 2
    sample = ds[0]
    assert sample["board"].shape == (8, 8, 12)
    assert sample["temporal"].shape == (TEMPORAL_DIM,)
    assert float(sample["think_time"]) == 2.5
    assert int(sample["move_from"]) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL with import error

- [ ] **Step 3: Write minimal implementation**

`sahformer/dataset.py`:
```python
import numpy as np
import torch
from torch.utils.data import Dataset

_KEYS = ["board", "history", "stm", "elo_self", "elo_opp", "temporal",
         "move_from", "move_to", "promo", "result", "think_time"]

class ShardDataset(Dataset):
    """Loads one or more .npz shards fully into memory (Colab-sized shards)."""
    def __init__(self, shard_paths):
        self.data = {k: [] for k in _KEYS}
        for path in shard_paths:
            with np.load(path) as z:
                for k in _KEYS:
                    self.data[k].append(z[k])
        self.data = {k: np.concatenate(v, axis=0) for k, v in self.data.items()}
        self.n = self.data["board"].shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        out = {}
        for k in _KEYS:
            val = self.data[k][i]
            if k in ("board", "history", "temporal"):
                out[k] = torch.from_numpy(np.ascontiguousarray(val)).float()
            elif k == "think_time":
                out[k] = torch.tensor(float(val), dtype=torch.float32)
            else:
                out[k] = torch.tensor(int(val), dtype=torch.long)
        return out
```

`scripts/build_dataset.py`:
```python
"""Build balanced .npz shards from a Lichess .pgn.zst file.

Usage:
    python scripts/build_dataset.py INPUT.pgn.zst OUTDIR --max-games 200000 --seed 0
"""
import argparse, os
import numpy as np
from sahformer.download import iter_games_from_zst, is_target_game
from sahformer.records import game_to_records
from sahformer.shards import records_to_arrays, balance_indices, save_shard

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("outdir")
    ap.add_argument("--max-games", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    records = []
    seen = 0
    for game in iter_games_from_zst(args.input):
        if not is_target_game(game):
            continue
        records.extend(game_to_records(game))
        seen += 1
        if seen >= args.max_games:
            break

    arr = records_to_arrays(records)
    idx = balance_indices(arr["elo_self"], seed=args.seed)
    balanced = {k: v[idx] for k, v in arr.items()}
    out = os.path.join(args.outdir, "shard0.npz")
    save_shard(out, balanced)
    print(f"games={seen} positions_kept={len(idx)} -> {out}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the FULL test suite**

Run: `pytest -v`
Expected: all tests pass (17 total)

- [ ] **Step 6: Commit**

```bash
git add sahformer/dataset.py scripts/build_dataset.py tests/test_dataset.py
git commit -m "feat: torch ShardDataset and end-to-end build script"
```

---

## Task 7: Smoke-test on real Lichess data (manual)

**Files:** none (operational verification)

- [ ] **Step 1: Download one small Lichess shard**

Lichess publishes monthly `.pgn.zst` dumps at `database.lichess.org`. **Ask the user to
confirm before downloading** (large file). A recent standard-games month is tens of GB
compressed; for a smoke test, the *smallest available* monthly file (early 2013) is a few MB.

Run (after user confirms the URL/size):
```bash
curl -O https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst
```

- [ ] **Step 2: Build a tiny dataset**

Run: `python scripts/build_dataset.py lichess_db_standard_rated_2013-01.pgn.zst data/ --max-games 5000`
Expected: prints `games=... positions_kept=... -> data/shard0.npz`

Note: 2013 data predates `%clk` annotations, so `is_target_game` may reject most games.
For a clock-bearing smoke test use a **2020-or-later** month (clocks were added mid-2017).
Confirm size with the user first — later months are much larger.

- [ ] **Step 3: Sanity-check the shard**

Run:
```bash
python -c "import numpy as np; z=np.load('data/shard0.npz'); print({k:z[k].shape for k in z.files}); print('think_time mean', z['think_time'].mean())"
```
Expected: shapes match the contract; `think_time` mean is a plausible few seconds.

- [ ] **Step 4: Commit any fixes** (if the smoke test revealed parsing issues)

```bash
git add -A && git commit -m "fix: data pipeline issues found in smoke test"
```

---

## Self-review notes

- **Spec coverage:** Section 1 (data pipeline) fully covered — 3+0 filter (Task 4), clock/think-time derivation (Task 3), temporal block incl. premove flags, buckets, burn rate, last-k window (Task 2), Elo balancing (Task 5), keeps time-pressure positions (no filtering on clock — Task 3 yields every clocked ply). Board+7 history (Task 3). Raw think-time float target for the MDN (Task 3). No leakage: `own_think_history` only holds prior moves (Task 3 bookkeeping updates AFTER yield; verified by `test_temporal_has_no_future_leak`).
- **Deferred to later plans (correctly out of scope here):** skill embeddings, GAB, FiLM, heads, losses, training, evaluation.
- **Placeholder scan:** none — every step has runnable code/commands.
- **Type consistency:** `PositionRecord` fields identical across records.py, shards.py, dataset.py; `TEMPORAL_DIM=21` single source in encoding.py; `_KEYS` in dataset.py matches `records_to_arrays` keys.

## Open items for the next plan (Model)

- Confirm history is fed as 7 stacked planes concatenated on the channel dim (→ 12×8=96 channels) vs. separate tokens — Plan 2 decides and must read `board`+`history` accordingly.
- Skill-embedding interpolation endpoints (Elo 0 / 5000) and `temporal`→`t` MLP live in Plan 2.
