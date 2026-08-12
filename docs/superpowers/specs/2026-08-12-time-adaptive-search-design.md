# Time-Adaptive Search Design ("ALLIE-lite" on our model)

> Sub-design of the clock-aware Chessformer project. Parent spec:
> `2026-08-11-clock-aware-chessformer-design.md`. Adds a search step on top of the trained
> model so it *calculates more on hard positions* — the ALLIE idea — reusing the policy,
> value, and MDN think-time heads we already have.

## North star (read this first)

The goal of this project is **human realism, not raw strength** ([[human-feel-over-accuracy]]).
So this search is **not** here to make the bot play the objectively best move. It is here to make
the bot **spend effort the way a human does** — glance at easy positions, calculate on hard ones
(sacrifices, only-moves, sharp tactics) — and, as a side effect, play the *human-plausible*
resolution of those hard positions a bit more reliably. Every design choice below is budgeted to
**keep it inside the human distribution**, not to climb toward engine strength.

## Background: Maia vs ALLIE, and where we sit

- **Maia** = board-input model, predicts the human *move* per skill level, **no time, no search**.
- **ALLIE** = move-sequence model with policy + think-time + value heads, and a **time-adaptive
  MCTS**: it predicts how long a human would think, then searches proportionally (more search on
  positions humans ponder). Reported think-time correlation r≈0.70; ~49 Elo human-alignment gap.
- **Us** = Maia's board backbone + skill dial + ALLIE's three heads (policy/value/MDN-time) +
  a **live clock-awareness layer neither has**. The one missing ALLIE piece is the **search**.

This design adds that missing piece, giving us: looks at the real board (Maia) + calculates-more-
on-hard-positions (ALLIE) + manages its clock under pressure (ours).

## What we already have (reused, not rebuilt)

- **Policy head** → prior probability over moves (the search's move-ordering guide).
- **Value head** → W/D/L; converted to a scalar `v = P(win) − P(loss) ∈ [−1, 1]` for leaf eval.
- **MDN think-time head** → expected seconds; drives the **search budget**.
- `move_to_index` / `encode_move` → score legal moves; `self_play`'s input construction → build
  model inputs for any board.

## The search: budgeted PUCT MCTS

A standard AlphaZero-style **PUCT Monte-Carlo Tree Search**, single-threaded, using the value head
as the leaf evaluator (no random rollouts to game end). One "simulation" = descend the tree by
PUCT, expand a leaf, evaluate it with the value head, back up the value.

- **Selection (PUCT):** at each node pick the child maximizing
  `Q(s,a) + c_puct · P(s,a) · √(ΣN(s,b)) / (1 + N(s,a))`, where `P` is the policy prior for that
  move, `N` visit counts, `Q` the mean backed-up value. `c_puct ≈ 1.5` (tunable).
- **Expansion + evaluation:** run the model once on the leaf position (board+history+clock+elo),
  cache its policy prior over that node's legal moves and its value `v`.
- **Backup:** propagate `v` up the path, flipping sign each ply (side-to-move perspective).
- **Move choice:** after the budget is spent, play the **most-visited** child (optionally sampled
  by visit counts with a temperature, to preserve human variety — see guardrails).

Legal-move handling reuses `move_to_index(*encode_move(board, m))` to gather each legal move's
prior from the 4352-way policy, exactly as `self_play` already does.

## Time → search budget (the ALLIE trick)

The MDN head predicts seconds; convert to a simulation count:

```
sims = clamp(round(predicted_seconds * SIMS_PER_SECOND), MIN_SIMS, MAX_SIMS)
```

- `SIMS_PER_SECOND` (default ~16): a blitz "obvious move" (~0.2s) → ~3 sims (basically the raw
  policy); a hard position (~8s) → ~128 sims (real calculation).
- `MIN_SIMS = 1` (fall back to the raw policy on instant moves — keeps it fast and human).
- `MAX_SIMS` (default ~256): a **human-realism cap**, not just a compute cap — humans don't
  calculate 10,000 lines in blitz, and we don't want the bot to either.

So the *depth of calculation tracks the human think-time we already predict* — the core idea.
And because a sacrifice drives predicted think-time up, it automatically gets more search →
a better-checked reply → and it takes longer. The exact behavior you wanted, with no
sacrifice-detector.

## Human-realism guardrails (critical)

Search naturally pulls toward *stronger* play; we must keep it *human*:

1. **Budget is small and human-time-scaled** (`MAX_SIMS` cap above). Little/no search where humans
   blitz; modest search where they ponder. This is the primary guardrail.
2. **Priors stay human.** The tree is guided by the *human* policy and evaluated by the
   *human-outcome* value — both already in-distribution, so the search explores human-plausible
   lines, not engine lines.
3. **Move selection keeps variety.** Sample from visit counts with a temperature (reuse the
   existing `temperature` knob) rather than always the argmax, so it still plays the occasional
   second-choice human move ([[human-feel-over-accuracy]]).
4. **A `search` on/off switch** everywhere, default configurable. Pure-policy play (current
   behavior) stays available and is the honest baseline for "did search help human-realism?".

## Integration

```
sahformer/
  search.py         # mcts_move(model, board, plane_hist, clocks, elo, cfg) -> (move, stats)
                    #   + time_to_sims(seconds, cfg)
```

- **`self_play` / viewer:** a `search=True` option routes move choice through `mcts_move`; the
  predicted think-time still drives the on-screen pacing (now it *also* set the search depth).
- **`uci_engine.py`:** a `Search` UCI option (sims budget or on/off). The existing `Pace` still
  governs the visible/clock time; search governs the move quality on hard positions.
- Reuses the input-building and think-time sampling already in `play.py` (factor the shared bits
  into a small helper so `search.py` and `self_play` don't duplicate).

## Cost & limits (honest)

- **Each simulation is a model forward pass.** On CPU, ~a few ms each → a 128-sim hard move is
  ~0.3–1s of real compute. Fine for the viewer and analysis; heavier for live blitz (cap sims or
  run on GPU).
- **This does not add a genuine "calculation process" to the model** — it bolts search *around*
  a one-shot model (like AlphaZero/ALLIE). The model still doesn't *learn* to think; we're
  spending external compute guided by its heads. That's the honest scope: it's ALLIE-lite, not a
  from-scratch pondering network.
- Search can only reach human-plausible lines the policy/value already know; it won't conjure
  engine-only ideas (which is *good* for realism).

## Testing (TDD)

- `time_to_sims`: monotonic; clamps to `[MIN_SIMS, MAX_SIMS]`; ~0.2s → few sims, ~8s → many.
- `mcts_move` returns a **legal** move and visit stats; more sims never crashes; works from the
  start position and from a mid-game FEN.
- **Behavioral:** on a position with a single obvious winning capture (free queen), a modest
  search should concentrate visits on that capture more than the raw policy does — i.e. search
  *sharpens* the obvious tactic. (A gentle check that search is doing something sane, not a
  strength benchmark.)
- **Determinism:** with a fixed seed and `temperature=0`, `mcts_move` is reproducible.
- Integration: `self_play(search=True)` and the UCI engine with `Search` on both produce legal,
  complete games.

## Deferred / open questions

- **The real "pondering network"** (the model learns variable internal computation, Flavor 1) —
  a bigger, separate research effort; this MCTS is the pragmatic stepping stone.
- **Tuning** `c_puct`, `SIMS_PER_SECOND`, `MAX_SIMS`, and the move-selection temperature against
  *human-alignment* metrics (not strength) once we have a held-out eval.
- **Does search actually improve human-realism?** — needs the eval plan: compare pure-policy vs
  search on move-match *and* on the time-pressure / sacrifice subset, on unseen games.
- Batched/parallel simulations and a transposition table for speed (optimization, later).
