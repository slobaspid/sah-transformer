# Clock-Aware Chessformer for 3+0 Blitz — Design

**Date:** 2026-08-11
**Status:** Approved design, pre-implementation

## One-line summary

A clock-aware extension of Maia-3 / Chessformer that models the **average human of a
given Elo** playing **3+0 blitz** — predicting not just the move, but also **how long a
human would think** — by injecting temporal context through **FiLM + time-conditioned
Geometric Attention Bias (GAB)**.

## Goal & success criteria

**Primary goal: human *feel*, not benchmark accuracy.** The model should manage the clock
like a real 3+0 player — blitz openings, spend time on hard positions, degrade under time
pressure, premove in known lines, and scramble on the flag.

Move-matching accuracy is a **guard rail, not the objective**: the clock-aware model must
not get *dramatically* worse than a plain baseline, but small accuracy differences are
acceptable — behavioral and temporal realism is what we optimize for.

Non-goals:
- Not modeling a specific individual across games — we model an **Elo cohort**, like Maia.
- Not other time controls — **3+0 only** (fixed 180s base, 0 increment) for the first model.
- Not chasing SOTA leaderboard numbers.

## Background: what Maia-3 / Chessformer is

From the Chessformer paper (ICLR 2026) and the CSSLab/maia3 repo:

- **Board-as-tokens**: 64 squares = 64 tokens, each a 12-dim piece one-hot, plus **7
  historical positions** stacked in the channel dimension (gives move-sequence context).
- **Geometric Attention Bias (GAB)**: a learned, board-state-dependent bias added to
  attention logits (per-head 64×64 matrices). The paper's core novelty.
- **Rating conditioning**: two 128-dim skill embeddings (one per player), interpolated
  between a learnable weak (Elo 0) and strong (Elo 5000) endpoint, prepended as tokens.
- **Heads**: source→destination attention policy (64×64 + promotion bias) and a
  mean-pooled Win/Draw/Loss value head.
- **Sizes**: 5M / 23M / 79M. Trained on ~884k rating-balanced Lichess blitz positions.
- **Crucially for us**: Maia-3 **deliberately discards time** — it removed time-pressure
  positions and only carried a time-pressure flag "for infrastructure compatibility without
  performance impact." **No one has modeled clock-aware human play in this architecture.**
  That gap is this project.

## Compute plan

- **Train** the **5M** model on a **Colab free T4 (~16GB)**. Small is fine — Maia-3 shows 5M
  is respectable. Checkpoint to Google Drive constantly (Colab disconnects/time-outs).
- **Serve/play/analyze** on the user's Xeon box (32GB RAM) — even 79M runs fine on CPU for
  *inference* (~320MB fp32 weights). Never *train* 79M on CPU (compute-bound, weeks/months).
- Later: rent a bigger GPU for a day to scale 5M → 23M/79M if the idea proves out.

---

## Section 1 — Data pipeline

- **Source**: Lichess open database, filtered to **3+0 rated blitz** with `%clk` clock
  annotations. Data is abundant (tens of millions of games/month) — compute/disk is the
  bottleneck, not data.
- **Per-position features:**
  - Board + 7 history planes (Maia-3 style).
  - Both players' Elo → skill embeddings (unchanged from Maia-3).
  - **Temporal feature block** (raw, in-game observations only):
    - log clocks for both sides (`log(t+1)`, normalized), plus clock **differential**
    - **last-k think-times** (k≈3–5) of the current game, raw — lets the model derive
      in-game rhythm itself
    - **premove flags** on recent moves (think-time < ~100ms)
    - **time-pressure buckets / RBF features** near zero (`<2s`, `<5s`, `<10s`, `>30s`) so
      the net can represent nonlinear panic thresholds
    - burn rate (how fast each side spends time)
- **Targets per position:** (a) move played, (b) game result W/D/L, (c) **this move's
  think-time** stored as a **raw float (seconds)** — the MDN head consumes it directly in
  log-space; no bucketing.
- **Key difference from Maia-3**: we **KEEP** time-pressure and flag-scramble positions —
  that's the whole point.
- **Balancing**: equalize 22 Elo bins (Maia-3 style). Start ~1M positions for a Colab run.

### Design principle: get the most signal out of time

Raw "seconds left" is the weakest use of time. Higher-signal ideas, folded into the block above:

1. **The Elo embedding owns the pace baseline.** A 1200 and a 2200 have different natural
   tempos, panic thresholds, and premove habits. We do NOT hand-engineer a per-individual
   baseline (no persistent identity). Instead we feed **absolute in-game observations** and
   let the model compare them against its **Elo-conditioned expectation** internally — the
   relativity is emergent, not hand-coded.
2. **Rhythm over snapshots** — feed the recent *sequence* of think-times, not just the last.
3. **Nonlinear near zero** — log + buckets/RBF, because human play has hard cliffs (<3s ≈ reflex).
4. **Premove detection** — sub-100ms is qualitatively different (decided before opponent moved).
5. **No leakage** — this move's think-time is a **target**; only think-times of moves 1…n−1
   may be inputs.

---

## Section 2 — Model architecture

Backbone = **Chessformer 5M, unchanged**. We add three things around it.

**Two conditioning signals:**
1. **Skill/pace embedding** (unchanged): both players' Elo → interpolated 128-dim
   embeddings. Owns the baseline pace / "time personality."
2. **Temporal context vector `t`** (new): the Section-1 temporal block → small MLP → ~128-dim.

**How time modulates the network (FiLM + GAB — the chosen approach):**
- **Time-conditioned GAB**: concatenate `t` (and the skill embedding) into GAB's compression
  input, so the **geometric attention itself becomes clock-aware** — under time pressure the
  net can narrow attention onto forcing moves/captures; in calm positions it spreads out.
  *This is the core idea: time reshapes how the board is read.*
- **FiLM on encoder blocks**: `t` → generator → per-layer **(γ, β)** scaling/shifting each
  transformer block's activations. A cheap second knob for reprocessing under time pressure.

**Three heads:**
1. **Policy** (unchanged): source→destination 64×64 attention + promotion bias.
2. **Value** (unchanged): mean-pool → W/D/L.
3. **Think-time head** (new): mean-pool encoder output (+ `t`, + skill embedding) → MLP →
   **Mixture Density Network** outputting the parameters of a mixture of **log-normals** (e.g.
   M=3 components → mixture weights `π`, means `μ`, spreads `σ` in log-time space). This gives
   a **continuous, multimodal, samplable** think-time distribution — it can represent "60%
   premove ~0.05s, 40% real think ~4s" instead of collapsing to a meaningless mean. Predicting
   in **log-time** space naturally handles the heavy tail; the premove spike becomes one
   low-mean component.

New parameters are only: temporal MLP, FiLM generator, time head. Conditioning stays faithful
to Maia-3.

---

## Section 3 — Losses & training

**Combined loss:**
- **Policy** — cross-entropy on move, weight **1.0** (main objective).
- **Value** — cross-entropy on W/D/L, weight **~0.1** (Maia-3 default).
- **Think-time** — **negative log-likelihood** of the observed think-time under the predicted
  log-normal mixture (MDN loss), weight **~0.2–0.3** (tuned: high enough to learn time, low
  enough not to hurt move accuracy). Numerical stability: compute NLL in log-space with a
  log-sum-exp over components; floor `σ` with a softplus + epsilon.

Move-accuracy is watched as a guard metric; if the time loss tanks it vs. baseline, turn the
weight down.

**Optimizer / schedule** (mirror Maia-3, scaled for Colab):
- AdamW, cyclic cosine annealing, batch 128 (→64 if T4 is tight), **fp16 mixed precision**.
- Not 1M steps — as many as fit a Colab session, **checkpoint to Drive** so runs resume
  across disconnects.

**Staged curriculum (always leaves a working checkpoint):**
1. Plain 5M → roughly reproduce Maia-3-ish move accuracy. **The baseline.**
2. + think-time head → predicts time above chance?
3. + FiLM → move accuracy in time-pressure positions improves?
4. + time-conditioned GAB → the full model.

If stage 4 gets finicky, stages 2/3 still yield a publishable clock-aware model.

---

## Section 4 — Evaluation

**Success = human feel.** Ordered by importance:

1. **Behavioral realism (headline).** Simulate full 3+0 games using both move and time heads.
   Check emergent human patterns: openings blitzed, hard middlegames eat time, flag scrambles
   produce fast/degraded moves. Compare model clock trajectories against real 3+0 games.
2. **Time-pressure slices.** Bucket test positions by clock (`>60s`, `10–60s`, `3–10s`, `<3s`)
   and report move-match per bucket. Hypothesis: clock-awareness helps most in **low-time**
   buckets — exactly where Maia-3 is blind.
3. **Think-time head quality.** Bucket accuracy / top-3 vs. a median-predicting baseline;
   calibration (predicted "long think" ↔ humans actually think long); sanity (time rises in
   tactical positions, falls in forced recaptures).
4. **Interpretability (pays off FiLM/GAB).** Visualize GAB attention for the *same position*
   at high vs. low clock — attention should visibly **narrow onto forcing moves** under time
   pressure. Concrete, publishable picture of the model "playing faster."
5. **Accuracy ablation (guard rail, demoted).** Clock-aware vs. no-time baseline on overall
   move-match — only to confirm we didn't get dramatically worse. Small deltas are fine.

---

## Open questions / to settle during planning

- Exact k for the think-time window, number of MDN components M, and normalization constants.
- Whether to reimplement Chessformer from scratch or adapt CSSLab/maia3's open model code.
- Precise Lichess month(s) to download and target dataset size vs. Colab epoch time.
