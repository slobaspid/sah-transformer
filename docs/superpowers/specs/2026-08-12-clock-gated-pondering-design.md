# Clock-Gated Pondering Design (adaptive computation)

> Sub-design of the clock-aware Chessformer project. Parent spec:
> `2026-08-11-clock-aware-chessformer-design.md`. Adds **adaptive computation** to the model:
> it refines its read of the *current* position a variable number of internal steps, with a
> learned halting gate that is **modulated by the remaining clock** — think deeper with time,
> cut it short under time pressure. Complements (does not replace) the external search.

## North star

Human realism, not strength ([[human-feel-over-accuracy]]). Pondering is a second, *internal*
"effort" axis (the search is the *external* one): the model spends more internal computation on
hard positions and **less when its clock is low** — mirroring how a human's depth of thought
collapses in a time scramble. The number of ponder steps becomes a measurable "how hard did I
think" signal that also **deepens the brain the search runs on**.

## Background & how it ties to the search

Two different kinds of thinking (see the research thread):
- **Search (already built):** looks *ahead* — explores future positions (external, no training).
- **Pondering (this design):** looks *deeper* at the *current* position — refines the internal
  representation (internal, trained into the model).

They stack: a pondering model is a *deeper brain*, and the search evaluates positions with that
deeper brain. Grounded in adaptive-computation research — **ACT** (Graves 2016), **PonderNet**
(Banino 2021), and the 2025–26 looped/recurrent-depth line (Ouro's confidence-based early exit,
etc.). We use the **PonderNet** formulation: a per-step halting probability, a probability-weighted
loss over steps, and a geometric prior that stops the halting from degenerating.

## Architecture

Built on the clock-aware backbone. New pieces:

### `PonderBlock` (shared refinement step)
One pre-norm transformer-style block — RMSNorm → multi-head self-attention → RMSNorm → MLP —
operating on the encoder's token output `enc (B,64,dim_vit)`. **Weights are shared across ponder
steps** (it's a *recurrent* refinement, not new layers per step). Refines the position's read
without any look-ahead.

### `PonderChessformer`
```
tok   = InputEmbedding(board, skill)
t     = TemporalEncoder(temporal)          # carries the remaining-clock features
enc0  = Encoder(tok, t, film)              # the initial read (clock-aware backbone, unchanged)
h_0   = enc0
for n in 0..K-1:                           # up to max_ponder steps
    lambda_n = sigmoid(HaltHead(mean(h_n), t))   # P(halt at step n | reached it) — CLOCK-GATED
    (move_n, value_n, mdn_n) = heads(h_n)        # per-step predictions (for training)
    h_{n+1} = PonderBlock(h_n)              # refine
```
- **`HaltHead`**: small MLP on `[mean-pooled h_n ; t]` → scalar → sigmoid. Feeding `t` (the clock
  context) is the *clock gate*: low clock biases `lambda_n` high (halt early).
- **Heads reused as-is** (policy 4352, value W/D/L, MDN think-time + the difficulty feature).

### Halting probabilities (PonderNet)
`p_n = lambda_n * Π_{m<n}(1 - lambda_m)` — the probability the process *halts exactly at step n*.
The last step's `lambda` is forced to 1 so probabilities sum to 1 over the `K` steps.

## Training (PonderNet loss)

Per **sample** (not just batch mean), so we can weight by that sample's halting probs:
```
L = Σ_n  p_n · per_sample_loss(step_n_outputs, target)     # expected task loss over halting step
  + β · KL( p  ||  Geometric(prior_lambda) )               # keep halting near a target depth
```
- `per_sample_loss` = policy CE + w_v·value CE + w_t·MDN-NLL, reduced **per row** (shape `(B,)`).
- `Geometric(prior_lambda)` over `K` steps (e.g. `prior_lambda = 0.4` → expected ~2–3 steps). The
  KL term is the guardrail against the two PonderNet failure modes (always-halt / never-halt).
- `β` (KL weight, ~0.01) and `prior_lambda` are tunable.

## Inference (drop-in, clock-budgeted)

`PonderChessformer.forward(batch)` must return the **same single-output dict** the other models do
(`move_logits`, `value_logits`, `mdn`) so `play`, `search`, and the UCI engine work unchanged.
At inference it runs the loop and **halts** by whichever comes first:
- cumulative halt probability crosses a threshold (e.g. 0.5), **or**
- a **clock-derived step cap** (few steps when the clock is low, up to `max_ponder` when high) — the
  explicit clock gate at play time, mirroring the search's time→budget mapping.

It returns the halting step's head outputs, **plus `ponder_steps`** (the count) as the measurable
"how hard did I think" readout. A separate `ponder_train(batch)` returns the per-step outputs +
`p_halt (B,K)` for the training loss.

## Integration

```
sahformer/model/
  ponder.py        # PonderBlock, HaltHead, PonderChessformer, ponder_forward/ponder_train
sahformer/training/
  build.py         # (+) mode "ponder" -> PonderChessformer
  losses.py        # (+) per_sample_loss, ponder_loss (PonderNet weighting + KL prior)
  loop.py          # (+) when mode == "ponder", train with ponder_loss
```
- `build_model("ponder", cfg)` returns a `PonderChessformer`. `SearchConfig`/play/engine are
  unchanged (they call `model(batch)` and read the drop-in dict).
- `ModelConfig` gains `max_ponder` (default 4) and `ponder_prior` (default 0.4).

## Guardrails / risks (honest)

- **This is the largest, riskiest change yet.** Pondering is finicky to train — the KL prior is
  essential, and it can still collapse. Start with `max_ponder = 4` and a modest `β`.
- **Requires a full retrain from scratch** (new architecture, new weights).
- **Payoff is uncertain** — it may make positional/timing play more human, or be a wash. It's a
  genuine experiment; the honest test is the eval stage.
- **Keep the think-time head separate** (don't equate ponder-count with think-time yet) — measure
  their relationship later instead of coupling training.
- Cost: heads are computed per ponder step during **training** (≈ K× head compute); small `K`
  keeps it cheap. Inference halts early on easy positions.

## Testing (TDD)

- `PonderBlock` preserves shape and has gradients.
- Halting probs `p_n` are non-negative and **sum to 1** over the K steps (last lambda forced to 1).
- `HaltHead` responds to the clock: a low-clock temporal vector yields higher early-halt
  probability than a high-clock one (the clock gate works).
- `ponder_train` returns K per-step outputs + `p_halt (B,K)`; `ponder_loss` is finite and
  backward-able; a single-step degenerate case matches the plain loss.
- `PonderChessformer.forward` is a **drop-in**: returns `move_logits (B,4352)`, `value_logits (B,3)`,
  `mdn`, and `ponder_steps`; `search`/`self_play` run with mode "ponder" unchanged.
- End-to-end: a few ponder training steps on a tiny shard reduce the loss.

## Deferred

- Tie `ponder_steps` to the MDN think-time (measure correlation first, in the eval plan).
- Per-token halting (halt different squares at different depths) — a more advanced variant.
- Measuring **thought-depth vs Elo** using `ponder_steps` (the eval-plan question) — pondering
  gives a second, internal depth signal alongside search-sims.
