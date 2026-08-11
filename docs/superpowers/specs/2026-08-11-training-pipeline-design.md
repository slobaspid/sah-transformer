# Training Pipeline Design (Plan 4 of 5)

> Sub-design of the clock-aware Chessformer project. Parent spec:
> `2026-08-11-clock-aware-chessformer-design.md`. Consumes the Plan 1 data pipeline
> and the Plan 2 v2 / Plan 3 v2 models. Terminal deliverable: a trainable pipeline
> proven on a **tiny real 3+0 shard**, with the baseline-vs-clock-aware ablation wired in.

## Goal

Fit our model on a small real 3+0 blitz shard, report **move-accuracy** and **think-time
NLL**, and support the core experiment: does the time layer help versus the clock-blind
baseline? All machinery smoke-tested on a tiny shard; the same code scales to a full run
later (deferred to a future plan). No multi-GPU, no hyperparameter search, no eval-vs-Maia
benchmark here.

## Data acquisition (decided)

`is_target_game` requires TimeControl `180+0`, both Elos, and at least one `%clk` clock
annotation. Lichess exports include clocks only from **2017-04 onward**, so the earliest
usable month is 2017-04. We avoid a full multi-GB download with a **bounded streaming
fetch**: stream the `.pgn.zst` over HTTP, decompress on the fly, and early-stop after N
target games — transferring only the first ~tens of MB.

- **`sahformer/download.py` (+)**: refactor the zst parsing into one core
  `_iter_games(reader)` shared by the existing `iter_games_from_zst(path)` and a new
  `stream_games_from_url(url)` (streams the HTTP response body through the zstd stream
  decompressor). No behavior change to the local-file path.
- **`scripts/fetch_sample.py`**: stream from a URL, filter with `is_target_game`, collect
  up to `--max-games`, balance by Elo (`balance_indices`), and `save_shard` into `data/`.
  Run **once**, manually, with explicit user confirmation of the exact URL and observed
  size (download = explicit-permission action).

## Model ablation (decided: keep all four modes)

One switch selects the run's model. Two new boolean flags on `ClockAwareChessformer`
(`use_film`, `use_time_gab`) express the middle modes; the baseline stays the separate
`FaithfulChessformer` class.

| mode        | model                | time-conditioned GAB | FiLM |
|-------------|----------------------|----------------------|------|
| `baseline`  | FaithfulChessformer  | —                    | —    |
| `film_only` | ClockAwareChessformer| off                  | on   |
| `gab_only`  | ClockAwareChessformer| on                   | off  |
| `full`      | ClockAwareChessformer| on                   | on   |

- `use_time_gab` selects `Encoder(cfg, time_conditioned=…)` at construction and whether `t`
  is passed to the encoder for GAB conditioning.
- `use_film` gates whether the FiLM `(γ,β)` tuple is passed into `encoder.forward`.
- In every clock-aware mode the temporal encoder still runs and `t` is fused into the MDN
  pooled vector (the think-time head is always clock-aware; the ablation is about how the
  clock reshapes the **trunk**).
- `build_model(mode, cfg)` in `sahformer/training/build.py` returns the right module.

## Losses & metrics

`sahformer/training/losses.py`:
- `move_target_index(move_from, move_to, promo) -> LongTensor`: vectorized form of
  `heads.move_to_index` (non-promo `from*64+to`; promo `4096 + to*4 + (promo-1)`).
- `compute_losses(out, batch, weights) -> dict`: cross-entropy policy loss (4352 classes),
  cross-entropy value loss (W/D/L), MDN NLL time loss (`mdn_nll`), and
  `total = w_p·policy + w_v·value + w_t·time`. Default weights policy 1.0 / value 0.1 /
  time 0.2 (configurable via `TrainConfig`).
- `move_accuracy(move_logits, target) -> float`: top-1 accuracy, the guard metric compared
  across ablations.

## Training loop

`sahformer/training/loop.py`:
- `TrainConfig` dataclass: `mode`, `lr` (default 3e-4), `weight_decay` (0.01), `batch_size`
  (128), `max_steps`, `warmup_steps`, `w_policy/w_value/w_time`, `amp` (bool), `device`,
  `seed`, `out_dir`, `log_every`, `ckpt_every`.
- `train(cfg, shard_paths) -> dict`: builds model via `build_model`, `ShardDataset` +
  `DataLoader`, AdamW, linear-warmup→cosine LR schedule, optional AMP (`torch.cuda.amp`),
  step loop over batches (re-iterating the loader until `max_steps`), periodic logging of
  `{step, lr, policy, value, time, total, move_acc}`, and checkpointing.
- **Checkpoint**: `save_checkpoint(path, model, cfg, step, best_metric)` writes
  `{model_state, cfg, step, best_metric}`; the loop saves `out_dir/last.pt` every
  `ckpt_every` and `out_dir/best.pt` on a new best (lowest total loss).
  `load_checkpoint(path, model)` restores the model state.

## CLI

`scripts/train.py`: args `--shards` (glob), `--mode`, `--max-steps`, `--batch-size`,
`--lr`, `--out`, `--amp`. Loads shards, constructs `TrainConfig`, calls `train`.

## Data flow

`shard.npz → ShardDataset → DataLoader → batch dict → model(batch) →
{move_logits, value_logits, mdn} → compute_losses(targets from move_from/to/promo, result,
think_time) → backward → AdamW.step`. Clock-aware modes additionally consume `batch["temporal"]`.

## File structure

```
sahformer/
  download.py            # (+) _iter_games, stream_games_from_url
  training/
    __init__.py
    build.py             # build_model(mode, cfg)
    losses.py            # move_target_index, compute_losses, move_accuracy
    loop.py              # TrainConfig, train, save/load_checkpoint
scripts/
  fetch_sample.py        # bounded streaming fetch -> data/ shard
  train.py               # training CLI
tests/
  test_download_stream.py
  training/
    test_losses.py
    test_build.py
    test_loop.py
```

## Testing (TDD)

- `move_target_index` vectorized == scalar `move_to_index` across promo/non-promo cases.
- `compute_losses` returns finite total, backward populates grads; `move_accuracy` ∈ [0,1].
- **Overfit test**: one tiny synthetic batch, ~50 steps → total loss strictly decreases
  (proves the loop actually learns).
- All four ablation modes build, forward, and backward; `baseline` has no temporal params,
  `full` responds to the clock (reuse the time-sensitivity check).
- `save_checkpoint`/`load_checkpoint` round-trip: restored params equal originals.
- Streaming parser: build an in-memory zstd byte stream of a couple PGNs, feed the shared
  `_iter_games` core → yields those games (no real HTTP in tests).

## Deferred (future plans)

Full-scale training run, Colab/Drive checkpoint targets, learning-rate/loss-weight sweeps,
and the eval-vs-Maia move-matching + think-time-distribution benchmark.
