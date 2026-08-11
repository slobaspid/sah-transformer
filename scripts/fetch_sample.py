"""Stream a tiny 3+0 sample from a Lichess .pgn.zst URL into one balanced shard.

Only the first chunk of the archive is transferred (we early-stop at --max-games).

Usage:
    python scripts/fetch_sample.py URL data/ --max-games 2000 --seed 0
"""
import argparse
import os
from sahformer.download import stream_games_from_url, is_target_game
from sahformer.records import game_to_records
from sahformer.shards import records_to_arrays, balance_indices, save_shard

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("outdir")
    ap.add_argument("--max-games", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    records = []
    kept = 0
    for game in stream_games_from_url(args.url):
        if not is_target_game(game):
            continue
        records.extend(game_to_records(game))
        kept += 1
        if kept >= args.max_games:
            break

    if kept == 0:
        raise SystemExit("no target (3+0 with clocks) games found — try a 2017-04+ month")
    arr = records_to_arrays(records)
    idx = balance_indices(arr["elo_self"], seed=args.seed)
    balanced = {k: v[idx] for k, v in arr.items()}
    out = os.path.join(args.outdir, "sample.npz")
    save_shard(out, balanced)
    print(f"target_games={kept} positions_kept={len(idx)} -> {out}")

if __name__ == "__main__":
    main()
