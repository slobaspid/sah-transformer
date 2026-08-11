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
