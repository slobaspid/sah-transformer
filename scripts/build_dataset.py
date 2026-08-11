"""Build chunked .npz shards from a local Lichess .pgn.zst file.

Memory stays flat regardless of dataset size (writes one chunk at a time).

Usage:
    python scripts/build_dataset.py INPUT.pgn.zst OUTDIR --max-positions 300000 --chunk 150000
"""
import argparse
from sahformer.download import iter_games_from_zst
from sahformer.dataset_build import build_shards, records_from_games

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("outdir")
    ap.add_argument("--max-positions", type=int, default=300000)
    ap.add_argument("--chunk", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--balance", action="store_true", help="Elo-balance each chunk")
    args = ap.parse_args()

    paths = build_shards(
        records_from_games(iter_games_from_zst(args.input)),
        args.outdir, chunk_positions=args.chunk, max_positions=args.max_positions,
        balance=args.balance, seed=args.seed, progress_every=2000)
    print(f"wrote {len(paths)} shard(s) to {args.outdir}")

if __name__ == "__main__":
    main()
