"""Stream a 3+0 sample from a Lichess .pgn.zst URL into chunked shards.

Only the first chunk of the archive is transferred (we early-stop at --max-positions),
and memory stays flat regardless of dataset size (writes one chunk at a time).

Usage:
    python scripts/fetch_sample.py URL OUTDIR --max-positions 300000 --chunk 150000
"""
import argparse
from sahformer.download import stream_games_from_url
from sahformer.dataset_build import build_shards, records_from_games

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("outdir")
    ap.add_argument("--max-positions", type=int, default=300000)
    ap.add_argument("--chunk", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--balance", action="store_true", help="Elo-balance each chunk")
    args = ap.parse_args()

    paths = build_shards(
        records_from_games(stream_games_from_url(args.url)),
        args.outdir, chunk_positions=args.chunk, max_positions=args.max_positions,
        balance=args.balance, seed=args.seed, progress_every=2000)
    if not paths:
        raise SystemExit("no target (3+0 with clocks) games found — try a 2017-04+ month")
    print(f"wrote {len(paths)} shard(s) to {args.outdir}")

if __name__ == "__main__":
    main()
