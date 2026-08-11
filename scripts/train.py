"""Train a (clock-aware) Chessformer on shard(s).

Usage:
    python scripts/train.py "data/*.npz" --mode full --max-steps 2000 --out checkpoints/
"""
import argparse
import glob
from sahformer.training.loop import TrainConfig, train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", help="glob for .npz shard(s)")
    ap.add_argument("--mode", default="full",
                    choices=["baseline", "film_only", "gab_only", "full"])
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.shards))
    if not paths:
        raise SystemExit(f"no shards matched: {args.shards}")
    cfg = TrainConfig(mode=args.mode, max_steps=args.max_steps, batch_size=args.batch_size,
                      lr=args.lr, out_dir=args.out, amp=args.amp, device=args.device)
    res = train(cfg, paths)
    print(f"done. best_total={res['best']:.4f} steps={len(res['history'])}")

if __name__ == "__main__":
    main()
