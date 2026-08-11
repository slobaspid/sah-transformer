import os
import gc
from sahformer.download import is_target_game
from sahformer.records import game_to_records
from sahformer.shards import records_to_arrays, balance_indices, save_shard

def records_from_games(games):
    """Filter to target (3+0 with clocks) games and expand each into PositionRecords,
    one at a time (streaming — never materializes the whole list)."""
    for g in games:
        if is_target_game(g):
            yield from game_to_records(g)

def _flush(buf, idx, outdir, prefix, balance, seed):
    arr = records_to_arrays(buf)
    if balance:
        keep = balance_indices(arr["elo_self"], seed=seed)
        arr = {k: v[keep] for k, v in arr.items()}
    path = os.path.join(outdir, f"{prefix}{idx:04d}.npz")
    save_shard(path, arr)
    return path

def build_shards(records, outdir, chunk_positions=150000, max_positions=None,
                 balance=False, seed=0, prefix="shard", progress_every=0):
    """Stream PositionRecords to chunked .npz shards on disk.

    Never holds more than `chunk_positions` records in memory at once, so peak RAM stays
    flat no matter how large the total dataset is. When `balance` is True each chunk is
    Elo-balanced independently. Returns the list of shard paths written.
    """
    os.makedirs(outdir, exist_ok=True)
    paths = []
    buf = []
    total = 0
    shard_i = 0
    for r in records:
        buf.append(r)
        total += 1
        if progress_every and total % progress_every == 0:
            print(f"positions {total} | shards written {len(paths)}")
        if len(buf) >= chunk_positions:
            paths.append(_flush(buf, shard_i, outdir, prefix, balance, seed))
            shard_i += 1
            buf = []
            gc.collect()
        if max_positions is not None and total >= max_positions:
            break
    if buf:
        paths.append(_flush(buf, shard_i, outdir, prefix, balance, seed))
    return paths
