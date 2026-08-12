import bisect
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

_KEYS = ["board", "history", "stm", "elo_self", "elo_opp", "temporal",
         "move_from", "move_to", "promo", "result", "think_time"]

def _to_tensors(d, i):
    out = {}
    for k in _KEYS:
        val = d[k][i]
        if k in ("board", "history", "temporal"):
            out[k] = torch.from_numpy(np.ascontiguousarray(val)).float()
        elif k == "think_time":
            out[k] = torch.tensor(float(val), dtype=torch.float32)
        else:
            out[k] = torch.tensor(int(val), dtype=torch.long)
    return out

class ShardDataset(Dataset):
    """Loads one or more .npz shards into memory and indexes across them without
    concatenating, so peak RAM is ~1x the data (not 2x). Use for datasets that fit in RAM."""
    def __init__(self, shard_paths):
        self.shards = []        # one dict of arrays per shard
        self._cum = []          # cumulative lengths, for global i -> (shard, row)
        total = 0
        for path in shard_paths:
            with np.load(path) as z:
                d = {k: z[k] for k in _KEYS}
            total += d["board"].shape[0]
            self.shards.append(d)
            self._cum.append(total)
        self.n = total

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = bisect.bisect_right(self._cum, i)
        base = self._cum[s - 1] if s > 0 else 0
        return _to_tensors(self.shards[s], i - base)

class StreamingShardDataset(IterableDataset):
    """Streams samples from shards a few at a time, so training RAM stays flat regardless of
    total dataset size. Loads `buffer_shards` shards, shuffles within them, yields, repeats.
    Use for datasets too big to hold in RAM (millions of positions)."""
    def __init__(self, shard_paths, shuffle=True, buffer_shards=2, seed=0):
        self.paths = list(shard_paths)
        self.shuffle = shuffle
        self.buffer_shards = max(1, buffer_shards)
        self.seed = seed
        self._epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        paths = list(self.paths)
        if self.shuffle:
            rng.shuffle(paths)
        for gs in range(0, len(paths), self.buffer_shards):
            group = paths[gs:gs + self.buffer_shards]
            data = {k: [] for k in _KEYS}
            for p in group:
                with np.load(p) as z:
                    for k in _KEYS:
                        data[k].append(z[k])
            data = {k: np.concatenate(v, axis=0) for k, v in data.items()}
            order = np.arange(data["board"].shape[0])
            if self.shuffle:
                rng.shuffle(order)
            for i in order:
                yield _to_tensors(data, int(i))
