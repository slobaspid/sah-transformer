import bisect
import numpy as np
import torch
from torch.utils.data import Dataset

_KEYS = ["board", "history", "stm", "elo_self", "elo_opp", "temporal",
         "move_from", "move_to", "promo", "result", "think_time"]

class ShardDataset(Dataset):
    """Loads one or more .npz shards into memory and indexes across them without
    concatenating, so peak RAM is ~1x the data (not 2x)."""
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
        d = self.shards[s]
        row = i - base
        out = {}
        for k in _KEYS:
            val = d[k][row]
            if k in ("board", "history", "temporal"):
                out[k] = torch.from_numpy(np.ascontiguousarray(val)).float()
            elif k == "think_time":
                out[k] = torch.tensor(float(val), dtype=torch.float32)
            else:
                out[k] = torch.tensor(int(val), dtype=torch.long)
        return out
