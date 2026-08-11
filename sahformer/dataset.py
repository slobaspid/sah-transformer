import numpy as np
import torch
from torch.utils.data import Dataset

_KEYS = ["board", "history", "stm", "elo_self", "elo_opp", "temporal",
         "move_from", "move_to", "promo", "result", "think_time"]

class ShardDataset(Dataset):
    """Loads one or more .npz shards fully into memory (Colab-sized shards)."""
    def __init__(self, shard_paths):
        self.data = {k: [] for k in _KEYS}
        for path in shard_paths:
            with np.load(path) as z:
                for k in _KEYS:
                    self.data[k].append(z[k])
        self.data = {k: np.concatenate(v, axis=0) for k, v in self.data.items()}
        self.n = self.data["board"].shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        out = {}
        for k in _KEYS:
            val = self.data[k][i]
            if k in ("board", "history", "temporal"):
                out[k] = torch.from_numpy(np.ascontiguousarray(val)).float()
            elif k == "think_time":
                out[k] = torch.tensor(float(val), dtype=torch.float32)
            else:
                out[k] = torch.tensor(int(val), dtype=torch.long)
        return out
