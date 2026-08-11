import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sahformer.records import PositionRecord
from sahformer.encoding import TEMPORAL_DIM
from sahformer.shards import records_to_arrays, save_shard
from sahformer.dataset import ShardDataset
from sahformer.model.config import ModelConfig
from sahformer.model.chessformer import FaithfulChessformer
from sahformer.model.heads import move_to_index, mdn_nll

def _rec(mf, mt, promo, result, tt):
    return PositionRecord(
        board=np.zeros((8, 8, 12), np.int8), history=np.zeros((7, 8, 8, 12), np.int8),
        stm=0, elo_self=1500, elo_opp=1500, temporal=np.zeros(TEMPORAL_DIM, np.float32),
        move_from=mf, move_to=mt, promo=promo, result=result, think_time=tt)

def test_end_to_end(tmp_path):
    recs = [_rec(3, 19, 0, 2, 1.5), _rec(8, 16, 0, 1, 0.05)]
    save_shard(str(tmp_path / "s.npz"), records_to_arrays(recs))
    batch = next(iter(DataLoader(ShardDataset([str(tmp_path / "s.npz")]), batch_size=2)))
    model = FaithfulChessformer(ModelConfig())
    out = model(batch)
    b = batch["move_from"].shape[0]
    target = torch.tensor([move_to_index(int(batch["move_from"][i]), int(batch["move_to"][i]),
                                         int(batch["promo"][i])) for i in range(b)])
    policy_loss = F.cross_entropy(out["move_logits"], target)
    value_loss = F.cross_entropy(out["value_logits"], batch["result"])
    pi, mu, sig = out["mdn"]
    time_loss = mdn_nll(pi, mu, sig, batch["think_time"])
    total = policy_loss + 0.1 * value_loss + 0.2 * time_loss
    assert torch.isfinite(total)
    total.backward()
    assert any(p.grad is not None for p in model.parameters())
