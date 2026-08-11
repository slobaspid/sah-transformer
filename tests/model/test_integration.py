import numpy as np
import torch
from torch.utils.data import DataLoader
from sahformer.records import PositionRecord
from sahformer.encoding import TEMPORAL_DIM
from sahformer.shards import records_to_arrays, save_shard
from sahformer.dataset import ShardDataset
from sahformer.model.config import ModelConfig
from sahformer.model.chessformer import Chessformer
from sahformer.model.heads import mdn_nll
import torch.nn.functional as F

def _rec(elo, mf, mt, promo, result, tt):
    return PositionRecord(
        board=np.zeros((8, 8, 12), np.int8),
        history=np.zeros((7, 8, 8, 12), np.int8),
        stm=0, elo_self=elo, elo_opp=elo,
        temporal=np.zeros(TEMPORAL_DIM, np.float32),
        move_from=mf, move_to=mt, promo=promo, result=result, think_time=tt,
    )

def test_end_to_end_batch_and_losses(tmp_path):
    recs = [_rec(1500, 3, 19, 0, 2, 1.5), _rec(1200, 8, 16, 0, 1, 0.05)]
    save_shard(str(tmp_path / "s.npz"), records_to_arrays(recs))
    ds = ShardDataset([str(tmp_path / "s.npz")])
    loader = DataLoader(ds, batch_size=2)
    batch = next(iter(loader))

    model = Chessformer(ModelConfig())
    out = model(batch)

    b = batch["move_from"].shape[0]
    move_target = batch["move_from"] * 64 + batch["move_to"]
    policy_loss = F.cross_entropy(out["move_logits"].reshape(b, -1), move_target)
    value_loss = F.cross_entropy(out["value_logits"], batch["result"])
    pi, mu, sigma_p = out["mdn"]
    time_loss = mdn_nll(pi, mu, sigma_p, batch["think_time"])

    total = policy_loss + 0.1 * value_loss + 0.2 * time_loss
    assert torch.isfinite(total)
    total.backward()
    assert any(p.grad is not None for p in model.parameters())
