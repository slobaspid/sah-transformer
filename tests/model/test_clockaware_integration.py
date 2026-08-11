import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sahformer.records import PositionRecord
from sahformer.encoding import build_temporal
from sahformer.shards import records_to_arrays, save_shard
from sahformer.dataset import ShardDataset
from sahformer.model.config import ModelConfig
from sahformer.model.clockaware import ClockAwareChessformer
from sahformer.model.heads import move_to_index, mdn_nll

def _rec(temporal):
    return PositionRecord(
        board=np.zeros((8, 8, 12), np.int8), history=np.zeros((7, 8, 8, 12), np.int8),
        stm=0, elo_self=1500, elo_opp=1500, temporal=temporal.astype(np.float32),
        move_from=3, move_to=19, promo=0, result=2, think_time=1.5)

def test_end_to_end_and_time_sensitivity(tmp_path):
    calm = build_temporal(150.0, 150.0, [4.0], 8)
    panic = build_temporal(1.5, 80.0, [0.05, 0.05], 50)
    save_shard(str(tmp_path / "s.npz"), records_to_arrays([_rec(calm), _rec(panic)]))
    batch = next(iter(DataLoader(ShardDataset([str(tmp_path / "s.npz")]), batch_size=2)))
    model = ClockAwareChessformer(ModelConfig())
    out = model(batch)
    b = batch["move_from"].shape[0]
    target = torch.tensor([move_to_index(int(batch["move_from"][i]), int(batch["move_to"][i]),
                                         int(batch["promo"][i])) for i in range(b)])
    total = (F.cross_entropy(out["move_logits"], target)
             + 0.1 * F.cross_entropy(out["value_logits"], batch["result"])
             + 0.2 * mdn_nll(*out["mdn"], batch["think_time"]))
    assert torch.isfinite(total)
    total.backward()
    # behavioral: the two rows (calm vs panic clock) get different predicted think-time means
    model.eval()
    with torch.no_grad():
        mu = model(batch)["mdn"][1]
    assert not torch.allclose(mu[0], mu[1], atol=1e-6)
