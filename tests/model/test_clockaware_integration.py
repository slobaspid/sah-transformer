import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sahformer.records import PositionRecord
from sahformer.encoding import TEMPORAL_DIM, build_temporal
from sahformer.shards import records_to_arrays, save_shard
from sahformer.dataset import ShardDataset
from sahformer.model.config import ModelConfig
from sahformer.model.clockaware import ClockAwareChessformer
from sahformer.model.heads import mdn_nll

def _rec(temporal):
    return PositionRecord(
        board=np.zeros((8, 8, 12), np.int8),
        history=np.zeros((7, 8, 8, 12), np.int8),
        stm=0, elo_self=1500, elo_opp=1500,
        temporal=temporal.astype(np.float32),
        move_from=3, move_to=19, promo=0, result=2, think_time=1.5,
    )

def test_end_to_end_with_temporal(tmp_path):
    calm = build_temporal(my_clock=120.0, opp_clock=120.0, own_think_history=[3.0], ply=10)
    panic = build_temporal(my_clock=2.0, opp_clock=90.0, own_think_history=[0.05], ply=40)
    save_shard(str(tmp_path / "s.npz"), records_to_arrays([_rec(calm), _rec(panic)]))
    ds = ShardDataset([str(tmp_path / "s.npz")])
    batch = next(iter(DataLoader(ds, batch_size=2)))

    model = ClockAwareChessformer(ModelConfig())
    out = model(batch)

    b = batch["move_from"].shape[0]
    policy_loss = F.cross_entropy(out["move_logits"].reshape(b, -1),
                                  batch["move_from"] * 64 + batch["move_to"])
    value_loss = F.cross_entropy(out["value_logits"], batch["result"])
    pi, mu, sigma_p = out["mdn"]
    time_loss = mdn_nll(pi, mu, sigma_p, batch["think_time"])
    total = policy_loss + 0.1 * value_loss + 0.2 * time_loss
    assert torch.isfinite(total)
    total.backward()
    assert any(p.grad is not None for p in model.parameters())

def test_time_pressure_shifts_think_time_distribution(tmp_path):
    calm = build_temporal(my_clock=150.0, opp_clock=150.0, own_think_history=[4.0], ply=8)
    panic = build_temporal(my_clock=1.5, opp_clock=80.0, own_think_history=[0.05, 0.05], ply=50)
    save_shard(str(tmp_path / "s.npz"), records_to_arrays([_rec(calm), _rec(panic)]))
    ds = ShardDataset([str(tmp_path / "s.npz")])
    batch = next(iter(DataLoader(ds, batch_size=2)))
    model = ClockAwareChessformer(ModelConfig())
    model.eval()
    with torch.no_grad():
        out = model(batch)
    mu = out["mdn"][1]
    assert not torch.allclose(mu[0], mu[1], atol=1e-6)
