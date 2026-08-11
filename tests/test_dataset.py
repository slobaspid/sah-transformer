import numpy as np
from sahformer.shards import records_to_arrays, save_shard
from sahformer.records import PositionRecord
from sahformer.encoding import TEMPORAL_DIM
from sahformer.dataset import ShardDataset

def _rec():
    return PositionRecord(
        board=np.zeros((8, 8, 12), np.int8),
        history=np.zeros((7, 8, 8, 12), np.int8),
        stm=0, elo_self=1500, elo_opp=1500,
        temporal=np.arange(TEMPORAL_DIM, dtype=np.float32),
        move_from=3, move_to=19, promo=0, result=2, think_time=2.5,
    )

def test_dataset_roundtrip(tmp_path):
    arr = records_to_arrays([_rec(), _rec()])
    p = tmp_path / "shard0.npz"
    save_shard(str(p), arr)
    ds = ShardDataset([str(p)])
    assert len(ds) == 2
    sample = ds[0]
    assert sample["board"].shape == (8, 8, 12)
    assert sample["temporal"].shape == (TEMPORAL_DIM,)
    assert float(sample["think_time"]) == 2.5
    assert int(sample["move_from"]) == 3
