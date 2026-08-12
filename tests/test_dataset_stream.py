import glob
import numpy as np
from sahformer.records import PositionRecord
from sahformer.encoding import TEMPORAL_DIM
from sahformer.dataset_build import build_shards
from sahformer.dataset import StreamingShardDataset

def _rec(mf):
    return PositionRecord(
        board=np.zeros((8, 8, 12), np.int8), history=np.zeros((7, 8, 8, 12), np.int8),
        stm=0, elo_self=1500, elo_opp=1500, temporal=np.zeros(TEMPORAL_DIM, np.float32),
        move_from=mf, move_to=19, promo=0, result=2, think_time=1.0)

def test_streaming_yields_every_sample(tmp_path):
    build_shards((_rec(i % 64) for i in range(250)), str(tmp_path), chunk_positions=60)
    paths = sorted(glob.glob(str(tmp_path / "*.npz")))
    ds = StreamingShardDataset(paths, shuffle=True, buffer_shards=2, seed=0)
    items = list(ds)
    assert len(items) == 250
    assert items[0]["board"].shape == (8, 8, 12)
    # re-iterating starts a fresh epoch and still yields everything
    assert len(list(ds)) == 250

def test_streaming_covers_all_move_values(tmp_path):
    # distinct move_from per shard so we can confirm streaming visits every shard
    build_shards((_rec(i) for i in range(120)), str(tmp_path), chunk_positions=40)
    paths = sorted(glob.glob(str(tmp_path / "*.npz")))
    seen = {int(x["move_from"]) for x in StreamingShardDataset(paths, seed=1)}
    assert seen == set(range(120))

def test_streamed_training_runs(tmp_path):
    from sahformer.training.loop import TrainConfig, train
    build_shards((_rec(i % 64) for i in range(200)), str(tmp_path), chunk_positions=60)
    paths = sorted(glob.glob(str(tmp_path / "*.npz")))
    cfg = TrainConfig(mode="full", max_steps=8, warmup_steps=2, batch_size=16,
                      stream=True, out_dir=str(tmp_path / "ck"))
    res = train(cfg, paths)
    assert len(res["history"]) == 8
