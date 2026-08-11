import glob
import numpy as np
from sahformer.records import PositionRecord
from sahformer.encoding import TEMPORAL_DIM
from sahformer.dataset_build import build_shards, records_from_games
from sahformer.dataset import ShardDataset

def _rec(elo=1500):
    return PositionRecord(
        board=np.zeros((8, 8, 12), np.int8), history=np.zeros((7, 8, 8, 12), np.int8),
        stm=0, elo_self=elo, elo_opp=elo, temporal=np.zeros(TEMPORAL_DIM, np.float32),
        move_from=3, move_to=19, promo=0, result=2, think_time=1.0)

def test_build_shards_makes_multiple_chunks(tmp_path):
    paths = build_shards((_rec() for _ in range(250)), str(tmp_path), chunk_positions=100)
    assert len(paths) == 3                       # 100 + 100 + 50
    ds = ShardDataset(sorted(glob.glob(str(tmp_path / "*.npz"))))
    assert len(ds) == 250
    assert ds[0]["board"].shape == (8, 8, 12)

def test_build_shards_respects_max_positions(tmp_path):
    build_shards((_rec() for _ in range(1000)), str(tmp_path),
                 chunk_positions=100, max_positions=250)
    ds = ShardDataset(sorted(glob.glob(str(tmp_path / "*.npz"))))
    assert len(ds) == 250

def test_records_from_games_filters(monkeypatch):
    import sahformer.dataset_build as db
    monkeypatch.setattr(db, "is_target_game", lambda g: g["ok"])
    monkeypatch.setattr(db, "game_to_records", lambda g: [_rec(), _rec()])
    games = [{"ok": True}, {"ok": False}, {"ok": True}]
    recs = list(records_from_games(games))
    assert len(recs) == 4                         # 2 kept games x 2 records each

def test_multi_shard_dataset_indexing(tmp_path):
    from sahformer.shards import records_to_arrays, save_shard
    a = _rec(); a.move_from = 1
    b = _rec(); b.move_from = 2
    save_shard(str(tmp_path / "s00.npz"), records_to_arrays([a, a]))
    save_shard(str(tmp_path / "s01.npz"), records_to_arrays([b, b]))
    ds = ShardDataset([str(tmp_path / "s00.npz"), str(tmp_path / "s01.npz")])
    assert len(ds) == 4
    assert int(ds[0]["move_from"]) == 1          # first shard
    assert int(ds[3]["move_from"]) == 2          # second shard
