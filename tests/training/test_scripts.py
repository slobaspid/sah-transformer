import sys
import importlib.util
import numpy as np
from sahformer.records import PositionRecord
from sahformer.encoding import TEMPORAL_DIM
from sahformer.shards import records_to_arrays, save_shard

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _rec():
    return PositionRecord(
        board=np.zeros((8, 8, 12), np.int8), history=np.zeros((7, 8, 8, 12), np.int8),
        stm=0, elo_self=1500, elo_opp=1500, temporal=np.zeros(TEMPORAL_DIM, np.float32),
        move_from=3, move_to=19, promo=0, result=2, think_time=1.5)

def test_train_cli_runs(tmp_path, monkeypatch):
    save_shard(str(tmp_path / "s.npz"), records_to_arrays([_rec(), _rec()]))
    mod = _load("train_cli", "scripts/train.py")
    monkeypatch.setattr(sys, "argv", [
        "train.py", str(tmp_path / "s.npz"), "--mode", "baseline",
        "--max-steps", "3", "--batch-size", "2", "--out", str(tmp_path / "ck")])
    mod.main()
    assert (tmp_path / "ck" / "last.pt").exists()
