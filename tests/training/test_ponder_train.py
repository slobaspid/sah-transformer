import numpy as np
from sahformer.records import PositionRecord
from sahformer.encoding import TEMPORAL_DIM
from sahformer.shards import records_to_arrays, save_shard
from sahformer.model.config import ModelConfig
from sahformer.model.ponder import PonderChessformer
from sahformer.training.build import build_model, MODES
from sahformer.training.loop import TrainConfig, train

def _rec(seed):
    board = np.zeros((8, 8, 12), np.int8); board[seed % 8, 0, 0] = 1
    return PositionRecord(board=board, history=np.zeros((7, 8, 8, 12), np.int8),
        stm=0, elo_self=1500, elo_opp=1500, temporal=np.zeros(TEMPORAL_DIM, np.float32),
        move_from=3, move_to=19, promo=0, result=2, think_time=1.5)

def test_ponder_in_modes_and_builds():
    assert "ponder" in MODES
    m = build_model("ponder", ModelConfig())
    assert isinstance(m, PonderChessformer)

def test_ponder_training_reduces_loss(tmp_path):
    save_shard(str(tmp_path / "s.npz"), records_to_arrays([_rec(0), _rec(1)]))
    cfg = TrainConfig(mode="ponder", max_steps=60, warmup_steps=5, batch_size=2,
                      lr=1e-3, out_dir=str(tmp_path / "ck"))
    res = train(cfg, [str(tmp_path / "s.npz")])
    assert res["history"][-1]["total"] < res["history"][0]["total"]
