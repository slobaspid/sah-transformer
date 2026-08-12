import numpy as np
import torch
from sahformer.records import PositionRecord
from sahformer.encoding import TEMPORAL_DIM
from sahformer.shards import records_to_arrays, save_shard
from sahformer.model.config import ModelConfig
from sahformer.training.build import build_model
from sahformer.training.loop import TrainConfig, train, save_checkpoint, load_checkpoint

def _rec(seed, mf, mt):
    board = np.zeros((8, 8, 12), np.int8)
    board[seed % 8, 0, 0] = 1  # distinct inputs so the batch is cleanly fittable
    return PositionRecord(
        board=board, history=np.zeros((7, 8, 8, 12), np.int8),
        stm=0, elo_self=1500, elo_opp=1500, temporal=np.zeros(TEMPORAL_DIM, np.float32),
        move_from=mf, move_to=mt, promo=0, result=2, think_time=1.5)

def _shard(tmp_path):
    save_shard(str(tmp_path / "s.npz"), records_to_arrays([_rec(0, 3, 19), _rec(1, 8, 16)]))
    return [str(tmp_path / "s.npz")]

def test_overfit_decreases_loss(tmp_path):
    cfg = TrainConfig(mode="full", max_steps=80, warmup_steps=5, batch_size=2,
                      lr=1e-3, out_dir=str(tmp_path / "ck"))
    res = train(cfg, _shard(tmp_path))
    first = res["history"][0]["total"]
    last = res["history"][-1]["total"]
    assert last < first, (first, last)

def test_train_writes_checkpoints(tmp_path):
    cfg = TrainConfig(mode="baseline", max_steps=10, warmup_steps=2, batch_size=2,
                      ckpt_every=5, out_dir=str(tmp_path / "ck"))
    train(cfg, _shard(tmp_path))
    assert (tmp_path / "ck" / "last.pt").exists()
    assert (tmp_path / "ck" / "best.pt").exists()

def test_checkpoint_roundtrip(tmp_path):
    cfg = TrainConfig(mode="baseline", max_steps=6, warmup_steps=1, batch_size=2,
                      out_dir=str(tmp_path / "ck"))
    res = train(cfg, _shard(tmp_path))
    fresh = build_model("baseline", ModelConfig())
    load_checkpoint(str(tmp_path / "ck" / "last.pt"), fresh)
    for (_, p1), (_, p2) in zip(res["model"].named_parameters(), fresh.named_parameters()):
        assert torch.allclose(p1.detach().cpu(), p2.detach().cpu())

def test_checkpoint_saves_optimizer_state(tmp_path):
    cfg = TrainConfig(mode="baseline", max_steps=6, warmup_steps=1, batch_size=2,
                      out_dir=str(tmp_path / "ck"))
    train(cfg, _shard(tmp_path))
    ck = torch.load(str(tmp_path / "ck" / "last.pt"), map_location="cpu", weights_only=False)
    assert ck.get("opt_state") is not None            # optimizer state is saved for resuming

def test_resume_continues_from_saved_step(tmp_path):
    shards = _shard(tmp_path)
    train(TrainConfig(mode="baseline", max_steps=5, warmup_steps=1, batch_size=2,
                      out_dir=str(tmp_path / "ck")), shards)
    # resume toward a higher TOTAL step count, writing to a fresh dir
    res = train(TrainConfig(mode="baseline", max_steps=9, warmup_steps=1, batch_size=2,
                            out_dir=str(tmp_path / "ck2"),
                            resume=str(tmp_path / "ck" / "last.pt")), shards)
    assert res["history"][0]["step"] == 5             # continued, not restarted from 0
    assert res["history"][-1]["step"] == 8
