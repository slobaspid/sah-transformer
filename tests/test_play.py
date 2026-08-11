import chess
from sahformer.model.config import ModelConfig
from sahformer.training.build import build_model
from sahformer.play import self_play

def test_self_play_yields_only_legal_moves():
    model = build_model("full", ModelConfig())
    # high start_clock so an untrained model's random think-times don't flag early
    plies = list(self_play(model, max_plies=12, start_clock=100000.0, seed=0))
    assert len(plies) >= 1
    board = chess.Board()
    for rec in plies:
        assert rec["move"] in board.legal_moves        # every move is legal
        assert rec["think"] >= 0.0                      # think-times are non-negative
        board.push(rec["move"])

def test_self_play_respects_ply_cap():
    model = build_model("baseline", ModelConfig()) if False else build_model("full", ModelConfig())
    plies = list(self_play(model, max_plies=5, start_clock=100000.0, seed=1))
    assert len(plies) <= 5
