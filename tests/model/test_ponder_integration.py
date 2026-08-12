import chess
from sahformer.model.config import ModelConfig
from sahformer.training.build import build_model
from sahformer.play import self_play
from sahformer.search import SearchConfig, mcts_move

def test_ponder_model_self_plays():
    model = build_model("ponder", ModelConfig())
    plies = list(self_play(model, max_plies=6, start_clock=100000.0, seed=0))
    assert len(plies) >= 1
    board = chess.Board()
    for rec in plies:
        assert rec["move"] in board.legal_moves
        board.push(rec["move"])

def test_ponder_model_with_search():
    model = build_model("ponder", ModelConfig())
    board = chess.Board()
    move, _ = mcts_move(model, board, [], sims=6, cfg=SearchConfig(), seed=0)
    assert move in board.legal_moves
