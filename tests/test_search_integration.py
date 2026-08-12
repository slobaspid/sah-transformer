import chess
from sahformer.model.config import ModelConfig
from sahformer.training.build import build_model
from sahformer.play import self_play
from sahformer.search import SearchConfig

def test_self_play_with_search_is_legal():
    model = build_model("full", ModelConfig())
    board = chess.Board()
    scfg = SearchConfig(max_sims=8, elo=1500, temperature=0.0)
    plies = list(self_play(model, max_plies=8, start_clock=100000.0, seed=0,
                           search=True, search_cfg=scfg))
    assert len(plies) >= 1
    for rec in plies:
        assert rec["move"] in board.legal_moves
        board.push(rec["move"])
