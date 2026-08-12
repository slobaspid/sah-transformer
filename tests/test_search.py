import numpy as np
import torch
import chess
from sahformer.search import SearchConfig, time_to_sims, mcts_move
from sahformer.model.config import ModelConfig
from sahformer.training.build import build_model
from sahformer.model.heads import move_to_index
from sahformer.encoding import encode_move

def test_time_to_sims_scales_and_clamps():
    c = SearchConfig(sims_per_second=16.0, min_sims=1, max_sims=256)
    assert time_to_sims(0.0, c) == 1                 # clamped up to the floor
    assert time_to_sims(0.2, c) in (3, 4)            # ~0.2s obvious move -> a few sims
    assert time_to_sims(8.0, c) == 128               # hard position -> real calculation
    assert time_to_sims(10_000.0, c) == 256          # clamped to the human-realism cap
    # monotonic non-decreasing
    xs = [time_to_sims(s, c) for s in (0.1, 1, 3, 8, 20)]
    assert xs == sorted(xs)

def test_mcts_returns_legal_move_and_visits():
    model = build_model("full", ModelConfig()); model.eval()
    board = chess.Board()
    cfg = SearchConfig(elo=1500, temperature=0.0)
    move, root = mcts_move(model, board, [], sims=16, cfg=cfg, seed=0)
    assert move in board.legal_moves
    total_visits = sum(ch.N for ch in root.children.values())
    assert total_visits >= 16                         # every sim visited some child

def test_mcts_deterministic_with_temp0():
    model = build_model("full", ModelConfig()); model.eval()
    board = chess.Board()
    cfg = SearchConfig(temperature=0.0)
    m1, _ = mcts_move(model, board, [], sims=24, cfg=cfg, seed=0)
    m2, _ = mcts_move(model, board, [], sims=24, cfg=cfg, seed=0)
    assert m1 == m2

def test_mcts_prior_drives_visits():
    board = chess.Board()
    favored = chess.Move.from_uci("e2e4")
    favored_idx = move_to_index(*encode_move(board, favored))

    class _MockModel:
        def eval(self): return self
        def __call__(self, batch):
            logits = torch.full((1, 4352), -10.0)
            logits[0, favored_idx] = 10.0             # huge prior on e2e4 at the root
            return {"move_logits": logits,
                    "value_logits": torch.zeros(1, 3),  # neutral value
                    "mdn": None}

    cfg = SearchConfig(temperature=0.0)
    move, root = mcts_move(_MockModel(), board, [], sims=40, cfg=cfg, seed=0)
    assert move == favored                            # search concentrates on the high-prior move
    assert root.children[favored].N == max(ch.N for ch in root.children.values())

def test_mcts_works_midgame():
    model = build_model("full", ModelConfig()); model.eval()
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    move, _ = mcts_move(model, board, [], sims=8, cfg=SearchConfig(), seed=1)
    assert move in board.legal_moves
