import io
import chess.pgn
import numpy as np
from sahformer.records import game_to_records, PositionRecord

# 3+0 game with %clk after every move. White wins.
PGN = """[Event "Rated Blitz game"]
[White "a"]
[Black "b"]
[Result "1-0"]
[WhiteElo "1500"]
[BlackElo "1480"]
[TimeControl "180+0"]

1. e4 { [%clk 0:02:58] } e5 { [%clk 0:02:57] } 2. Nf3 { [%clk 0:02:55] } Nc6 { [%clk 0:02:50] } 3. Bb5 { [%clk 0:02:52] } a6 { [%clk 0:02:40] } 1-0
"""

def _first_game():
    return chess.pgn.read_game(io.StringIO(PGN))

def test_yields_one_record_per_move():
    recs = list(game_to_records(_first_game()))
    assert len(recs) == 6  # 3 full moves = 6 plies

def test_first_move_think_time():
    recs = list(game_to_records(_first_game()))
    # white spent 180 - 178 = 2s on move 1 (clock shows 2:58 = 178s)
    assert abs(recs[0].think_time - 2.0) < 1e-6
    assert recs[0].elo_self == 1500
    assert recs[0].elo_opp == 1480

def test_result_is_stm_relative():
    recs = list(game_to_records(_first_game()))
    # white wins -> white-to-move records have result 2 (win)
    assert recs[0].result == 2   # white's 1st move
    assert recs[1].result == 0   # black's reply -> loss from black view

def test_temporal_has_no_future_leak():
    recs = list(game_to_records(_first_game()))
    # move 1 white: no prior own think times -> last5 slots zero
    assert recs[0].temporal[5] == 0.0
    # move 2 white (index 2): prior own think time = 2.0s present
    assert recs[2].temporal[5] > 0.0
