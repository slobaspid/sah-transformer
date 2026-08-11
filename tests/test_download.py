import io
import chess.pgn
from sahformer.download import is_target_game, iter_games_from_text

MIXED_PGN = """[Event "Rated Blitz game"]
[TimeControl "180+0"]
[WhiteElo "1500"]
[BlackElo "1500"]
[Result "1-0"]

1. e4 { [%clk 0:02:58] } 1-0

[Event "Rated Bullet game"]
[TimeControl "60+0"]
[WhiteElo "1500"]
[BlackElo "1500"]
[Result "0-1"]

1. e4 e5 0-1

[Event "Rated Blitz game"]
[TimeControl "180+2"]
[WhiteElo "1500"]
[BlackElo "1500"]
[Result "1-0"]

1. e4 { [%clk 0:02:58] } 1-0
"""

def test_only_3plus0_with_clocks_pass():
    games = list(iter_games_from_text(MIXED_PGN))
    keep = [g for g in games if is_target_game(g)]
    assert len(keep) == 1
    assert keep[0].headers["TimeControl"] == "180+0"

def test_rejects_missing_clock():
    pgn = ('[TimeControl "180+0"]\n[WhiteElo "1"]\n[BlackElo "1"]\n'
           '[Result "1-0"]\n\n1. e4 1-0\n')
    g = chess.pgn.read_game(io.StringIO(pgn))
    assert is_target_game(g) is False
