import io
import zstandard as zstd
from sahformer.download import _iter_games_from_binary, is_target_game

_PGN = (
    '[Event "A"]\n[TimeControl "180+0"]\n[WhiteElo "1500"]\n[BlackElo "1500"]\n\n'
    '1. e4 { [%clk 0:03:00] } e5 { [%clk 0:03:00] } *\n\n'
    '[Event "B"]\n[TimeControl "600+0"]\n[WhiteElo "1500"]\n[BlackElo "1500"]\n\n'
    '1. d4 d5 *\n\n'
)

def test_iter_games_from_binary_parses_zst_stream():
    comp = zstd.ZstdCompressor().compress(_PGN.encode("utf-8"))
    games = list(_iter_games_from_binary(io.BytesIO(comp)))
    assert len(games) == 2
    # first is a 3+0 game with clocks; second is 10+0 (filtered out)
    assert is_target_game(games[0])
    assert not is_target_game(games[1])
