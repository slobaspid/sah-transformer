import io
import chess.pgn

TARGET_TC = "180+0"  # 3+0 blitz

def is_target_game(game) -> bool:
    if game is None:
        return False
    if game.headers.get("TimeControl") != TARGET_TC:
        return False
    if not game.headers.get("WhiteElo") or not game.headers.get("BlackElo"):
        return False
    # require at least one clock annotation in the mainline
    node = game
    while node.variations:
        node = node.variation(0)
        if node.clock() is not None:
            return True
    return False

def iter_games_from_text(text: str):
    stream = io.StringIO(text)
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            return
        yield game

def iter_games_from_zst(path: str):
    """Stream games from a Lichess .pgn.zst file without full decompression."""
    import zstandard as zstd
    with open(path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            while True:
                game = chess.pgn.read_game(text_stream)
                if game is None:
                    return
                yield game
