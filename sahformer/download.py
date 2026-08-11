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

def _iter_games_from_binary(reader):
    """Decompress a zstd binary stream (file-like with .read) and yield games."""
    import zstandard as zstd
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(reader) as decompressed:
        text_stream = io.TextIOWrapper(decompressed, encoding="utf-8")
        while True:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                return
            yield game

def iter_games_from_zst(path: str):
    """Stream games from a local Lichess .pgn.zst file."""
    with open(path, "rb") as fh:
        yield from _iter_games_from_binary(fh)

def stream_games_from_url(url: str):
    """Stream games directly from a Lichess .pgn.zst URL, decompressing on the fly.
    The consumer should early-stop; only the first chunk of the archive is transferred."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "sahformer/0.1"})
    with urllib.request.urlopen(req) as resp:
        yield from _iter_games_from_binary(resp)
