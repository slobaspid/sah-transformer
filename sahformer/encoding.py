import chess
import numpy as np

# 0..11: own P N B R Q K, then opponent p n b r q k (board is flipped to stm)
PIECE_ORDER = [
    chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING
]
_PROMO_ID = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}

def _oriented_square(sq: int, white_to_move: bool) -> int:
    """Return square index oriented so the side to move is 'at the bottom'."""
    return sq if white_to_move else chess.square_mirror(sq)

def encode_board(board: chess.Board) -> np.ndarray:
    """8x8x12 int8 planes, oriented to the side to move. Own pieces planes 0-5,
    opponent pieces planes 6-11."""
    stm_white = board.turn == chess.WHITE
    planes = np.zeros((8, 8, 12), dtype=np.int8)
    for sq, piece in board.piece_map().items():
        osq = _oriented_square(sq, stm_white)
        rank, file = divmod(osq, 8)
        own = piece.color == board.turn
        base = 0 if own else 6
        plane = base + PIECE_ORDER.index(piece.piece_type)
        planes[rank, file, plane] = 1
    return planes

def encode_move(board: chess.Board, move: chess.Move):
    """Return (from_sq, to_sq, promo_id) oriented to side to move."""
    stm_white = board.turn == chess.WHITE
    frm = _oriented_square(move.from_square, stm_white)
    to = _oriented_square(move.to_square, stm_white)
    promo = _PROMO_ID[move.promotion]
    return frm, to, promo

TEMPORAL_DIM = 21
_LOG180 = float(np.log1p(180.0))
_LOG30 = float(np.log1p(30.0))

def build_temporal(my_clock: float, opp_clock: float,
                   own_think_history, ply: int) -> np.ndarray:
    """own_think_history: this player's think-times for PRIOR moves, most-recent first.
    Never includes the current move (no leakage)."""
    v = np.zeros(TEMPORAL_DIM, dtype=np.float32)
    v[0] = my_clock / 180.0
    v[1] = opp_clock / 180.0
    v[2] = (my_clock - opp_clock) / 180.0
    v[3] = np.log1p(max(my_clock, 0.0)) / _LOG180
    v[4] = np.log1p(max(opp_clock, 0.0)) / _LOG180
    last5 = list(own_think_history[:5])
    for i, t in enumerate(last5):
        v[5 + i] = np.log1p(max(t, 0.0)) / _LOG30
        v[10 + i] = 1.0 if t < 0.1 else 0.0
    v[15] = 1.0 if my_clock < 2.0 else 0.0
    v[16] = 1.0 if my_clock < 5.0 else 0.0
    v[17] = 1.0 if my_clock < 10.0 else 0.0
    v[18] = 1.0 if my_clock > 30.0 else 0.0
    v[19] = (float(np.mean(last5)) / 30.0) if last5 else 0.0
    v[20] = ply / 80.0
    return v
