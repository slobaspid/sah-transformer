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
