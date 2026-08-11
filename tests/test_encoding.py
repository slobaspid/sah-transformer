import chess
import numpy as np
from sahformer.encoding import encode_board, encode_move, PIECE_ORDER

def test_encode_board_startpos_white():
    board = chess.Board()  # white to move
    planes = encode_board(board)
    assert planes.shape == (8, 8, 12)
    assert planes.dtype == np.int8
    # 8 own pawns on rank 2 (stm=white, own pawns plane 0)
    assert planes[:, :, 0].sum() == 8
    # own king (plane 5) present exactly once
    assert planes[:, :, 5].sum() == 1
    # opponent king (plane 11) present exactly once
    assert planes[:, :, 11].sum() == 1

def test_encode_board_flips_for_black():
    board = chess.Board()
    board.push_san("e4")  # now black to move
    planes = encode_board(board)
    # after flip, side-to-move (black) own pawns still counted in plane 0
    assert planes[:, :, 0].sum() == 8
    # the just-moved white pawn is an opponent pawn (plane 6)
    assert planes[:, :, 6].sum() == 8

def test_encode_move_simple():
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    frm, to, promo = encode_move(board, move)
    assert 0 <= frm < 64 and 0 <= to < 64
    assert promo == 0

def test_encode_move_promotion():
    board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
    move = chess.Move.from_uci("a7a8q")
    frm, to, promo = encode_move(board, move)
    assert promo == 4  # queen
