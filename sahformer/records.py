from dataclasses import dataclass
import numpy as np
import chess
from sahformer.encoding import encode_board, encode_move, build_temporal, TEMPORAL_DIM

BASE_SECONDS = 180.0  # 3+0

@dataclass
class PositionRecord:
    board: np.ndarray        # int8[8,8,12]
    history: np.ndarray      # int8[7,8,8,12]
    stm: int
    elo_self: int
    elo_opp: int
    temporal: np.ndarray     # float32[TEMPORAL_DIM]
    move_from: int
    move_to: int
    promo: int
    result: int              # stm-relative: 0 loss, 1 draw, 2 win
    think_time: float

_RESULT_WHITE = {"1-0": 2, "0-1": 0, "1/2-1/2": 1}

def _result_for_stm(result_str: str, white_to_move: bool) -> int:
    w = _RESULT_WHITE.get(result_str, 1)
    if white_to_move:
        return w
    return {0: 2, 1: 1, 2: 0}[w]  # mirror for black

def game_to_records(game):
    """Yield a PositionRecord per ply that has a clock annotation."""
    result_str = game.headers.get("Result", "1/2-1/2")
    white_elo = int(game.headers.get("WhiteElo", 0) or 0)
    black_elo = int(game.headers.get("BlackElo", 0) or 0)

    board = game.board()
    prev_clock = {chess.WHITE: BASE_SECONDS, chess.BLACK: BASE_SECONDS}
    think_hist = {chess.WHITE: [], chess.BLACK: []}  # most-recent first
    plane_hist = []  # list of int8[8,8,12], newest last

    node = game
    ply = 0
    while node.variations:
        node = node.variation(0)
        move = node.move
        mover = board.turn                      # who is about to move
        clock_after = node.clock()              # seconds left AFTER this move
        if clock_after is None:
            board.push(move)
            continue
        think = max(prev_clock[mover] - clock_after, 0.0)

        my_clock = prev_clock[mover]
        opp_clock = prev_clock[not mover]
        temporal = build_temporal(
            my_clock=my_clock, opp_clock=opp_clock,
            own_think_history=think_hist[mover], ply=ply,
        )

        cur = encode_board(board)
        hist = _stack_history(plane_hist, cur)
        frm, to, promo = encode_move(board, move)

        yield PositionRecord(
            board=cur, history=hist, stm=0 if mover == chess.WHITE else 1,
            elo_self=white_elo if mover == chess.WHITE else black_elo,
            elo_opp=black_elo if mover == chess.WHITE else white_elo,
            temporal=temporal, move_from=frm, move_to=to, promo=promo,
            result=_result_for_stm(result_str, mover == chess.WHITE),
            think_time=think,
        )

        # advance bookkeeping
        prev_clock[mover] = clock_after
        think_hist[mover] = [think] + think_hist[mover]
        plane_hist.append(cur)
        board.push(move)
        ply += 1

def _stack_history(plane_hist, current):
    """Return int8[7,8,8,12]: the 7 plies before `current`, newest first,
    earliest repeated if fewer than 7 exist."""
    out = np.zeros((7, 8, 8, 12), dtype=np.int8)
    prev = list(reversed(plane_hist[-7:]))  # newest first
    for i in range(7):
        if i < len(prev):
            out[i] = prev[i]
        elif prev:
            out[i] = prev[-1]  # repeat earliest available
        else:
            out[i] = current   # very first ply: repeat current
    return out
