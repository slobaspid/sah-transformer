import math
import numpy as np
import chess
import torch
import torch.nn.functional as F
from sahformer.encoding import encode_board, encode_move, build_temporal
from sahformer.records import _stack_history
from sahformer.model.heads import move_to_index

def _softmax_np(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

def _sample_think_time(mdn, rng):
    """Sample a think-time (seconds) from the MDN mixture-of-log-normals output."""
    pi_logits = mdn[0][0].detach().cpu().numpy()
    mu = mdn[1][0].detach().cpu().numpy()
    sigma_p = mdn[2][0].detach().cpu().numpy()
    pi = _softmax_np(pi_logits)
    k = int(rng.choice(len(pi), p=pi))
    sigma = float(np.logaddexp(0.0, sigma_p[k])) + 1e-3   # stable softplus
    z = float(rng.standard_normal())
    t = math.exp(mu[k] + sigma * z)
    return float(min(max(t, 0.0), 180.0))

@torch.no_grad()
def self_play(model, max_plies=200, elo=1500, temperature=1.0,
              start_clock=180.0, device="cpu", seed=0):
    """Model plays both sides. Yields one dict per ply (the move it's about to play,
    its sampled think-time, and both clocks). Ends on game over, ply cap, or a flag."""
    rng = np.random.default_rng(seed)
    model.eval()
    board = chess.Board()
    plane_hist = []
    think_hist = {chess.WHITE: [], chess.BLACK: []}
    clock = {chess.WHITE: float(start_clock), chess.BLACK: float(start_clock)}
    ply = 0
    while not board.is_game_over() and ply < max_plies:
        mover = board.turn
        cur = encode_board(board)
        hist = _stack_history(plane_hist, cur)
        temporal = build_temporal(clock[mover], clock[not mover], think_hist[mover], ply)
        batch = {
            "board": torch.from_numpy(cur).float().unsqueeze(0).to(device),
            "history": torch.from_numpy(hist).float().unsqueeze(0).to(device),
            "elo_self": torch.tensor([elo]).to(device),
            "elo_opp": torch.tensor([elo]).to(device),
            "temporal": torch.from_numpy(temporal).float().unsqueeze(0).to(device),
        }
        out = model(batch)
        logits = out["move_logits"][0]
        legal = list(board.legal_moves)
        idxs = [move_to_index(*encode_move(board, m)) for m in legal]
        probs = F.softmax(logits[idxs] / temperature, dim=-1).detach().cpu().numpy()
        probs = probs / probs.sum()
        move = legal[int(rng.choice(len(legal), p=probs))]
        think = _sample_think_time(out["mdn"], rng)

        clock[mover] -= think
        flagged = clock[mover] <= 0.0
        yield {
            "ply": ply, "mover": "white" if mover == chess.WHITE else "black",
            "move": move, "san": board.san(move), "think": think,
            "white_clock": clock[chess.WHITE], "black_clock": clock[chess.BLACK],
            "fen": board.fen(), "flagged": flagged,
        }
        if flagged:
            break
        think_hist[mover] = [think] + think_hist[mover]
        plane_hist.append(cur)
        board.push(move)
        ply += 1
