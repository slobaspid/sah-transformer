from dataclasses import dataclass

@dataclass
class SearchConfig:
    c_puct: float = 1.5           # exploration constant in PUCT
    sims_per_second: float = 16.0  # predicted seconds -> simulation budget
    min_sims: int = 1             # instant moves fall back to (near) raw policy
    max_sims: int = 256           # human-realism cap: never calculate like an engine
    temperature: float = 0.0      # 0 = most-visited move; >0 = sample by visit counts
    elo: int = 1500               # skill to imitate during search

def time_to_sims(seconds: float, cfg: SearchConfig) -> int:
    """Turn a predicted human think-time (seconds) into a simulation budget."""
    n = round(float(seconds) * cfg.sims_per_second)
    return int(min(max(n, cfg.min_sims), cfg.max_sims))

import math
import numpy as np
import chess
import torch
import torch.nn.functional as F
from sahformer.encoding import encode_board, encode_move, build_temporal
from sahformer.records import _stack_history
from sahformer.model.heads import move_to_index

class _Node:
    __slots__ = ("board", "plane_hist", "prior", "children", "N", "W", "expanded")
    def __init__(self, board, plane_hist, prior=0.0):
        self.board = board
        self.plane_hist = plane_hist   # list of encoded boards before this position
        self.prior = prior
        self.children = {}             # chess.Move -> _Node
        self.N = 0                     # visit count
        self.W = 0.0                   # summed value, this node's mover POV
        self.expanded = False

    @property
    def Q(self):
        return self.W / self.N if self.N > 0 else 0.0

def _terminal_value(board):
    # from the side-to-move's POV: checkmated = loss (-1); any other game-over = draw (0)
    return -1.0 if board.is_checkmate() else 0.0

def _evaluate(model, node, cfg, my_clock, opp_clock, ply, device):
    """Run the model on node.board: set priors on its children, return value (mover POV)."""
    b = node.board
    cur = encode_board(b)
    hist = _stack_history(node.plane_hist, cur)
    temporal = build_temporal(my_clock, opp_clock, [], ply)
    batch = {
        "board": torch.from_numpy(cur).float().unsqueeze(0).to(device),
        "history": torch.from_numpy(hist).float().unsqueeze(0).to(device),
        "elo_self": torch.tensor([cfg.elo]).to(device),
        "elo_opp": torch.tensor([cfg.elo]).to(device),
        "temporal": torch.from_numpy(temporal).float().unsqueeze(0).to(device),
    }
    with torch.no_grad():
        out = model(batch)
    logits = out["move_logits"][0]
    legal = list(b.legal_moves)
    idxs = [move_to_index(*encode_move(b, m)) for m in legal]
    priors = F.softmax(logits[idxs], dim=-1).cpu().numpy()
    wdl = F.softmax(out["value_logits"][0], dim=-1).cpu().numpy()
    value = float(wdl[2] - wdl[0])                 # P(win) - P(loss), mover POV
    next_hist = (node.plane_hist + [cur])[-7:]
    for m, p in zip(legal, priors):
        child_board = b.copy(stack=False)
        child_board.push(m)
        node.children[m] = _Node(child_board, next_hist, prior=float(p))
    node.expanded = True
    return value

def _select_child(node, cfg):
    total_N = sum(c.N for c in node.children.values())
    sqrt_total = math.sqrt(total_N) if total_N > 0 else 1.0
    best_move, best_child, best_score = None, None, -1e18
    for m, c in node.children.items():
        u = cfg.c_puct * c.prior * sqrt_total / (1 + c.N)
        score = -c.Q + u                            # child Q is opponent POV -> negate for us
        if score > best_score:
            best_score, best_move, best_child = score, m, c
    return best_move, best_child

def _simulate(model, root, cfg, my_clock, opp_clock, ply, device):
    path = [root]
    node = root
    while node.expanded and node.children and not node.board.is_game_over():
        _, node = _select_child(node, cfg)
        path.append(node)
    if node.board.is_game_over():
        value = _terminal_value(node.board)
    else:
        value = _evaluate(model, node, cfg, my_clock, opp_clock, ply, device)
    for n in reversed(path):                        # back up, flipping perspective each ply
        n.N += 1
        n.W += value
        value = -value

def mcts_move(model, board, plane_hist, sims, cfg, my_clock=180.0, opp_clock=180.0,
              ply=0, device="cpu", seed=None):
    """Run `sims` PUCT simulations from `board`; return (chosen_move, root_node)."""
    rng = np.random.default_rng(seed)
    root = _Node(board.copy(stack=False), list(plane_hist))
    if board.is_game_over():
        return None, root
    _evaluate(model, root, cfg, my_clock, opp_clock, ply, device)   # expand root
    for _ in range(sims):
        _simulate(model, root, cfg, my_clock, opp_clock, ply, device)
    moves = list(root.children.keys())
    visits = np.array([root.children[m].N for m in moves], dtype=np.float64)
    if visits.sum() == 0:
        priors = np.array([root.children[m].prior for m in moves])
        return moves[int(priors.argmax())], root
    if cfg.temperature <= 1e-6:
        return moves[int(visits.argmax())], root
    p = visits ** (1.0 / cfg.temperature)
    p = p / p.sum()
    return moves[int(rng.choice(len(moves), p=p))], root
