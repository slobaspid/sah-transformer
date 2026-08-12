"""A UCI chess engine wrapper around the trained model, so it plays in any chess GUI
(En Croissant, Cutechess, Arena, ...).

Point the environment variable SAHFORMER_CKPT at your best.pt, then run this as the engine.
On Windows, use the sahformer_engine.bat launcher (it sets the path and the venv python).

Options exposed to the GUI:
  UCI_Elo      - skill level to imitate (600-2800), default 1500
  Temperature  - move variety: 0 = always top move, higher = looser, default 0.3
"""
import os
import sys

# make the project importable when the GUI launches this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import chess
import torch
import torch.nn.functional as F
from sahformer.model.config import ModelConfig
from sahformer.training.build import build_model
from sahformer.training.loop import load_model
from sahformer.encoding import encode_board, encode_move, build_temporal
from sahformer.records import _stack_history
from sahformer.model.heads import move_to_index
from sahformer.play import _sample_think_time

def _find_ckpt():
    # 1) explicit: --ckpt PATH, or any .pt argument
    for i, a in enumerate(sys.argv):
        if a == "--ckpt" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.lower().endswith(".pt"):
            return a
    # 2) environment variable
    env = os.environ.get("SAHFORMER_CKPT", "")
    if env:
        return env
    # 3) common default locations (so a GUI can launch it with no config)
    home = os.path.expanduser("~")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for c in (os.path.join(home, "Downloads", "model_full", "best.pt"),
              os.path.join(home, "Downloads", "best.pt"),
              os.path.join(root, "best.pt")):
        if os.path.exists(c):
            return c
    return ""

def main():
    ckpt = _find_ckpt()
    if ckpt and os.path.exists(ckpt):
        try:
            model, _ = load_model(ckpt, mode="full")   # auto-matches the trained size
        except Exception as e:                         # stale/incompatible checkpoint
            sys.stderr.write(f"WARNING: could not load {ckpt} ({e}); retrain needed. "
                             "Playing untrained.\n")
            sys.stderr.flush()
            model = build_model("full", ModelConfig())
    else:
        sys.stderr.write("WARNING: no checkpoint found — playing untrained!\n")
        sys.stderr.flush()
        model = build_model("full", ModelConfig())
    model.eval()
    def _initial(flag, env, default):
        for i, a in enumerate(sys.argv):
            if a == flag and i + 1 < len(sys.argv):
                try:
                    return float(sys.argv[i + 1])
                except ValueError:
                    return default
        try:
            return float(os.environ.get(env, default))
        except ValueError:
            return default

    rng = np.random.default_rng()
    opts = {"elo": 1500, "temperature": 0.3,
            "pace": _initial("--pace", "SAHFORMER_PACE", 0.0),
            "think_temp": _initial("--think-temp", "SAHFORMER_THINK_TEMP", 0.7)}
    moves = []
    start_fen = chess.STARTING_FEN

    def rebuild():
        b = chess.Board(start_fen)
        plane_hist = []
        for u in moves:
            plane_hist.append(encode_board(b))
            b.push_uci(u)
        return b, plane_hist

    def choose(b, plane_hist, wtime, btime):
        cur = encode_board(b)
        hist = _stack_history(plane_hist, cur)
        my = wtime if b.turn == chess.WHITE else btime
        opp = btime if b.turn == chess.WHITE else wtime
        my = my / 1000.0 if my is not None else 180.0
        opp = opp / 1000.0 if opp is not None else 180.0
        temporal = build_temporal(my, opp, [], len(moves))
        batch = {
            "board": torch.from_numpy(cur).float().unsqueeze(0),
            "history": torch.from_numpy(hist).float().unsqueeze(0),
            "elo_self": torch.tensor([opts["elo"]]),
            "elo_opp": torch.tensor([opts["elo"]]),
            "temporal": torch.from_numpy(temporal).float().unsqueeze(0),
        }
        with torch.no_grad():
            out = model(batch)
        logits = out["move_logits"][0]
        legal = list(b.legal_moves)
        idxs = [move_to_index(*encode_move(b, m)) for m in legal]
        scores = logits[idxs]
        t = opts["temperature"]
        if t <= 1e-6:
            move = legal[int(scores.argmax().item())]
        else:
            probs = F.softmax(scores / t, dim=-1).cpu().numpy()
            probs = probs / probs.sum()
            move = legal[int(rng.choice(len(legal), p=probs))]
        # optionally play at human tempo: pause by the predicted think-time
        if opts["pace"] > 0:
            think = _sample_think_time(out["mdn"], rng, think_temp=opts["think_temp"])
            # allow the occasional big think, but never blow more than ~40% of the clock on one move
            sleep = min(think / opts["pace"], 30.0, max(0.3, my * 0.4))
            time.sleep(sleep)
        return move

    def emit(s):
        sys.stdout.write(s + "\n")
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if line == "uci":
            emit("id name Sahformer")
            emit("id author sahformer")
            emit("option name UCI_Elo type spin default 1500 min 600 max 2800")
            emit("option name Temperature type string default 0.3")
            emit("option name Pace type string default 0")
            emit("option name ThinkTemp type string default 0.7")
            emit("uciok")
        elif line == "isready":
            emit("readyok")
        elif line.startswith("setoption"):
            parts = line.split()
            if "name" in parts and "value" in parts:
                name = parts[parts.index("name") + 1].lower()
                val = parts[parts.index("value") + 1]
                try:
                    if name in ("uci_elo", "elo"):
                        opts["elo"] = int(float(val))
                    elif name == "temperature":
                        opts["temperature"] = float(val)
                    elif name == "pace":
                        opts["pace"] = float(val)
                    elif name == "thinktemp":
                        opts["think_temp"] = float(val)
                except ValueError:
                    pass
        elif line == "ucinewgame":
            moves = []
            start_fen = chess.STARTING_FEN
        elif line.startswith("position"):
            toks = line.split()
            if "startpos" in toks:
                start_fen = chess.STARTING_FEN
                i = toks.index("startpos") + 1
            elif "fen" in toks:
                fi = toks.index("fen")
                start_fen = " ".join(toks[fi + 1:fi + 7])
                i = fi + 7
            else:
                i = len(toks)
            moves = toks[i + 1:] if (i < len(toks) and toks[i] == "moves") else []
        elif line.startswith("go"):
            toks = line.split()
            wtime = int(toks[toks.index("wtime") + 1]) if "wtime" in toks else None
            btime = int(toks[toks.index("btime") + 1]) if "btime" in toks else None
            b, plane_hist = rebuild()
            if b.is_game_over():
                emit("bestmove 0000")
            else:
                emit(f"bestmove {choose(b, plane_hist, wtime, btime).uci()}")
        elif line == "quit":
            break

if __name__ == "__main__":
    main()
