"""Watch the model play itself: it plays a full game, then replays in your browser
at its own human-like pace (fast on easy moves, slow on hard ones).

Usage (from the project root, with the venv python):
    ./.venv/Scripts/python.exe scripts/watch_selfplay.py --ckpt path/to/best.pt

Then it opens selfplay.html in your browser. Adjust with --speed, --temperature, --seed.
"""
import argparse
import json
import os
import sys
import webbrowser

# make the project importable when run directly (python scripts/watch_selfplay.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sahformer.model.config import ModelConfig
from sahformer.training.build import build_model
from sahformer.training.loop import load_checkpoint
from sahformer.play import selfplay_frames

_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>sahformer self-play</title>
<style>
 body{{background:#2b2b2b;color:#eee;font-family:system-ui,Arial,sans-serif;text-align:center;margin:0;padding:18px}}
 #board{{display:inline-block}}
 #cap{{min-height:1.4em;margin:12px auto;font-size:18px}}
 #sub{{color:#9ab;font-size:14px;margin-bottom:10px}}
 button{{background:#3a7;color:#fff;border:0;border-radius:6px;padding:8px 16px;font-size:15px;cursor:pointer;margin:4px}}
 #bar{{height:6px;background:#444;border-radius:3px;max-width:400px;margin:8px auto}}
 #fill{{height:6px;background:#3a7;border-radius:3px;width:0}}
</style></head><body>
 <h2>your bot plays itself</h2>
 <div id="board"></div>
 <div id="cap">starting position</div>
 <div id="sub"></div>
 <div id="bar"><div id="fill"></div></div>
 <button onclick="restart()">&#9654; restart</button>
 <button onclick="paused=!paused;if(!paused)step()">play / pause</button>
<script>
 const frames = {frames};
 const caps = {caps};
 const speed = {speed};
 let i = 0, paused = false, timer = null;
 const boardEl=document.getElementById('board'), capEl=document.getElementById('cap'),
       subEl=document.getElementById('sub'), fillEl=document.getElementById('fill');
 function show(){{
   boardEl.innerHTML = frames[i];
   const c = caps[i];
   capEl.textContent = c.mover ? (i+'. '+c.mover+' plays '+c.san+'  —  thought '+c.think.toFixed(2)+'s')
                               : 'starting position';
   subEl.textContent = c.mover ? ('White '+c.white.toFixed(1)+'s   |   Black '+c.black.toFixed(1)+'s') : '';
   fillEl.style.width = (100*i/(frames.length-1))+'%';
 }}
 function step(){{
   if(paused || i>=frames.length-1) return;
   i++; show();
   let d = Math.min(Math.max(caps[i].think/speed, 0.15), 3.0)*1000;
   timer = setTimeout(step, d);
 }}
 function restart(){{ clearTimeout(timer); i=0; paused=false; show(); timer=setTimeout(step,700); }}
 show(); timer=setTimeout(step,900);
</script></body></html>"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to best.pt")
    ap.add_argument("--plies", type=int, default=120)
    ap.add_argument("--speed", type=float, default=1.5, help="playback speed (1=real pace)")
    ap.add_argument("--elo", type=int, default=1500)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--think-temp", type=float, default=0.25,
                    help="timing randomness: 1=full human spread, lower=calmer, 0=typical only")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="selfplay.html")
    args = ap.parse_args()

    model = build_model("full", ModelConfig())
    load_checkpoint(args.ckpt, model)
    frames, caps = selfplay_frames(model, max_plies=args.plies, elo=args.elo,
                                   temperature=args.temperature, seed=args.seed,
                                   think_temp=args.think_temp)
    html = _HTML.format(frames=json.dumps(frames), caps=json.dumps(caps), speed=args.speed)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    path = os.path.abspath(args.out)
    print(f"wrote {path}  ({len(frames)} positions) — opening in your browser...")
    webbrowser.open("file://" + path)

if __name__ == "__main__":
    main()
