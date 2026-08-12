from sahformer.model.config import ModelConfig
from sahformer.training.build import build_model
from sahformer.play import selfplay_frames

def test_selfplay_frames_align_and_render():
    model = build_model("full", ModelConfig())
    frames, caps = selfplay_frames(model, max_plies=6, start_clock=100000.0, seed=0)
    assert len(frames) == len(caps)
    assert len(frames) >= 2                       # start + at least one move
    assert frames[0].lstrip().startswith("<")     # looks like SVG markup
    assert caps[0]["mover"] == ""                 # first frame is the start position
    assert all(c["think"] >= 0.0 for c in caps)
