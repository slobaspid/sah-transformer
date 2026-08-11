from sahformer.model.config import ModelConfig
from sahformer.model.chessformer import FaithfulChessformer
from sahformer.model.clockaware import ClockAwareChessformer

MODES = ("baseline", "film_only", "gab_only", "full")

def build_model(mode: str, cfg: ModelConfig):
    """Return the model for an ablation mode (see the design's ablation table)."""
    if mode == "baseline":
        return FaithfulChessformer(cfg)
    if mode == "film_only":
        return ClockAwareChessformer(cfg, use_film=True, use_time_gab=False)
    if mode == "gab_only":
        return ClockAwareChessformer(cfg, use_film=False, use_time_gab=True)
    if mode == "full":
        return ClockAwareChessformer(cfg, use_film=True, use_time_gab=True)
    raise ValueError(f"unknown mode: {mode!r} (expected one of {MODES})")
