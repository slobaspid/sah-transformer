import pytest
import torch
from sahformer.model.config import ModelConfig
from sahformer.training.build import build_model, MODES

def _batch(b=2, fill=0.5):
    return {"board": torch.zeros(b, 8, 8, 12), "history": torch.zeros(b, 7, 8, 8, 12),
            "elo_self": torch.tensor([1500] * b), "elo_opp": torch.tensor([1600] * b),
            "temporal": torch.full((b, 21), fill)}

def test_modes_list():
    assert MODES == ("baseline", "film_only", "gab_only", "full")

def test_all_modes_build_forward_backward():
    c = ModelConfig()
    for m in MODES:
        model = build_model(m, c)
        out = model(_batch(2))
        assert out["move_logits"].shape == (2, 4352)
        assert out["value_logits"].shape == (2, 3)
        assert out["mdn"][0].shape == (2, c.mdn_components)
        out["move_logits"].sum().backward()

def test_baseline_has_no_temporal_params():
    names = [n for n, _ in build_model("baseline", ModelConfig()).named_parameters()]
    assert not any(("temporal_enc" in n) or ("film_gen" in n) for n in names)

def test_gab_only_has_no_film_params():
    names = [n for n, _ in build_model("gab_only", ModelConfig()).named_parameters()]
    assert not any("film_gen" in n for n in names)

def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        build_model("nope", ModelConfig())
