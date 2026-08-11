import math
import torch
from sahformer.model.config import ModelConfig
from sahformer.model.heads import (move_to_index, PolicyHead, ValueHead,
                                   ThinkTimeMDNHead, mdn_nll)

def test_move_index_scheme():
    assert move_to_index(3, 19, 0) == 3 * 64 + 19          # non-promo
    assert move_to_index(0, 0, 4) == 4096 + 0 * 4 + 3      # queen promo to sq 0
    assert move_to_index(10, 63, 1) == 4096 + 63 * 4 + 0   # knight promo to sq 63

def test_policy_head_4352():
    c = ModelConfig()
    head = PolicyHead(c)
    logits = head(torch.randn(2, 64, c.dim_vit))
    assert logits.shape == (2, 4352)

def test_value_head():
    c = ModelConfig()
    head = ValueHead(c)
    assert head(torch.randn(2, 64, c.dim_vit)).shape == (2, 3)

def test_mdn_head_and_loss():
    c = ModelConfig()
    head = ThinkTimeMDNHead(c)
    pooled = torch.randn(4, c.dim_vit, requires_grad=True)
    pi, mu, sig = head(pooled)
    assert pi.shape == (4, c.mdn_components)
    loss = mdn_nll(pi, mu, sig, torch.tensor([0.05, 2.0, 12.0, 0.5]))
    assert torch.isfinite(loss)
    loss.backward()
    assert pooled.grad is not None

def test_mdn_single_component_matches_normal():
    pi = torch.zeros(1, 1); mu = torch.zeros(1, 1)
    sigp = torch.log(torch.expm1(torch.tensor([[1.0 - 1e-3]])))
    nll = mdn_nll(pi, mu, sigp, torch.tensor([1.0]))
    assert abs(nll.item() - 0.5 * math.log(2 * math.pi)) < 1e-3
