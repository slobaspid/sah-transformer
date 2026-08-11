import math
import torch
from sahformer.model.config import ModelConfig
from sahformer.model.heads import PolicyHead, ValueHead, ThinkTimeMDNHead, mdn_nll

def test_policy_head_shapes():
    c = ModelConfig()
    head = PolicyHead(c)
    board = torch.randn(2, 64, c.d_model)
    move_logits, promo_logits = head(board)
    assert move_logits.shape == (2, 64, 64)
    assert promo_logits.shape == (2, 64, 4)

def test_value_head_shape():
    c = ModelConfig()
    head = ValueHead(c)
    pooled = torch.randn(2, c.d_model)
    assert head(pooled).shape == (2, 3)

def test_mdn_head_shapes():
    c = ModelConfig()
    head = ThinkTimeMDNHead(c)
    pooled = torch.randn(2, c.d_model)
    pi, mu, sigma_p = head(pooled)
    assert pi.shape == (2, c.mdn_components)
    assert mu.shape == (2, c.mdn_components)
    assert sigma_p.shape == (2, c.mdn_components)

def test_mdn_nll_single_component_matches_normal():
    pi_logits = torch.zeros(1, 1)
    mu = torch.zeros(1, 1)
    target_sigma = 1.0 - 1e-3
    sigma_param = torch.log(torch.expm1(torch.tensor([[target_sigma]])))  # inverse softplus
    target_time = torch.tensor([1.0])
    nll = mdn_nll(pi_logits, mu, sigma_param, target_time)
    assert abs(nll.item() - 0.5 * math.log(2 * math.pi)) < 1e-3

def test_mdn_nll_finite_and_differentiable():
    c = ModelConfig()
    head = ThinkTimeMDNHead(c)
    pooled = torch.randn(4, c.d_model, requires_grad=True)
    pi, mu, sigma_p = head(pooled)
    target = torch.tensor([0.05, 2.0, 12.0, 0.5])
    loss = mdn_nll(pi, mu, sigma_p, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert pooled.grad is not None and torch.isfinite(pooled.grad).all()
