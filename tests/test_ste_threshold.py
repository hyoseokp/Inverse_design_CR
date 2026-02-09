from __future__ import annotations

import torch

from crinv.ops import hard_threshold, ste_threshold


def test_ste_forward_matches_hard():
    torch.manual_seed(0)
    u = torch.rand(4, 8, 8)
    x_hard = hard_threshold(u, tau=0.5)
    x_hard2, x_ste = ste_threshold(u, tau=0.5)
    assert torch.allclose(x_hard2, x_hard, atol=0.0, rtol=0.0)
    assert torch.allclose(x_ste, x_hard, atol=0.0, rtol=0.0)


def test_ste_backward_grad_is_one_everywhere():
    torch.manual_seed(0)
    u = torch.rand(2, 5, 5, requires_grad=True)
    _, x_ste = ste_threshold(u, tau=0.5)
    x_ste.sum().backward()
    assert u.grad is not None
    assert torch.allclose(u.grad, torch.ones_like(u), atol=0.0, rtol=0.0)

