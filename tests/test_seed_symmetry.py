from __future__ import annotations

import torch

from crinv.seed import seed_from_araw


def test_seed_is_symmetric():
    torch.manual_seed(0)
    a_raw = torch.randn(16, 16, dtype=torch.float32)
    s = seed_from_araw(a_raw)
    assert torch.allclose(s, s.T, atol=1e-6, rtol=0.0)


def test_seed_backprop_grad_exists_and_finite():
    torch.manual_seed(0)
    a_raw = torch.randn(4, 16, 16, dtype=torch.float32, requires_grad=True)
    s = seed_from_araw(a_raw)
    loss = (s ** 2).mean()
    loss.backward()
    assert a_raw.grad is not None
    assert a_raw.grad.shape == a_raw.shape
    assert torch.isfinite(a_raw.grad).all()

