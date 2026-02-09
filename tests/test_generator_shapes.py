from __future__ import annotations

import torch

from crinv.generator import generate_u_and_binary
from crinv.seed import seed_from_araw


def test_generator_shapes_and_binary():
    torch.manual_seed(0)
    a_raw = torch.randn(3, 16, 16)
    s = seed_from_araw(a_raw)
    x_hard, x_ste, u = generate_u_and_binary(s, sigma=1.0, tau=0.5, struct_size=128, use_ste=True)
    assert x_hard.shape == (3, 128, 128)
    assert x_ste.shape == (3, 128, 128)
    assert u.shape == (3, 128, 128)
    # binary forward path
    assert torch.all((x_hard == 0) | (x_hard == 1))
    assert torch.allclose(x_ste.detach(), x_hard, atol=0.0, rtol=0.0)


def test_generator_no_ste_returns_hard_only():
    torch.manual_seed(0)
    s = seed_from_araw(torch.randn(2, 16, 16))
    x_hard, x_ste, _ = generate_u_and_binary(s, sigma=1.0, tau=0.5, use_ste=False)
    assert torch.allclose(x_ste, x_hard, atol=0.0, rtol=0.0)

