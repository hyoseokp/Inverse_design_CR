from __future__ import annotations

import torch


def seed_from_araw(a_raw: torch.Tensor) -> torch.Tensor:
    """Blueprint rule: A_raw -> sigmoid -> structural y=x symmetry.

    S = 0.5 * (sigmoid(A_raw) + sigmoid(A_raw)^T)

    Supports shape [..., N, N].
    """
    if a_raw.ndim < 2:
        raise ValueError(f"a_raw must have at least 2 dims, got shape={tuple(a_raw.shape)}")
    if a_raw.shape[-1] != a_raw.shape[-2]:
        raise ValueError(f"a_raw must be square on last 2 dims, got shape={tuple(a_raw.shape)}")

    a = torch.sigmoid(a_raw)
    return 0.5 * (a + a.transpose(-1, -2))

