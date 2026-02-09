from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def upsample_bilinear_2d(x: torch.Tensor, out_hw: int) -> torch.Tensor:
    """Bilinear upsample for tensors shaped [B,H,W] -> [B,out_hw,out_hw]."""
    if x.ndim != 3:
        raise ValueError(f"x must have shape [B,H,W], got {tuple(x.shape)}")
    x4 = x.unsqueeze(1)  # [B,1,H,W]
    y4 = F.interpolate(x4, size=(out_hw, out_hw), mode="bilinear", align_corners=False)
    return y4.squeeze(1)


def gaussian_kernel1d(sigma: float, truncate: float = 3.0, device=None, dtype=None) -> torch.Tensor:
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    radius = int(truncate * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x**2) / (2 * sigma**2))
    k = k / k.sum()
    return k


def gaussian_blur_2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur for [B,H,W] via separable conv; keeps shape."""
    if x.ndim != 3:
        raise ValueError(f"x must have shape [B,H,W], got {tuple(x.shape)}")
    b, h, w = x.shape
    k1 = gaussian_kernel1d(sigma, device=x.device, dtype=x.dtype)
    # Separable conv: horizontal then vertical
    x4 = x.unsqueeze(1)  # [B,1,H,W]
    kx = k1.view(1, 1, 1, -1)
    ky = k1.view(1, 1, -1, 1)
    pad_x = (kx.shape[-1] // 2, kx.shape[-1] // 2, 0, 0)
    pad_y = (0, 0, ky.shape[-2] // 2, ky.shape[-2] // 2)
    y = F.conv2d(F.pad(x4, pad_x, mode="reflect"), kx)
    y = F.conv2d(F.pad(y, pad_y, mode="reflect"), ky)
    return y.squeeze(1)


def hard_threshold(u: torch.Tensor, tau: float) -> torch.Tensor:
    return (u > tau).to(dtype=u.dtype)


def ste_threshold(u: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (X_hard, X_ste).

    Forward: X_ste == X_hard
    Backward: dX_ste/du = 1 (straight-through)
    """
    x_hard = hard_threshold(u, tau)
    x_ste = x_hard + (u - u.detach())
    return x_hard, x_ste


@dataclass(frozen=True)
class RobustSample:
    sigma: float
    tau: float


def sample_sigma_tau(
    *,
    sigma_set: list[float],
    tau0: float,
    delta_tau: float,
    n: int,
    seed: int | None = None,
    device=None,
    dtype=None,
) -> list[RobustSample]:
    """Sample (sigma, tau) pairs for robust design.

    - sigma: uniform over discrete set
    - tau: uniform in [tau0 - delta_tau, tau0 + delta_tau]
    """
    if len(sigma_set) == 0:
        raise ValueError("sigma_set must be non-empty")
    if n <= 0:
        raise ValueError("n must be > 0")
    if delta_tau < 0:
        raise ValueError("delta_tau must be >= 0")

    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(int(seed))

    sigma_idx = torch.randint(low=0, high=len(sigma_set), size=(n,), generator=g, device=device)
    tau = (tau0 - delta_tau) + (2.0 * delta_tau) * torch.rand((n,), generator=g, device=device, dtype=dtype)

    out: list[RobustSample] = []
    for i in range(n):
        out.append(RobustSample(sigma=float(sigma_set[int(sigma_idx[i])]), tau=float(tau[i].item())))
    return out

