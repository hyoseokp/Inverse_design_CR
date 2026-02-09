from __future__ import annotations

import torch

from .config import GeneratorConfig
from .ops import gaussian_blur_2d, sample_sigma_tau, ste_threshold, upsample_bilinear_2d


def generate_u_and_binary(
    s16: torch.Tensor,
    *,
    sigma: float,
    tau: float,
    struct_size: int = 128,
    use_ste: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rule-based generator pipeline.

    Inputs:
      s16: [B,16,16] in [0,1] (symmetric seed)
    Outputs:
      x_hard: [B,128,128] (0/1 float)
      x_ste:  [B,128,128] (forward equals x_hard; backward passes grad from u)
      u:      [B,128,128] (blurred continuous field)
    """
    if s16.ndim != 3:
        raise ValueError(f"s16 must have shape [B,16,16], got {tuple(s16.shape)}")
    u0 = upsample_bilinear_2d(s16, out_hw=struct_size)
    u = gaussian_blur_2d(u0, sigma=sigma)
    x_hard, x_ste = ste_threshold(u, tau)
    if not use_ste:
        x_ste = x_hard.detach()
    return x_hard, x_ste, u


def sample_robust_params(
    cfg: GeneratorConfig,
    *,
    n: int,
    seed: int | None = None,
    device=None,
    dtype=None,
) -> list[tuple[float, float]]:
    samples = sample_sigma_tau(
        sigma_set=cfg.sigma_set,
        tau0=cfg.tau0,
        delta_tau=cfg.delta_tau,
        n=n,
        seed=seed,
        device=device,
        dtype=dtype,
    )
    return [(s.sigma, s.tau) for s in samples]

