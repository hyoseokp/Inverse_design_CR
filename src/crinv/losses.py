from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import InverseDesignConfig
from .generator import generate_u_and_binary_cfg
from .seed import seed_from_araw
from .spectral import (
    BandMetrics,
    band_averages_rgb,
    crosstalk_matrix_rgb,
    downsample_301_to_30_default,
    merge_rggb_to_rgb,
    wavelength_grid_nm,
)


def total_variation_2d(u: torch.Tensor) -> torch.Tensor:
    """Anisotropic TV per-sample: mean(|dx| + |dy|) over spatial dims. Input [B,H,W]."""
    if u.ndim != 3:
        raise ValueError(f"u must have shape [B,H,W], got {tuple(u.shape)}")
    dx = torch.abs(u[:, :, 1:] - u[:, :, :-1]).mean(dim=(1, 2))
    dy = torch.abs(u[:, 1:, :] - u[:, :-1, :]).mean(dim=(1, 2))
    return dx + dy


def gray_penalty(u: torch.Tensor) -> torch.Tensor:
    """Per-sample mean(u*(1-u)) encourages binarization."""
    return (u * (1.0 - u)).mean(dim=(1, 2))


@dataclass(frozen=True)
class LossTerms:
    # Per-sample vectors shaped [B]
    loss_total: torch.Tensor
    loss_spec: torch.Tensor
    loss_reg: torch.Tensor
    loss_ratio: torch.Tensor
    loss_abs: torch.Tensor
    loss_oob: torch.Tensor
    loss_purity: torch.Tensor
    loss_gray: torch.Tensor
    loss_tv: torch.Tensor
    loss_fill: torch.Tensor
    metrics: BandMetrics


def spectral_terms_from_rggb(
    t_rggb: torch.Tensor,
    *,
    n_channels_target: int,
    band_ranges_nm: dict[str, tuple[float, float]],
    band_indices_30: dict[str, tuple[int, int]] | None = None,
    rgb_weights: dict[str, float] | None = None,
) -> tuple[BandMetrics, torch.Tensor, torch.Tensor]:
    """Return (band_metrics, rgb_30_or_target) for a single batch.

    t_rggb: [B,2,2,C]
    """
    rgb = merge_rggb_to_rgb(t_rggb)  # [B,3,C]
    C = rgb.shape[-1]
    if C == 301 and n_channels_target == 30:
        rgb = downsample_301_to_30_default(rgb)
        C = 30

    # Optional per-color scaling before computing purity matrix A.
    # This is intentionally applied at the "spectrum -> A" stage (as requested).
    if rgb_weights:
        wR = float(rgb_weights.get("R", 1.0))
        wG = float(rgb_weights.get("G", 1.0))
        wB = float(rgb_weights.get("B", 1.0))
        if (wR != 1.0) or (wG != 1.0) or (wB != 1.0):
            rgb = rgb.clone()
            rgb[:, 0, :] = rgb[:, 0, :] * wR
            rgb[:, 1, :] = rgb[:, 1, :] * wG
            rgb[:, 2, :] = rgb[:, 2, :] * wB
    wl = wavelength_grid_nm(C)
    # A_{c,b}: (...,3,3) with c,b in (R,G,B).
    A = crosstalk_matrix_rgb(rgb, wl_nm=wl, band_ranges_nm=band_ranges_nm, band_indices_30=band_indices_30)
    m = band_averages_rgb(rgb, wl_nm=wl, band_ranges_nm=band_ranges_nm, band_indices_30=band_indices_30)
    return m, rgb, A


def compute_loss_from_surrogate(
    *,
    cfg: InverseDesignConfig,
    t_rggb: torch.Tensor,
    u: torch.Tensor,
    x_struct: torch.Tensor | None = None,
) -> LossTerms:
    """Compute blueprint losses from surrogate spectra and generator field u."""
    m, _rgb, A = spectral_terms_from_rggb(
        t_rggb,
        n_channels_target=int(cfg.spectra.n_channels),
        band_ranges_nm=cfg.spectra.band_ranges_nm,
        band_indices_30=getattr(cfg.spectra, "band_indices_30", None),
        rgb_weights=getattr(cfg.spectra, "rgb_weights", None),
    )

    eps = float(cfg.loss.epsilon)
    alpha = float(getattr(cfg.loss, "margin_alpha", 0.3))

    # Simple objective is defined on the 3x3 purity matrix A:
    #   A_{c,b} = mean_{k in band-bin b} s_{c,k}
    # with rows c in (R,G,B) and cols b in (R,G,B).
    # Purity drives A -> I (diagonal up, off-diagonal down).
    I = torch.eye(3, device=A.device, dtype=A.dtype).unsqueeze(0)  # (1,3,3)
    loss_purity = ((A - I) ** 2).sum(dim=(1, 2))

    # Optional explicit diagonal efficiency (can be redundant with purity).
    diagA = torch.stack([A[:, 0, 0], A[:, 1, 1], A[:, 2, 2]], dim=-1)
    loss_abs = ((1.0 - diagA) ** 2).sum(dim=-1)

    # Legacy/advanced terms (still computed for logging, but default weights are 0).
    loss_ratio = (
        F.softplus(m.O_R - alpha * m.D_R)
        + F.softplus(m.O_G - alpha * m.D_G)
        + F.softplus(m.O_B - alpha * m.D_B)
    )
    loss_oob = (m.O_R ** 2) + (m.O_G ** 2) + (m.O_B ** 2)

    loss_gray = gray_penalty(u)
    loss_tv = total_variation_2d(u)

    w_ratio = float(cfg.loss.w_ratio)
    w_abs = float(cfg.loss.w_abs)
    w_oob = float(getattr(cfg.loss, "w_oob", 1.0))
    w_purity = float(getattr(cfg.loss, "w_purity", 0.0))
    w_gray = float(cfg.loss.w_gray)
    w_tv = float(cfg.loss.w_tv)
    w_fill = float(getattr(cfg.loss, "w_fill", 0.0))
    fill_min = float(getattr(cfg.loss, "fill_min", 0.0))
    fill_max = float(getattr(cfg.loss, "fill_max", 1.0))

    loss_spec = (w_ratio * loss_ratio) + (w_abs * loss_abs) + (w_oob * loss_oob) + (w_purity * loss_purity)
    loss_reg = (w_gray * loss_gray) + (w_tv * loss_tv)

    # Fill-fraction constraint (mean over pixels). Penalize being outside [fill_min, fill_max].
    # This matters because the surrogate is trained on structures within a limited density range.
    if x_struct is None:
        loss_fill = torch.zeros_like(loss_reg)
    else:
        if x_struct.ndim != 3:
            raise ValueError(f"x_struct must be [B,H,W], got {tuple(x_struct.shape)}")
        fill = x_struct.mean(dim=(1, 2))
        lo = F.softplus(fill_min - fill)
        hi = F.softplus(fill - fill_max)
        loss_fill = (lo * lo) + (hi * hi)

    loss_total = loss_spec + loss_reg + (w_fill * loss_fill)

    return LossTerms(
        loss_total=loss_total,
        loss_spec=loss_spec,
        loss_reg=loss_reg,
        loss_ratio=loss_ratio,
        loss_abs=loss_abs,
        loss_oob=loss_oob,
        loss_purity=loss_purity,
        loss_gray=loss_gray,
        loss_tv=loss_tv,
        loss_fill=loss_fill,
        metrics=m,
    )


def robust_mc_loss(
    *,
    cfg: InverseDesignConfig,
    a_raw: torch.Tensor,
    surrogate_predict_fn,
    step_seed: int | None = None,
) -> LossTerms:
    """Monte Carlo estimate of E_{sigma,tau}[loss].

    surrogate_predict_fn: callable(x_binary: [B,128,128]) -> t_rggb: [B,2,2,C]
    a_raw: [B,16,16] requires_grad=True
    """
    B = a_raw.shape[0]
    s = seed_from_araw(a_raw)

    n = int(cfg.opt.robustness_samples)
    # Deterministic sampling per step if seed provided.
    samples = []
    base_seed = cfg.opt.random_seed if step_seed is None else int(step_seed)
    from .ops import sample_sigma_tau  # local import to avoid cycles

    samples = sample_sigma_tau(
        sigma_set=cfg.generator.sigma_set,
        tau0=cfg.generator.tau0,
        delta_tau=cfg.generator.delta_tau,
        n=n,
        seed=base_seed,
        device=a_raw.device,
        dtype=a_raw.dtype,
    )

    # Accumulate per-sample vectors.
    acc_total = torch.zeros((B,), device=a_raw.device, dtype=a_raw.dtype)
    acc_spec = torch.zeros((B,), device=a_raw.device, dtype=a_raw.dtype)
    acc_reg = torch.zeros((B,), device=a_raw.device, dtype=a_raw.dtype)
    acc_ratio = torch.zeros((B,), device=a_raw.device, dtype=a_raw.dtype)
    acc_abs = torch.zeros((B,), device=a_raw.device, dtype=a_raw.dtype)
    acc_oob = torch.zeros((B,), device=a_raw.device, dtype=a_raw.dtype)
    acc_purity = torch.zeros((B,), device=a_raw.device, dtype=a_raw.dtype)
    acc_gray = torch.zeros((B,), device=a_raw.device, dtype=a_raw.dtype)
    acc_tv = torch.zeros((B,), device=a_raw.device, dtype=a_raw.dtype)
    acc_fill = torch.zeros((B,), device=a_raw.device, dtype=a_raw.dtype)

    # Metrics: per-sample, averaged over MC.
    acc_D_R = torch.zeros((B,), device=a_raw.device, dtype=a_raw.dtype)
    acc_D_G = torch.zeros((B,), device=a_raw.device, dtype=a_raw.dtype)
    acc_D_B = torch.zeros((B,), device=a_raw.device, dtype=a_raw.dtype)
    acc_O_R = torch.zeros((B,), device=a_raw.device, dtype=a_raw.dtype)
    acc_O_G = torch.zeros((B,), device=a_raw.device, dtype=a_raw.dtype)
    acc_O_B = torch.zeros((B,), device=a_raw.device, dtype=a_raw.dtype)

    for samp in samples:
        # Use configured generator backend (rule/nn).
        x_hard, x_ste, u = generate_u_and_binary_cfg(
            cfg=cfg, s16=s, sigma=samp.sigma, tau=samp.tau, struct_size=cfg.struct_size, use_ste=True
        )
        # Surrogate input should be binary on forward path; x_ste is forward-equal to x_hard.
        t = surrogate_predict_fn(x_ste)
        terms = compute_loss_from_surrogate(cfg=cfg, t_rggb=t, u=u, x_struct=x_ste)
        acc_total = acc_total + terms.loss_total
        acc_spec = acc_spec + terms.loss_spec
        acc_reg = acc_reg + terms.loss_reg
        acc_ratio = acc_ratio + terms.loss_ratio
        acc_abs = acc_abs + terms.loss_abs
        acc_oob = acc_oob + terms.loss_oob
        acc_purity = acc_purity + terms.loss_purity
        acc_gray = acc_gray + terms.loss_gray
        acc_tv = acc_tv + terms.loss_tv
        acc_fill = acc_fill + terms.loss_fill

        acc_D_R = acc_D_R + terms.metrics.D_R
        acc_D_G = acc_D_G + terms.metrics.D_G
        acc_D_B = acc_D_B + terms.metrics.D_B
        acc_O_R = acc_O_R + terms.metrics.O_R
        acc_O_G = acc_O_G + terms.metrics.O_G
        acc_O_B = acc_O_B + terms.metrics.O_B

    inv_n = 1.0 / float(len(samples))
    metrics = BandMetrics(
        D_R=acc_D_R * inv_n,
        D_G=acc_D_G * inv_n,
        D_B=acc_D_B * inv_n,
        O_R=acc_O_R * inv_n,
        O_G=acc_O_G * inv_n,
        O_B=acc_O_B * inv_n,
    )
    # Return per-sample vectors (averaged across samples).
    return LossTerms(
        loss_total=acc_total * inv_n,
        loss_spec=acc_spec * inv_n,
        loss_reg=acc_reg * inv_n,
        loss_ratio=acc_ratio * inv_n,
        loss_abs=acc_abs * inv_n,
        loss_oob=acc_oob * inv_n,
        loss_purity=acc_purity * inv_n,
        loss_gray=acc_gray * inv_n,
        loss_tv=acc_tv * inv_n,
        loss_fill=acc_fill * inv_n,
        metrics=metrics,
    )
