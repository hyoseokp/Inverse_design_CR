from __future__ import annotations

from dataclasses import dataclass

import torch


def wavelength_grid_nm(n_channels: int, *, start_nm: float = 400.0, end_nm: float = 700.0) -> torch.Tensor:
    """Default wavelength grid for channels.

    The true training grid/downsample may differ; this is a deterministic default used for
    band masking/averages and for mock/dry-run behavior.
    """
    if n_channels <= 1:
        raise ValueError("n_channels must be >= 2")
    return torch.linspace(start_nm, end_nm, steps=n_channels, dtype=torch.float32)


def merge_rggb_to_rgb(t: torch.Tensor) -> torch.Tensor:
    """Merge RGGB (2x2) to RGB (3) by averaging the two greens.

    Input:
      t: (..., 2, 2, C)
    Output:
      rgb: (..., 3, C) in order [R, G, B]
    """
    if t.ndim < 3:
        raise ValueError(f"t must have at least 3 dims, got shape={tuple(t.shape)}")
    if t.shape[-3:-1] != (2, 2):
        raise ValueError(f"t must have trailing dims (...,2,2,C), got shape={tuple(t.shape)}")
    r = t[..., 0, 0, :]
    g = 0.5 * (t[..., 0, 1, :] + t[..., 1, 0, :])
    b = t[..., 1, 1, :]
    return torch.stack([r, g, b], dim=-2)


def downsample_301_to_30_default(t301: torch.Tensor) -> torch.Tensor:
    """Deterministic placeholder downsample for 301->30.

    Blueprint requires training-identical downsample; this default is a stable bin-average
    used for dry-run testing until the true training function is wired in.

    Input: (..., 301)
    Output: (..., 30)
    """
    if t301.shape[-1] != 301:
        raise ValueError(f"expected last dim 301, got {t301.shape[-1]}")
    # 301 points from 400..700 inclusive (301 = 300 intervals).
    # Bin into 30 groups of 10 intervals -> 30 groups of 11 points? We'll use edges in index space.
    # We choose 30 contiguous bins by slicing in index space, with near-equal widths.
    edges = torch.linspace(0, 301, steps=31, dtype=torch.int64)
    outs = []
    for i in range(30):
        a = int(edges[i].item())
        b = int(edges[i + 1].item())
        if b <= a:
            b = a + 1
        outs.append(t301[..., a:b].mean(dim=-1))
    return torch.stack(outs, dim=-1)


def _band_mask(wl_nm: torch.Tensor, start_nm: float, end_nm: float) -> torch.Tensor:
    # Inclusive range; tolerate float endpoints.
    return (wl_nm >= float(start_nm)) & (wl_nm <= float(end_nm))


def _trapz_mean(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if y.shape[-1] != x.shape[-1]:
        raise ValueError("y and x must align on last dim")
    if y.shape[-1] < 2:
        return y.mean(dim=-1)
    area = torch.trapz(y, x=x, dim=-1)
    denom = x[-1] - x[0]
    # Avoid divide-by-zero for degenerate x (shouldn't happen with linspace).
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    return area / denom


@dataclass(frozen=True)
class BandMetrics:
    # In-band (desired) averages: D_b = A_{b,b}
    D_R: torch.Tensor
    D_G: torch.Tensor
    D_B: torch.Tensor
    # Out-of-band averages: O_b = mean_{c!=b} A_{c,b}
    O_R: torch.Tensor
    O_G: torch.Tensor
    O_B: torch.Tensor


def band_averages_rgb(
    rgb: torch.Tensor,
    *,
    wl_nm: torch.Tensor,
    band_ranges_nm: dict[str, tuple[float, float]],
) -> BandMetrics:
    """Compute D_b and O_b from RGB spectra.

    Input:
      rgb: (..., 3, C) in [R,G,B] order.
      wl_nm: (C,) wavelengths in nm.
      band_ranges_nm: {"B": (400,500), "G": (500,600), "R": (600,700)} (or similar)
    """
    if rgb.shape[-2] != 3:
        raise ValueError(f"rgb must have 3 channels at dim -2, got shape={tuple(rgb.shape)}")
    if wl_nm.ndim != 1 or wl_nm.shape[0] != rgb.shape[-1]:
        raise ValueError("wl_nm must be 1D and match rgb last dim")

    # Compute A_{c,b}: average of color c over band b.
    A: dict[tuple[str, str], torch.Tensor] = {}
    for bname, (b0, b1) in band_ranges_nm.items():
        mask = _band_mask(wl_nm, b0, b1)
        if not bool(mask.any().item()):
            raise ValueError(f"band {bname} has empty mask for given wavelength grid")
        x = wl_nm[mask]
        for c_idx, cname in enumerate(["R", "G", "B"]):
            y = rgb[..., c_idx, :][..., mask]
            A[(cname, bname)] = _trapz_mean(y, x)

    # Desired in-band per band: D_b = A_{b,b}
    D_R = A[("R", "R")]
    D_G = A[("G", "G")]
    D_B = A[("B", "B")]

    # Out-of-band per band: O_b = mean_{c!=b} A_{c,b}
    O_R = 0.5 * (A[("G", "R")] + A[("B", "R")])
    O_G = 0.5 * (A[("R", "G")] + A[("B", "G")])
    O_B = 0.5 * (A[("R", "B")] + A[("G", "B")])

    return BandMetrics(D_R=D_R, D_G=D_G, D_B=D_B, O_R=O_R, O_G=O_G, O_B=O_B)

