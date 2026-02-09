from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .spectral import downsample_301_to_30_default, merge_rggb_to_rgb


@dataclass(frozen=True)
class PostprocessOptions:
    flip_sign: bool = False  # some pipelines use T = -T depending on training convention
    n_channels_target: int = 30


def process_fdtd_rggb(
    t_rggb: np.ndarray,
    *,
    options: PostprocessOptions | None = None,
) -> torch.Tensor:
    """Convert raw FDTD RGGB spectra to torch RGB spectra for metrics.

    Input:
      t_rggb: (K,2,2,301) or (K,2,2,30)
    Output:
      rgb: (K,3,30) (or target channels)
    """
    options = options or PostprocessOptions()
    arr = np.asarray(t_rggb)
    if arr.ndim != 4 or arr.shape[1:3] != (2, 2):
        raise ValueError(f"expected (K,2,2,C), got {arr.shape}")

    t = torch.from_numpy(arr.astype(np.float32))
    if options.flip_sign:
        t = -t

    rgb = merge_rggb_to_rgb(t)  # (K,3,C)
    C = rgb.shape[-1]
    if C == 301 and int(options.n_channels_target) == 30:
        rgb = downsample_301_to_30_default(rgb)
    return rgb

