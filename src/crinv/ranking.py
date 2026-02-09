from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .postprocess import PostprocessOptions, process_fdtd_rggb
from .spectral import band_averages_rgb, wavelength_grid_nm


@dataclass(frozen=True)
class RankingResult:
    order: np.ndarray  # (K,) indices sorted best->worst
    scores: np.ndarray  # (K,) higher is better
    report_path: Path


def score_from_band_metrics(m) -> torch.Tensor:
    # Simple FDTD-first score: maximize in-band minus out-of-band.
    return (m.D_R + m.D_G + m.D_B) - (m.O_R + m.O_G + m.O_B)


def rank_by_fdtd(
    *,
    fdtd_rggb: np.ndarray,  # (K,2,2,C)
    out_dir: str | Path = "data/final",
    band_ranges_nm: dict[str, tuple[float, float]] | None = None,
    options: PostprocessOptions | None = None,
    surrogate_rggb: np.ndarray | None = None,
) -> RankingResult:
    """Rank candidates by FDTD metrics and write a concise markdown report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "ranking_report.md"

    rgb = process_fdtd_rggb(fdtd_rggb, options=options)  # (K,3,C)
    C = rgb.shape[-1]
    wl = wavelength_grid_nm(int(C))
    bands = band_ranges_nm or {"B": (400.0, 500.0), "G": (500.0, 600.0), "R": (600.0, 700.0)}
    m = band_averages_rgb(rgb, wl_nm=wl, band_ranges_nm=bands)
    s = score_from_band_metrics(m)  # (K,)

    scores = s.detach().cpu().numpy().astype(np.float32)
    order = np.argsort(-scores)  # best first

    # Optional surrogate-vs-FDTD error (on merged RGB after any downsample).
    err_line = ""
    if surrogate_rggb is not None:
        s_rgb = process_fdtd_rggb(surrogate_rggb, options=options)
        mse = torch.mean((s_rgb - rgb) ** 2).item()
        err_line = f"\nSurrogate vs FDTD RGB MSE (merged): `{mse:.6g}`\n"

    top = int(min(10, len(order)))
    lines = [
        "# FDTD-First Ranking Report",
        "",
        f"Candidates: {len(order)}",
        "",
        "Score = (D_R + D_G + D_B) - (O_R + O_G + O_B)  (higher is better)",
        err_line.strip(),
        "",
        "## Top candidates",
        "",
        "| rank | idx | score |",
        "|---:|---:|---:|",
    ]
    for r in range(top):
        idx = int(order[r])
        lines.append(f"| {r+1} | {idx} | {scores[idx]:.6g} |")
    report_path.write_text("\n".join([l for l in lines if l != ""]) + "\n", encoding="utf-8")

    return RankingResult(order=order, scores=scores, report_path=report_path)

