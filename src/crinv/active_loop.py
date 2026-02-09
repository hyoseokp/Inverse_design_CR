from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .dataset_io import append_sample


class SurrogateFineTuner(Protocol):
    """Interface stub for forward surrogate fine-tune.

    Real implementation will be project-specific (data_CR loaders, GPU training, etc).
    """

    def fine_tune(self, *, dataset_root: Path) -> None: ...


@dataclass(frozen=True)
class ActiveLoopConfig:
    dataset_root: Path = Path("data/active_dataset")


def append_validated_topm(
    *,
    cfg: ActiveLoopConfig,
    struct128_topm: np.ndarray,  # (M,128,128)
    spectra_rggb_topm: np.ndarray,  # (M,2,2,C)
    id_prefix: str = "fdtd",
) -> list[Path]:
    if struct128_topm.shape[1:] != (128, 128):
        raise ValueError(f"struct128_topm must be (M,128,128), got {struct128_topm.shape}")
    if spectra_rggb_topm.ndim != 4 or spectra_rggb_topm.shape[1:3] != (2, 2):
        raise ValueError(f"spectra_rggb_topm must be (M,2,2,C), got {spectra_rggb_topm.shape}")
    if struct128_topm.shape[0] != spectra_rggb_topm.shape[0]:
        raise ValueError("M mismatch between struct and spectra arrays")

    out: list[Path] = []
    M = struct128_topm.shape[0]
    for i in range(M):
        sid = f"{id_prefix}_{i:05d}"
        out.append(
            append_sample(
                dataset_root=cfg.dataset_root,
                sample_id=sid,
                struct128=struct128_topm[i],
                spectra_rggb=spectra_rggb_topm[i],
            )
        )
    return out


def fine_tune_forward_stub(*, fine_tuner: SurrogateFineTuner, cfg: ActiveLoopConfig) -> None:
    """Hook point. External GPU training is intentionally not run here."""
    fine_tuner.fine_tune(dataset_root=cfg.dataset_root)

