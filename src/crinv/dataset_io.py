from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DatasetPaths:
    root: Path

    @property
    def meta_path(self) -> Path:
        return self.root / "meta.json"

    @property
    def samples_dir(self) -> Path:
        return self.root / "samples"

    def sample_path(self, sample_id: str) -> Path:
        return self.samples_dir / f"{sample_id}.npz"


def init_dataset(root: str | Path) -> DatasetPaths:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    dp = DatasetPaths(root=root)
    dp.samples_dir.mkdir(parents=True, exist_ok=True)
    if not dp.meta_path.exists():
        dp.meta_path.write_text('{"version": 1}\n', encoding="utf-8")
    return dp


def append_sample(
    *,
    dataset_root: str | Path,
    sample_id: str,
    struct128: np.ndarray,
    spectra_rggb: np.ndarray,
) -> Path:
    """Append one validated sample to a simple NPZ-based dataset.

    struct128: (128,128) uint8/bool
    spectra_rggb: (2,2,C) float32 (C=301 or 30)
    """
    dp = init_dataset(dataset_root)
    s = np.asarray(struct128)
    t = np.asarray(spectra_rggb)
    if s.shape != (128, 128):
        raise ValueError(f"struct128 must be (128,128), got {s.shape}")
    if t.ndim != 3 or t.shape[0:2] != (2, 2):
        raise ValueError(f"spectra_rggb must be (2,2,C), got {t.shape}")

    out = dp.sample_path(str(sample_id))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, struct128=s.astype(np.uint8), spectra_rggb=t.astype(np.float32))
    return out


def load_sample(*, dataset_root: str | Path, sample_id: str) -> tuple[np.ndarray, np.ndarray]:
    dp = DatasetPaths(root=Path(dataset_root))
    p = dp.sample_path(str(sample_id))
    with np.load(p) as z:
        return z["struct128"], z["spectra_rggb"]


def list_samples(dataset_root: str | Path) -> list[str]:
    dp = DatasetPaths(root=Path(dataset_root))
    if not dp.samples_dir.exists():
        return []
    ids: list[str] = []
    for p in sorted(dp.samples_dir.glob("*.npz")):
        ids.append(p.stem)
    return ids

