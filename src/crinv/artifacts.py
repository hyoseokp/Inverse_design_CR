from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import InverseDesignConfig
from .generator import generate_u_and_binary_cfg


@dataclass(frozen=True)
class TopKPack:
    seed16: np.ndarray  # (K,16,16) float32
    struct128: np.ndarray  # (K,128,128) uint8
    metrics: dict[str, np.ndarray]  # key -> (K,)


def nominal_sigma(cfg: InverseDesignConfig) -> float:
    s = cfg.generator.sigma_set
    if len(s) == 0:
        return 1.0
    return float(s[len(s) // 2])


def struct_from_seed_nominal(seed16: torch.Tensor, *, cfg: InverseDesignConfig) -> torch.Tensor:
    """Generate hard binary struct with nominal (sigma0,tau0)."""
    x_hard, _x_ste, _u = generate_u_and_binary_cfg(
        cfg=cfg,
        s16=seed16,
        sigma=nominal_sigma(cfg),
        tau=cfg.generator.tau0,
        struct_size=cfg.struct_size,
        use_ste=False,
    )
    return x_hard


def save_topk_npz(path: str | Path, pack: TopKPack) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        seed16_topk=pack.seed16,
        struct128_topk=pack.struct128,
        **{f"metric_{k}": v for k, v in pack.metrics.items()},
    )
