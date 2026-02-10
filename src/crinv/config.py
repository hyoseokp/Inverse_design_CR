from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from .constants import DEFAULT_BAND_RANGES_NM


class PathsConfig(BaseModel):
    # Path indirection: we keep a single machine-local YAML for absolute paths.
    paths_yaml: str = Field(default="configs/paths.example.yaml")


class GeneratorConfig(BaseModel):
    sigma_set: list[float] = Field(default_factory=lambda: [0.8, 1.0, 1.2])
    tau0: float = 0.5
    delta_tau: float = 0.05
    # Structure generator backend:
    # - "rule": bilinear upsample + gaussian blur + threshold (fast)
    # - "rule_mfs": differentiable approximation of the dataset rule
    #              (bicubic upsample + gaussian blur + soft-threshold + morphological MFS)
    # - "nn": 16->128 predictor network (trained to match dataset rule incl. MFS)
    backend: Literal["rule", "rule_mfs", "nn"] = "rule"

    # rule_mfs knobs (interpreted in 128x128 pixel units)
    sym_mode: Literal["avg", "upper_copy"] = "avg"
    threshold_temp: float = 0.05  # soft threshold temperature for sigmoid((u-tau)/temp)
    mfs_radius_px: int = 5  # ~ MIN_FEATURE_SIZE/2 in user's script (10 -> 5)
    mfs_iters: int = 2  # keep small; each iter does opening+closing for solid+void
    mfs_kernel: Literal["square", "soft_circle"] = "soft_circle"
    # soft_circle params: higher beta -> closer to hard max/min; dist_scale -> stronger center bias
    mfs_soft_beta: float = 20.0
    mfs_dist_scale: float = 0.25
    # Extra circular padding before MFS iterations (then cropped back).
    # Larger values reduce boundary artifacts at the cost of compute.
    mfs_pad_mult: int = 4


class SpectraConfig(BaseModel):
    n_channels: Literal[30, 301] = 30
    # Stored as mapping to keep YAML simple: {B: [400,500], ...}
    band_ranges_nm: dict[str, tuple[float, float]] = Field(
        default_factory=lambda: {
            k: (v.start_nm, v.end_nm) for k, v in DEFAULT_BAND_RANGES_NM.items()
        }
    )
    # For 30-channel spectra, optionally use fixed channel index groups for band averages.
    # This makes the 3x3 purity matrix exactly correspond to channel bins:
    #   B: 0-9, G: 10-19, R: 20-29 (inclusive).
    band_indices_30: dict[str, tuple[int, int]] = Field(
        default_factory=lambda: {"B": (0, 9), "G": (10, 19), "R": (20, 29)}
    )
    # Optional per-color scaling applied before computing purity matrix A and band metrics.
    # This lets you weight certain colors more strongly at the "spectrum -> A" stage.
    # Default: emphasize green by 2x.
    rgb_weights: dict[str, float] = Field(default_factory=lambda: {"R": 1.0, "G": 2.0, "B": 1.0})


class LossConfig(BaseModel):
    epsilon: float = 1.0e-6
    # Simple, stable objective:
    #  - purity term drives A -> I (signal up, crosstalk down)
    #  - optional abs term explicitly pushes diag(A) -> 1
    #
    # Keep legacy knobs for experimentation, but default to the simple form.
    w_purity: float = 1.0
    w_abs: float = 0.0

    # Legacy/advanced (defaults off)
    w_ratio: float = 0.0
    margin_alpha: float = 0.3
    w_oob: float = 0.0
    w_gray: float = 0.1
    w_tv: float = 0.01

    # Keep structures within the surrogate's training distribution.
    # CR_recon config (reference) uses constraints: sum_min=0.45, sum_max=0.95.
    # Here "sum" means mean fill fraction over pixels.
    w_fill: float = 0.0
    fill_min: float = 0.45
    fill_max: float = 0.95


class OptimizationConfig(BaseModel):
    engine: Literal["adam", "ga"] = "adam"
    n_start: int = 200
    n_steps: int = 2000
    lr: float = 0.01
    topk: int = 50
    robustness_samples: int = 8
    log_every_n_steps: int = 25
    random_seed: int = 0
    # Process candidates in chunks to control memory (important for real surrogate on CPU).
    # 0 disables chunking (process all n_start at once).
    chunk_size: int = 0
    # How to aggregate per-candidate losses into the scalar objective.
    # - "mean": minimize average loss across n_start (default, stable)
    # - "sum":  minimize sum of losses (equivalent to mean with lr scaled by n_start)
    loss_reduction: Literal["mean", "sum"] = "mean"

    # GA-specific knobs (used when engine == "ga")
    ga_elite: int = 8
    ga_tournament_k: int = 5
    ga_crossover_alpha: float = 0.5
    ga_mutation_sigma: float = 0.15
    ga_mutation_p: float = 0.2


class InverseDesignConfig(BaseModel):
    seed_size: int = 16
    struct_size: int = 128

    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    spectra: SpectraConfig = Field(default_factory=SpectraConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    opt: OptimizationConfig = Field(default_factory=OptimizationConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "InverseDesignConfig":
        data = _load_yaml_dict(path)
        return cls.model_validate(data)


class FDTDRuntimeConfig(BaseModel):
    lumerical_root: str = ""
    template_fsp: str = ""
    hide: bool = True
    chunk_size: int = 8
    max_retries: int = 2
    timeout_s: dict[str, int] = Field(
        default_factory=lambda: {"import": 120, "run": 600, "extract": 120}
    )


class FDTDConfig(BaseModel):
    fdtd: FDTDRuntimeConfig = Field(default_factory=FDTDRuntimeConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FDTDConfig":
        data = _load_yaml_dict(path)
        return cls.model_validate(data)


def _load_yaml_dict(path: str | Path) -> dict:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise TypeError(f"YAML root must be a mapping, got: {type(obj).__name__}")
    return obj
