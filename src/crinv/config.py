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


class SpectraConfig(BaseModel):
    n_channels: Literal[30, 301] = 30
    # Stored as mapping to keep YAML simple: {B: [400,500], ...}
    band_ranges_nm: dict[str, tuple[float, float]] = Field(
        default_factory=lambda: {
            k: (v.start_nm, v.end_nm) for k, v in DEFAULT_BAND_RANGES_NM.items()
        }
    )


class LossConfig(BaseModel):
    epsilon: float = 1.0e-6
    # Separation margin term (softplus) weight and hyperparameter.
    w_ratio: float = 0.1
    margin_alpha: float = 0.3
    # Diagonal (in-band) efficiency and off-diagonal leakage penalties.
    w_abs: float = 1.0
    w_oob: float = 1.0
    # Crosstalk purity: column-normalized A_{c,b} should approach Identity(3).
    w_purity: float = 0.2
    w_gray: float = 0.1
    w_tv: float = 0.01


class OptimizationConfig(BaseModel):
    n_start: int = 200
    n_steps: int = 2000
    lr: float = 0.01
    topk: int = 50
    robustness_samples: int = 8
    log_every_n_steps: int = 25
    random_seed: int = 0


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
