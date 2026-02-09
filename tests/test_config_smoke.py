from __future__ import annotations

from pathlib import Path

from crinv.config import FDTDConfig, InverseDesignConfig


def test_default_configs_construct():
    InverseDesignConfig()
    FDTDConfig()


def test_inverse_yaml_load_smoke():
    repo_root = Path(__file__).resolve().parents[1]
    cfg = InverseDesignConfig.from_yaml(repo_root / "configs" / "inverse.yaml")
    assert cfg.seed_size == 16
    assert cfg.struct_size == 128
    assert cfg.spectra.n_channels in (30, 301)


def test_fdtd_yaml_load_smoke():
    repo_root = Path(__file__).resolve().parents[1]
    cfg = FDTDConfig.from_yaml(repo_root / "configs" / "fdtd.yaml")
    assert "run" in cfg.fdtd.timeout_s

