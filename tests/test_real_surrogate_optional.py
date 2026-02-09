from __future__ import annotations

from pathlib import Path

import pytest
import torch

from crinv.config import InverseDesignConfig
from crinv.surrogate_interface import CRReconSurrogate


def test_crrecon_surrogate_load_and_predict_optional():
    cfg = InverseDesignConfig.from_yaml("configs/inverse.yaml")
    paths_yaml = Path(cfg.paths.paths_yaml)
    if not paths_yaml.exists():
        pytest.skip("configs/paths.yaml not present; real surrogate not configured")

    import yaml

    paths = yaml.safe_load(paths_yaml.read_text(encoding="utf-8")) or {}
    root = Path(paths.get("forward_model_root", ""))
    ckpt = Path(paths.get("forward_checkpoint", ""))
    cfg_yaml = Path(paths.get("forward_config_yaml", ""))
    if not (root.exists() and ckpt.exists() and cfg_yaml.exists()):
        pytest.skip("CR_recon paths not set correctly")

    surr = CRReconSurrogate(
        forward_model_root=root,
        checkpoint_path=ckpt,
        config_yaml=cfg_yaml,
        device=torch.device("cpu"),
    )
    x = torch.zeros((2, 128, 128), dtype=torch.float32)
    y = surr.predict(x)
    assert y.shape[0] == 2
    assert y.shape[1:3] == (2, 2)
    assert y.shape[-1] == surr.n_channels

