from __future__ import annotations

from pathlib import Path

import numpy as np

from crinv.active_loop import ActiveLoopConfig, append_validated_topm
from crinv.dataset_io import append_sample, list_samples, load_sample


def test_dataset_append_and_load(tmp_path: Path):
    root = tmp_path / "ds"
    struct = np.zeros((128, 128), dtype=np.uint8)
    struct[10:20, 10:20] = 1
    spectra = np.ones((2, 2, 30), dtype=np.float32) * 0.123

    p = append_sample(dataset_root=root, sample_id="s00001", struct128=struct, spectra_rggb=spectra)
    assert p.exists()

    ids = list_samples(root)
    assert ids == ["s00001"]

    s2, t2 = load_sample(dataset_root=root, sample_id="s00001")
    assert s2.shape == (128, 128)
    assert t2.shape == (2, 2, 30)


def test_active_loop_append_topm(tmp_path: Path):
    cfg = ActiveLoopConfig(dataset_root=tmp_path / "active")
    struct = np.zeros((2, 128, 128), dtype=np.uint8)
    spectra = np.zeros((2, 2, 2, 30), dtype=np.float32)
    out = append_validated_topm(cfg=cfg, struct128_topm=struct, spectra_rggb_topm=spectra, id_prefix="x")
    assert len(out) == 2
    assert out[0].exists()
    assert out[1].exists()

