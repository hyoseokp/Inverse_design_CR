from __future__ import annotations

from pathlib import Path

import numpy as np

from crinv.config import InverseDesignConfig
from crinv.inverse_opt import run_inverse_opt
from crinv.surrogate_interface import MockSurrogate


def test_inverse_opt_dryrun_creates_topk_and_progress(tmp_path: Path):
    cfg = InverseDesignConfig()
    cfg.opt.n_start = 6
    cfg.opt.n_steps = 3
    cfg.opt.topk = 2
    cfg.opt.robustness_samples = 2
    cfg.opt.random_seed = 0

    out_root = tmp_path / "candidates"
    progress = tmp_path / "progress"

    res = run_inverse_opt(
        cfg=cfg,
        surrogate=MockSurrogate(n_channels=int(cfg.spectra.n_channels)),
        out_root=out_root,
        progress_dir=progress,
        device="cpu",
    )
    assert res.topk_path.exists()
    assert (progress / "metrics.jsonl").exists()

    npz = np.load(res.topk_path)
    assert npz["seed16_topk"].shape == (2, 16, 16)
    assert npz["struct128_topk"].shape == (2, 128, 128)

