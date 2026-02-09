from __future__ import annotations

from pathlib import Path

import numpy as np

from crinv.ranking import rank_by_fdtd


def test_ranking_smoke(tmp_path: Path):
    # Create two candidates where candidate 0 has better in-band behavior under our simplistic score.
    K, C = 2, 30
    fdtd = np.zeros((K, 2, 2, C), dtype=np.float32)
    fdtd[0, :, :, :] = 0.9
    fdtd[1, :, :, :] = 0.1

    res = rank_by_fdtd(fdtd_rggb=fdtd, out_dir=tmp_path / "final")
    assert res.report_path.exists()
    assert res.order.shape == (K,)
    assert int(res.order[0]) == 0

