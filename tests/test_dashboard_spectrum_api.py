from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from crinv.dashboard_app import create_app
from crinv.surrogate_interface import MockSurrogate


def test_dashboard_spectrum_endpoint(tmp_path: Path):
    prog = tmp_path / "progress"
    prog.mkdir(parents=True, exist_ok=True)

    # Minimal snapshot file.
    struct = np.zeros((2, 128, 128), dtype=np.uint8)
    struct[0, 10:20, 10:20] = 1
    struct[1, 30:40, 30:40] = 1
    seed = np.zeros((2, 16, 16), dtype=np.float32)
    np.savez_compressed(prog / "topk_step-0.npz", struct128_topk=struct, seed16_topk=seed, metric_best_loss=np.array([1.0, 2.0]))

    app = create_app(progress_dir=prog, surrogate=MockSurrogate(n_channels=30))
    c = TestClient(app)

    r = c.get("/api/topk/0/0/spectrum")
    assert r.status_code == 200
    obj = r.json()
    assert obj["n_channels"] == 30
    assert len(obj["rgb"]) == 3
    assert len(obj["rgb"][0]) == 30

