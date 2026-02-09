from __future__ import annotations

from pathlib import Path

import numpy as np

from crinv.gds_export import export_struct128_to_gds


def test_gds_export_smoke(tmp_path: Path):
    struct = np.zeros((16, 16), dtype=np.uint8)
    struct[4:12, 5:11] = 1  # rectangle

    out = tmp_path / "structure_00001.gds"
    p = export_struct128_to_gds(struct, out_path=out, structure_id=1)
    assert p.exists()
    assert p.stat().st_size > 0

