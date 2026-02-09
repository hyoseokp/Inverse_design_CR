from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BandRangeNm:
    start_nm: float
    end_nm: float


DEFAULT_BAND_RANGES_NM: dict[str, BandRangeNm] = {
    # Blueprint v1.3
    "B": BandRangeNm(400.0, 500.0),
    "G": BandRangeNm(500.0, 600.0),
    "R": BandRangeNm(600.0, 700.0),
}

