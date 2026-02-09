from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ProgressLogger:
    progress_dir: Path

    def __post_init__(self) -> None:
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        (self.progress_dir / "previews").mkdir(parents=True, exist_ok=True)

    @property
    def meta_path(self) -> Path:
        return self.progress_dir / "run_meta.json"

    @property
    def metrics_path(self) -> Path:
        return self.progress_dir / "metrics.jsonl"

    def write_meta(self, meta: dict[str, Any]) -> Path:
        # Overwrite; the latest run is what the dashboard should reflect.
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        return self.meta_path

    def log_metrics(self, record: dict[str, Any]) -> None:
        # Append-only JSONL
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def write_topk_snapshot(
        self,
        *,
        step: int,
        seed16_topk: np.ndarray,
        struct128_topk: np.ndarray,
        metrics_topk: dict[str, np.ndarray],
    ) -> Path:
        out = self.progress_dir / f"topk_step-{int(step)}.npz"
        np.savez_compressed(
            out,
            seed16_topk=seed16_topk,
            struct128_topk=struct128_topk,
            **{f"metric_{k}": v for k, v in metrics_topk.items()},
        )
        return out
