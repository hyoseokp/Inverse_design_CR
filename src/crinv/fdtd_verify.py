from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import FDTDConfig, InverseDesignConfig
from .fdtd_runner import FDTDRunPaths, FDTDRuntimeOptions, run_fdtd_batch
from .gds_export import export_struct128_to_gds
from .lumapi_bridge import LumapiBridge
from .ranking import rank_by_fdtd


def _load_npz_struct_topk(path: Path) -> np.ndarray:
    z = np.load(path, allow_pickle=False)
    if "struct128_topk" not in z.files:
        raise KeyError(f"struct128_topk missing in {path}")
    arr = np.asarray(z["struct128_topk"])
    if arr.ndim != 3:
        raise ValueError(f"expected struct128_topk (K,128,128), got {arr.shape}")
    return arr


def _run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"fdtd_{ts}"


@dataclass(frozen=True)
class FDTDVerifyResult:
    out_dir: Path
    fdtd_rggb_path: Path
    ranking_report: Path | None
    done_ids: list[int]


def verify_topk_with_fdtd(
    *,
    topk_npz: str | Path,
    inverse_cfg: InverseDesignConfig,
    fdtd_cfg: FDTDConfig,
    out_dir: str | Path = "data/fdtd_results",
    k: int | None = None,
    layer_map: str = "1:0",
    cell_prefix: str = "structure",
) -> FDTDVerifyResult:
    """Run Lumerical FDTD on the Top-K structures and write a ranking report.

    Inputs:
      topk_npz: path to a .npz containing struct128_topk (K,128,128)
      inverse_cfg: used only for band ranges when ranking
      fdtd_cfg: lumerical/template runtime config
    Output layout:
      out_dir/<run_id>/
        gds/structure_00000.gds ...
        spectra/structure_00000/spectra.npy (per-structure raw)
        fdtd_rggb.npy (K,2,2,C) stacked
        ranking_report.md
    """
    topk_npz = Path(topk_npz)
    struct_topk = _load_npz_struct_topk(topk_npz)
    K = int(struct_topk.shape[0])
    if k is None:
        k_use = K
    else:
        k_use = int(k)
        k_use = max(1, min(K, k_use))
        struct_topk = struct_topk[:k_use]

    out_root = Path(out_dir)
    run_dir = out_root / _run_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    gds_dir = run_dir / "gds"
    gds_dir.mkdir(parents=True, exist_ok=True)

    spectra_dir = run_dir / "spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)

    # Export GDS for each structure.
    items: list[tuple[int, Path, str]] = []
    for i in range(k_use):
        sid = int(i)
        cell_name = f"{cell_prefix}_{sid:05d}"
        gds_path = gds_dir / f"{cell_name}.gds"
        export_struct128_to_gds(struct_topk[i], out_path=gds_path, structure_id=sid)
        items.append((sid, gds_path, cell_name))

    bridge = LumapiBridge(
        lumerical_root=Path(fdtd_cfg.fdtd.lumerical_root),
        template_fsp=Path(fdtd_cfg.fdtd.template_fsp),
        hide=bool(fdtd_cfg.fdtd.hide),
        layer_map=str(layer_map),
    )
    paths = FDTDRunPaths(out_dir=spectra_dir)
    options = FDTDRuntimeOptions(
        chunk_size=int(fdtd_cfg.fdtd.chunk_size),
        max_retries=int(fdtd_cfg.fdtd.max_retries),
    )
    done = run_fdtd_batch(bridge=bridge, items=items, paths=paths, options=options)

    # Stack RGGB for ranking.
    # Each spectra.npy is expected to be (2,2,C).
    rggb_list: list[np.ndarray] = []
    for sid in range(k_use):
        p = paths.spectra_path(sid)
        if not p.exists():
            raise FileNotFoundError(f"missing spectra for id={sid}: {p}")
        arr = np.load(p)
        if arr.ndim != 3 or arr.shape[0:2] != (2, 2):
            raise ValueError(f"expected spectra (2,2,C) for id={sid}, got {arr.shape}")
        rggb_list.append(arr.astype(np.float32))
    fdtd_rggb = np.stack(rggb_list, axis=0)  # (K,2,2,C)
    fdtd_rggb_path = run_dir / "fdtd_rggb.npy"
    np.save(fdtd_rggb_path, fdtd_rggb)

    # Rank using the same band definition used by inverse.
    rr = rank_by_fdtd(
        fdtd_rggb=fdtd_rggb,
        out_dir=run_dir,
        band_ranges_nm=inverse_cfg.spectra.band_ranges_nm,
    )

    return FDTDVerifyResult(
        out_dir=run_dir,
        fdtd_rggb_path=fdtd_rggb_path,
        ranking_report=rr.report_path,
        done_ids=done,
    )


def resolve_fdtd_cfg(*, fdtd_yaml: str | Path, inverse_cfg: InverseDesignConfig) -> FDTDConfig:
    """Load FDTD config and patch-in machine-local paths.yaml if needed."""
    cfg = FDTDConfig.from_yaml(fdtd_yaml)

    # Prefer configs/paths.yaml (git-ignored) values when fdtd.yaml leaves blanks.
    try:
        import yaml

        paths = Path(inverse_cfg.paths.paths_yaml)
        if paths.exists():
            obj = yaml.safe_load(paths.read_text(encoding="utf-8")) or {}
            if isinstance(obj, dict):
                if not str(cfg.fdtd.lumerical_root).strip():
                    cfg.fdtd.lumerical_root = str(obj.get("lumerical_root", "") or "")
                if not str(cfg.fdtd.template_fsp).strip():
                    cfg.fdtd.template_fsp = str(obj.get("fdtd_template_fsp", "") or "")
    except Exception:
        pass

    if not str(cfg.fdtd.lumerical_root).strip():
        raise ValueError("FDTD lumerical_root not set (configs/fdtd.yaml or configs/paths.yaml)")
    if not str(cfg.fdtd.template_fsp).strip():
        raise ValueError("FDTD template_fsp not set (configs/fdtd.yaml or configs/paths.yaml)")
    return cfg

