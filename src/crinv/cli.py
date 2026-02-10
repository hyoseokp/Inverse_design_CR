from __future__ import annotations

import argparse
import time
from pathlib import Path
import threading

from .config import InverseDesignConfig
from .inverse_opt import run_inverse_opt
from .fdtd_verify import resolve_fdtd_cfg, verify_topk_with_fdtd
from .ranking import rank_by_fdtd
from .surrogate_interface import CRReconSurrogate, MockSurrogate


def _load_local_paths(paths_yaml: str) -> dict:
    import yaml

    p = Path(paths_yaml)
    if not p.exists():
        raise FileNotFoundError(f"paths.yaml not found: {paths_yaml}")
    with p.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise TypeError("paths.yaml root must be a mapping")
    return obj


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="crinv")
    sub = p.add_subparsers(dest="cmd", required=True)

    inv = sub.add_parser("inverse", help="Run inverse optimization")
    inv.add_argument("--config", default="configs/inverse.yaml")
    inv.add_argument("--dry-run", action="store_true", help="Use MockSurrogate (no external model)")
    inv.add_argument("--n-start", type=int, default=None)
    inv.add_argument("--n-steps", type=int, default=None)
    inv.add_argument("--topk", type=int, default=None)
    inv.add_argument("--robustness-samples", type=int, default=None)
    inv.add_argument("--chunk-size", type=int, default=None, help="Process n_start candidates in chunks (memory control)")
    inv.add_argument(
        "--loss-reduction",
        choices=["mean", "sum"],
        default=None,
        help="Aggregate per-candidate losses into objective (mean|sum).",
    )
    inv.add_argument(
        "--fdtd-verify",
        choices=["on", "off", "ask"],
        default="ask",
        help="After inverse completes, run Lumerical FDTD on the produced Top-K (on|off|ask).",
    )
    inv.add_argument("--fdtd-k", type=int, default=None, help="How many top-k items to verify via FDTD (default: all)")
    inv.add_argument("--fdtd-config", default="configs/fdtd.yaml", help="FDTD config YAML for verification")
    inv.add_argument(
        "--fdtd-every",
        type=int,
        default=0,
        help="If >0, run FDTD verification periodically on topk snapshots every N steps (non-blocking).",
    )
    inv.add_argument(
        "--print-every",
        type=int,
        default=None,
        help="Print progress every N steps (default: auto based on n_steps)",
    )
    inv.add_argument("--out-root", default="data/candidates")
    inv.add_argument("--progress-dir", default="data/progress")
    inv.add_argument("--device", default="cpu")

    rank = sub.add_parser("rank", help="Rank by FDTD (FDTD-first)")
    rank.add_argument("--fdtd-npy", required=True, help="Path to FDTD RGGB spectra .npy (K,2,2,C)")
    rank.add_argument("--out-dir", default="data/final")

    fv = sub.add_parser("fdtd-verify", help="Run Lumerical FDTD on Top-K structures")
    fv.add_argument("--inverse-config", default="configs/inverse.yaml")
    fv.add_argument("--fdtd-config", default="configs/fdtd.yaml")
    fv.add_argument("--topk-npz", required=True, help="Path to topk_pack.npz (must contain struct128_topk)")
    fv.add_argument("--out-dir", default="data/fdtd_results")
    fv.add_argument("--k", type=int, default=None, help="How many top-k to verify (default: all in npz)")
    fv.add_argument("--layer-map", default="1:0", help="GDS layer map for Lumerical gdsimport, e.g. 1:0")

    verify = sub.add_parser("integration-verify", help="Write integration verify reports")
    verify.add_argument("--out-dir", default="REPORTS")
    verify.add_argument("--config", default="configs/inverse.yaml")

    return p


def cmd_inverse(args: argparse.Namespace) -> int:
    cfg = InverseDesignConfig.from_yaml(args.config)
    if args.n_start is not None:
        cfg.opt.n_start = int(args.n_start)
    if args.n_steps is not None:
        cfg.opt.n_steps = int(args.n_steps)
    if args.topk is not None:
        cfg.opt.topk = int(args.topk)
    if args.robustness_samples is not None:
        cfg.opt.robustness_samples = int(args.robustness_samples)
    if args.chunk_size is not None:
        cfg.opt.chunk_size = int(args.chunk_size)
    if args.loss_reduction is not None:
        cfg.opt.loss_reduction = str(args.loss_reduction)

    if args.dry_run:
        # If the user didn't override via flags, keep dry-run small and fast.
        if (
            args.n_start is None
            and args.n_steps is None
            and args.topk is None
            and args.robustness_samples is None
        ):
            cfg.opt.n_start = 8
            cfg.opt.n_steps = 10
            cfg.opt.topk = 4
            cfg.opt.robustness_samples = 2
        surrogate = MockSurrogate(n_channels=int(cfg.spectra.n_channels))
        surrogate_name = "MockSurrogate"
    else:
        # Real surrogate wiring: CR_recon checkpoint loader.
        import torch

        paths = _load_local_paths(cfg.paths.paths_yaml)
        forward_root = Path(paths.get("forward_model_root", ""))
        forward_cfg = Path(paths.get("forward_config_yaml", forward_root / "configs" / "default.yaml"))
        forward_ckpt = Path(paths.get("forward_checkpoint", forward_root / "outputs" / "cnn_xattn_best.pt"))
        surrogate = CRReconSurrogate(
            forward_model_root=forward_root,
            config_yaml=forward_cfg,
            checkpoint_path=forward_ckpt,
            device=torch.device(args.device),
        )
        surrogate_name = "CRReconSurrogate"

    print_every = args.print_every
    if print_every is None:
        # Small runs: more chatty. Big runs: keep noise down.
        print_every = 1 if int(cfg.opt.n_steps) <= 200 else max(5, int(cfg.opt.log_every_n_steps))
    print_every = max(1, int(print_every))

    print(
        f"[START] inverse: n_start={cfg.opt.n_start} n_steps={cfg.opt.n_steps} topk={cfg.opt.topk} "
        f"robustness_samples={cfg.opt.robustness_samples} device={args.device} surrogate={surrogate_name} "
        f"progress_dir={args.progress_dir} gen_backend={cfg.generator.backend} chunk_size={getattr(cfg.opt,'chunk_size',0)}"
    )

    t0 = time.monotonic()
    last_print_step = -1

    # Decide FDTD mode up-front so we can wire periodic verification during the run.
    do_fdtd = str(args.fdtd_verify)
    if do_fdtd == "ask":
        try:
            ans = input("Run FDTD verification on Top-K now? [y/N] ").strip().lower()
            do_fdtd = "on" if ans in ("y", "yes") else "off"
        except Exception:
            do_fdtd = "off"

    fdtd_every = int(getattr(args, "fdtd_every", 0) or 0)
    if fdtd_every < 0:
        raise ValueError("--fdtd-every must be >= 0")

    fdtd_cfg = None
    if do_fdtd == "on":
        try:
            fdtd_cfg = resolve_fdtd_cfg(fdtd_yaml=args.fdtd_config, inverse_cfg=cfg)
        except Exception as e:
            print(f"[WARN] FDTD config not ready; disabling FDTD verify: {e}")
            do_fdtd = "off"

    class _FDTDScheduler:
        def __init__(self):
            self.lock = threading.Lock()
            self.thread: threading.Thread | None = None
            self.pending: tuple[int, str] | None = None
            self.verified: set[int] = set()

        def request(self, *, step: int, topk_npz: str) -> None:
            with self.lock:
                if step in self.verified:
                    return
                self.pending = (int(step), str(topk_npz))
                self._maybe_start_locked()

        def _maybe_start_locked(self) -> None:
            if self.thread is not None and self.thread.is_alive():
                return
            if self.pending is None:
                return
            step, topk_npz = self.pending
            self.pending = None

            def _worker() -> None:
                try:
                    assert fdtd_cfg is not None
                    v = verify_topk_with_fdtd(
                        topk_npz=topk_npz,
                        inverse_cfg=cfg,
                        fdtd_cfg=fdtd_cfg,
                        out_dir="data/fdtd_results",
                        k=args.fdtd_k,
                    )
                    import shutil
                    import json as _json

                    pdir = Path(args.progress_dir)
                    pdir.mkdir(parents=True, exist_ok=True)
                    dst = pdir / f"fdtd_rggb_step-{int(step)}.npy"
                    shutil.copyfile(v.fdtd_rggb_path, dst)
                    meta = {"step": int(step), "k": int(args.fdtd_k) if args.fdtd_k is not None else None, "out_dir": str(v.out_dir)}
                    (pdir / "fdtd_meta.json").write_text(_json.dumps(meta, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
                    print(f"[OK] fdtd-verify saved: {dst}", flush=True)
                    with self.lock:
                        self.verified.add(int(step))
                except Exception as e:
                    print(f"[WARN] fdtd-verify failed (step={step}): {e}", flush=True)
                finally:
                    with self.lock:
                        self.thread = None
                        self._maybe_start_locked()

            self.thread = threading.Thread(target=_worker, daemon=False)
            self.thread.start()

        def drain(self) -> None:
            # Block until no worker and no pending request remain.
            while True:
                with self.lock:
                    th = self.thread
                    pending = self.pending
                if th is None and pending is None:
                    return
                if th is not None:
                    th.join(timeout=0.5)

    fdtd_sched = _FDTDScheduler() if (do_fdtd == "on" and fdtd_every > 0 and fdtd_cfg is not None) else None

    def _hook(info: dict) -> None:
        nonlocal last_print_step
        step = int(info.get("step", -1))
        n_steps = int(info.get("n_steps", cfg.opt.n_steps))
        if (step % print_every != 0) and (step != n_steps - 1):
            return
        # Avoid double-prints if hook is called multiple times per step.
        if step == last_print_step:
            return
        last_print_step = step
        elapsed = time.monotonic() - t0
        loss_total = float(info.get("loss_total", float("nan")))
        print(f"[STEP {step}/{n_steps-1}] loss_total={loss_total:.6f} elapsed_s={elapsed:.1f}", flush=True)

    def _snap_hook(info: dict) -> None:
        if fdtd_sched is None:
            return
        step = int(info.get("step", -1))
        n_steps = int(info.get("n_steps", cfg.opt.n_steps))
        topk_npz = str(info.get("topk_npz", "") or "")
        if not topk_npz:
            return
        if (step % fdtd_every != 0) and (step != n_steps - 1):
            return
        p = Path(topk_npz)
        if not p.exists():
            return
        fdtd_sched.request(step=step, topk_npz=topk_npz)

    res = run_inverse_opt(
        cfg=cfg,
        surrogate=surrogate,
        out_root=args.out_root,
        progress_dir=args.progress_dir,
        device=args.device,
        progress_hook=_hook,
        snapshot_hook=_snap_hook,
    )
    print(f"[OK] inverse done: run_dir={res.run_dir} topk={res.topk_path}")

    # If periodic verification was enabled, drain the queue so the last requested
    # step is written to progress_dir for the dashboard.
    if fdtd_sched is not None:
        fdtd_sched.drain()

    # Optional one-shot FDTD verification on Top-K (end-only).
    if do_fdtd == "on" and fdtd_every <= 0:
        try:
            inv_cfg = cfg
            assert fdtd_cfg is not None
            v = verify_topk_with_fdtd(
                topk_npz=res.topk_path,
                inverse_cfg=inv_cfg,
                fdtd_cfg=fdtd_cfg,
                out_dir="data/fdtd_results",
                k=args.fdtd_k,
            )
            # Also copy a per-step artifact into progress_dir for dashboard overlay.
            import shutil
            import numpy as np
            from pathlib import Path

            step_last = int(cfg.opt.n_steps) - 1
            pdir = Path(args.progress_dir)
            pdir.mkdir(parents=True, exist_ok=True)
            dst = pdir / f"fdtd_rggb_step-{step_last}.npy"
            shutil.copyfile(v.fdtd_rggb_path, dst)
            meta = {"step": step_last, "k": int(args.fdtd_k) if args.fdtd_k is not None else None, "out_dir": str(v.out_dir)}
            (pdir / "fdtd_meta.json").write_text(__import__("json").dumps(meta, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
            print(f"[OK] fdtd-verify saved: {dst}")
        except Exception as e:
            print(f"[WARN] fdtd-verify failed: {e}")
    return 0


def cmd_rank(args: argparse.Namespace) -> int:
    import numpy as np

    fdtd = np.load(args.fdtd_npy)
    rank_by_fdtd(fdtd_rggb=fdtd, out_dir=args.out_dir)
    return 0


def cmd_fdtd_verify(args: argparse.Namespace) -> int:
    inv_cfg = InverseDesignConfig.from_yaml(args.inverse_config)
    fdtd_cfg = resolve_fdtd_cfg(fdtd_yaml=args.fdtd_config, inverse_cfg=inv_cfg)
    res = verify_topk_with_fdtd(
        topk_npz=args.topk_npz,
        inverse_cfg=inv_cfg,
        fdtd_cfg=fdtd_cfg,
        out_dir=args.out_dir,
        k=args.k,
        layer_map=args.layer_map,
    )
    print(f"[OK] fdtd-verify done: out_dir={res.out_dir} fdtd_rggb={res.fdtd_rggb_path} report={res.ranking_report}")
    return 0


def cmd_integration_verify(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "integration-verify.summary.md"
    log = out_dir / "integration-verify.log.md"

    lines = []
    lines.append("# Integration Verify Summary")
    lines.append("")
    lines.append("This verify is dry-run only (no Lumerical lumapi).")
    lines.append("")

    # 1) Inverse dry-run (very small)
    cfg = InverseDesignConfig.from_yaml(args.config)
    cfg.opt.n_start = 4
    cfg.opt.n_steps = 2
    cfg.opt.topk = 2
    cfg.opt.robustness_samples = 2
    surrogate = MockSurrogate(n_channels=int(cfg.spectra.n_channels))
    run_inverse_opt(
        cfg=cfg,
        surrogate=surrogate,
        out_root="data/candidates",
        progress_dir="data/progress",
        device="cpu",
    )
    lines.append("- Inverse dry-run: PASS")

    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.write_text("PASS\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if args.cmd == "inverse":
        return cmd_inverse(args)
    if args.cmd == "rank":
        return cmd_rank(args)
    if args.cmd == "fdtd-verify":
        return cmd_fdtd_verify(args)
    if args.cmd == "integration-verify":
        return cmd_integration_verify(args)
    raise SystemExit(f"unknown cmd: {args.cmd}")
