from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .artifacts import TopKPack, save_topk_npz, struct_from_seed_nominal
from .config import InverseDesignConfig
from .losses import robust_mc_loss
from .progress_logger import ProgressLogger
from .seed import seed_from_araw
from .surrogate_interface import ForwardSurrogate
from .losses import spectral_terms_from_rggb


@dataclass(frozen=True)
class InverseRunResult:
    run_dir: Path
    topk_path: Path


def _run_id() -> str:
    # UTC timestamp to keep it stable across environments.
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}"


def run_inverse_opt(
    *,
    cfg: InverseDesignConfig,
    surrogate: ForwardSurrogate,
    out_root: str | Path = "data/candidates",
    progress_dir: str | Path = "data/progress",
    device: str | torch.device = "cpu",
    progress_hook: Callable[[dict], None] | None = None,
    snapshot_hook: Callable[[dict], None] | None = None,
) -> InverseRunResult:
    """Multi-start inverse optimization with top-K export and file-based progress logging."""
    device = torch.device(device)

    B = int(cfg.opt.n_start)
    steps = int(cfg.opt.n_steps)
    topk = int(cfg.opt.topk)
    lr = float(cfg.opt.lr)
    chunk = int(getattr(cfg.opt, "chunk_size", 0) or 0)
    reduction = str(getattr(cfg.opt, "loss_reduction", "mean") or "mean")
    if reduction not in ("mean", "sum"):
        raise ValueError("opt.loss_reduction must be 'mean' or 'sum'")
    if chunk < 0:
        raise ValueError("opt.chunk_size must be >= 0")
    if chunk == 0:
        chunk = B

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / _run_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    plog = ProgressLogger(Path(progress_dir))
    plog.write_meta(
        {
            "ts_start": datetime.now(timezone.utc).isoformat(),
            "n_start": B,
            "n_steps": steps,
            "topk": topk,
            "robustness_samples": int(cfg.opt.robustness_samples),
            "log_every_n_steps": int(cfg.opt.log_every_n_steps),
            "seed_size": int(cfg.seed_size),
            "struct_size": int(cfg.struct_size),
            "random_seed": int(cfg.opt.random_seed),
            "device": str(device),
            "generator": {
                "backend": str(cfg.generator.backend),
                "sigma_set": list(cfg.generator.sigma_set),
                "tau0": float(cfg.generator.tau0),
                "delta_tau": float(cfg.generator.delta_tau),
            },
            "loss": {
                "w_purity": float(getattr(cfg.loss, "w_purity", 0.0)),
                "w_abs": float(getattr(cfg.loss, "w_abs", 0.0)),
                "w_gray": float(getattr(cfg.loss, "w_gray", 0.0)),
                "w_tv": float(getattr(cfg.loss, "w_tv", 0.0)),
                "w_fill": float(getattr(cfg.loss, "w_fill", 0.0)),
                "fill_min": float(getattr(cfg.loss, "fill_min", 0.0)),
                "fill_max": float(getattr(cfg.loss, "fill_max", 1.0)),
            },
        }
    )

    g = torch.Generator(device=device)
    g.manual_seed(int(cfg.opt.random_seed))
    a_raw = torch.randn((B, cfg.seed_size, cfg.seed_size), generator=g, device=device, dtype=torch.float32)
    a_raw = torch.nn.Parameter(a_raw)

    opt = torch.optim.Adam([a_raw], lr=lr)

    best_loss = torch.full((B,), float("inf"), device=device)
    best_seed = torch.zeros((B, cfg.seed_size, cfg.seed_size), device=device)
    cur_loss = torch.zeros((B,), device=device)
    cur_seed = torch.zeros((B, cfg.seed_size, cfg.seed_size), device=device)

    log_every = max(1, int(cfg.opt.log_every_n_steps))

    for step in range(steps):
        opt.zero_grad(set_to_none=True)

        # Micro-batch over candidates to keep memory bounded.
        sum_loss_total = 0.0
        sum_loss_spec = 0.0
        sum_loss_reg = 0.0
        sum_loss_fill = 0.0
        sum_D_R = 0.0
        sum_D_G = 0.0
        sum_D_B = 0.0
        sum_O_R = 0.0
        sum_O_G = 0.0
        sum_O_B = 0.0

        denom = float(B) if reduction == "mean" else 1.0

        for i0 in range(0, B, chunk):
            i1 = min(B, i0 + chunk)
            a_chunk = a_raw[i0:i1]
            terms = robust_mc_loss(
                cfg=cfg,
                a_raw=a_chunk,
                surrogate_predict_fn=surrogate.predict,
                step_seed=int(cfg.opt.random_seed) + int(step),
            )
            loss_vec = terms.loss_total  # (Bc,)
            # Global reduction across all B:
            # - mean: sum(loss_vec)/B
            # - sum:  sum(loss_vec)
            loss = loss_vec.sum() / denom
            loss.backward()

            with torch.no_grad():
                sum_loss_total += float(loss_vec.detach().sum().item())
                sum_loss_spec += float(terms.loss_spec.detach().sum().item())
                sum_loss_reg += float(terms.loss_reg.detach().sum().item())
                sum_loss_fill += float(terms.loss_fill.detach().sum().item())
                sum_D_R += float(terms.metrics.D_R.detach().sum().item())
                sum_D_G += float(terms.metrics.D_G.detach().sum().item())
                sum_D_B += float(terms.metrics.D_B.detach().sum().item())
                sum_O_R += float(terms.metrics.O_R.detach().sum().item())
                sum_O_G += float(terms.metrics.O_G.detach().sum().item())
                sum_O_B += float(terms.metrics.O_B.detach().sum().item())

                # Track per-candidate best for this chunk.
                curr = loss_vec.detach().view(-1)
                improved = curr < best_loss[i0:i1]
                best_loss[i0:i1] = torch.where(improved, curr, best_loss[i0:i1])
                s = seed_from_araw(a_chunk.detach())
                best_seed[i0:i1] = torch.where(improved[:, None, None], s, best_seed[i0:i1])
                # Track current snapshot (for diagnosing "stuck" best-of-run topk).
                cur_loss[i0:i1] = curr
                cur_seed[i0:i1] = s

        opt.step()
        # Reconstruct a scalar mean loss for logging/hook.
        loss_mean = sum_loss_total / float(B)
        loss_spec_mean = sum_loss_spec / float(B)
        loss_reg_mean = sum_loss_reg / float(B)
        loss_fill_mean = sum_loss_fill / float(B)

        with torch.no_grad():
            # Minimal dashboard contract logging (JSONL).
            plog.log_metrics(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "step": int(step),
                    "loss_total": float(loss_mean),
                    "loss_spec": float(loss_spec_mean),
                    "loss_reg": float(loss_reg_mean),
                    "loss_fill": float(loss_fill_mean),
                    "D_R": float(sum_D_R / float(B)),
                    "D_G": float(sum_D_G / float(B)),
                    "D_B": float(sum_D_B / float(B)),
                    "O_R": float(sum_O_R / float(B)),
                    "O_G": float(sum_O_G / float(B)),
                    "O_B": float(sum_O_B / float(B)),
                }
            )
            if progress_hook is not None:
                progress_hook(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "step": int(step),
                        "n_steps": int(steps),
                        "loss_total": float(loss_mean),
                        "loss_spec": float(loss_spec_mean),
                        "loss_reg": float(loss_reg_mean),
                        "loss_fill": float(loss_fill_mean),
                    }
                )

            # Periodic Top-K snapshot for the dashboard.
            if (step % log_every == 0) or (step == steps - 1):
                K = min(topk, B)
                idx_best = torch.topk(-best_loss, k=K).indices
                idx_cur = torch.topk(-cur_loss, k=K).indices

                seed16_best = best_seed[idx_best].detach().cpu()
                seed16_cur = cur_seed[idx_cur].detach().cpu()

                struct_best = struct_from_seed_nominal(seed16_best.to(device), cfg=cfg).detach().cpu()
                struct_cur = struct_from_seed_nominal(seed16_cur.to(device), cfg=cfg).detach().cpu()

                plog.write_topk_snapshot(
                    step=step,
                    seed16_topk=seed16_best.numpy().astype(np.float32),
                    struct128_topk=struct_best.numpy().astype(np.uint8),
                    metrics_topk={"best_loss": best_loss[idx_best].detach().cpu().numpy().astype(np.float32)},
                    prefix="topk",
                )
                plog.write_topk_snapshot(
                    step=step,
                    seed16_topk=seed16_cur.numpy().astype(np.float32),
                    struct128_topk=struct_cur.numpy().astype(np.uint8),
                    metrics_topk={"cur_loss": cur_loss[idx_cur].detach().cpu().numpy().astype(np.float32)},
                    prefix="topk_cur",
                )
                if snapshot_hook is not None:
                    snapshot_hook(
                        {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "step": int(step),
                            "n_steps": int(steps),
                            "topk_npz": str(Path(progress_dir) / f"topk_step-{int(step)}.npz"),
                        }
                    )

                # Extra debug: Top-1 purity matrix A for best vs current at this snapshot.
                try:
                    # Use nominal structure + single surrogate eval for diagnostics.
                    xb = struct_best[0].to(device=device, dtype=torch.float32).unsqueeze(0)
                    xc = struct_cur[0].to(device=device, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        tb = surrogate.predict(xb)  # (1,2,2,C)
                        tc = surrogate.predict(xc)
                        _mb, _rgbb, Ab = spectral_terms_from_rggb(
                            tb,
                            n_channels_target=int(cfg.spectra.n_channels),
                            band_ranges_nm=cfg.spectra.band_ranges_nm,
                            band_indices_30=getattr(cfg.spectra, "band_indices_30", None),
                        )
                        _mc, _rgbc, Ac = spectral_terms_from_rggb(
                            tc,
                            n_channels_target=int(cfg.spectra.n_channels),
                            band_ranges_nm=cfg.spectra.band_ranges_nm,
                            band_indices_30=getattr(cfg.spectra, "band_indices_30", None),
                        )

                    Ab_np = Ab[0].detach().cpu().numpy().astype(np.float32)
                    Ac_np = Ac[0].detach().cpu().numpy().astype(np.float32)
                    purity = {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "step": int(step),
                        "best": {
                            "A": Ab_np.tolist(),
                            "diag": [float(Ab_np[0, 0]), float(Ab_np[1, 1]), float(Ab_np[2, 2])],
                            "offdiag_mean": float((Ab_np.sum() - (Ab_np[0, 0] + Ab_np[1, 1] + Ab_np[2, 2])) / 6.0),
                            "loss0": float(best_loss[idx_best[0]].detach().cpu().item()),
                        },
                        "cur": {
                            "A": Ac_np.tolist(),
                            "diag": [float(Ac_np[0, 0]), float(Ac_np[1, 1]), float(Ac_np[2, 2])],
                            "offdiag_mean": float((Ac_np.sum() - (Ac_np[0, 0] + Ac_np[1, 1] + Ac_np[2, 2])) / 6.0),
                            "loss0": float(cur_loss[idx_cur[0]].detach().cpu().item()),
                        },
                    }
                    plog.write_json(name=f"purity_step-{int(step)}.json", payload=purity)
                except Exception:
                    # Diagnostics only; don't break optimization.
                    pass

    with torch.no_grad():
        # Select top-K among best_loss (lower is better).
        K = min(topk, B)
        idx = torch.topk(-best_loss, k=K).indices  # largest -loss == smallest loss
        seed16_topk = best_seed[idx].detach().cpu()

        struct_topk = struct_from_seed_nominal(seed16_topk.to(device), cfg=cfg).detach().cpu()
        pack = TopKPack(
            seed16=seed16_topk.numpy().astype(np.float32),
            struct128=struct_topk.numpy().astype(np.uint8),
            metrics={"best_loss": best_loss[idx].detach().cpu().numpy().astype(np.float32)},
        )
        topk_path = run_dir / "topk_pack.npz"
        save_topk_npz(topk_path, pack)

        # Final snapshot is already written by the periodic logger at step=steps-1.

    return InverseRunResult(run_dir=run_dir, topk_path=topk_path)
