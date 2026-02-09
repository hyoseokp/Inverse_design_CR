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
) -> InverseRunResult:
    """Multi-start inverse optimization with top-K export and file-based progress logging."""
    device = torch.device(device)

    B = int(cfg.opt.n_start)
    steps = int(cfg.opt.n_steps)
    topk = int(cfg.opt.topk)
    lr = float(cfg.opt.lr)

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
        }
    )

    g = torch.Generator(device=device)
    g.manual_seed(int(cfg.opt.random_seed))
    a_raw = torch.randn((B, cfg.seed_size, cfg.seed_size), generator=g, device=device, dtype=torch.float32)
    a_raw = torch.nn.Parameter(a_raw)

    opt = torch.optim.Adam([a_raw], lr=lr)

    best_loss = torch.full((B,), float("inf"), device=device)
    best_seed = torch.zeros((B, cfg.seed_size, cfg.seed_size), device=device)

    log_every = max(1, int(cfg.opt.log_every_n_steps))

    for step in range(steps):
        opt.zero_grad(set_to_none=True)

        terms = robust_mc_loss(
            cfg=cfg,
            a_raw=a_raw,
            surrogate_predict_fn=surrogate.predict,
            step_seed=int(cfg.opt.random_seed) + int(step),
        )
        # Per-candidate loss vector -> optimize mean, track per-candidate best.
        loss_vec = terms.loss_total
        loss = loss_vec.mean()
        loss.backward()
        opt.step()

        with torch.no_grad():
            # Track per-candidate best by current loss estimate.
            # (For dry-run/mock surrogate, this is sufficient; later can be replaced with
            # a more robust selection metric.)
            curr = loss_vec.detach()
            if curr.shape != (B,):
                curr = curr.view(-1)[:B]
            improved = curr < best_loss
            best_loss = torch.where(improved, curr, best_loss)
            s = seed_from_araw(a_raw.detach())
            best_seed = torch.where(improved[:, None, None], s, best_seed)

            # Minimal dashboard contract logging (JSONL).
            plog.log_metrics(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "step": int(step),
                    "loss_total": float(loss.detach().item()),
                    "loss_spec": float(terms.loss_spec.detach().mean().item()),
                    "loss_reg": float(terms.loss_reg.detach().mean().item()),
                    "loss_purity": float(terms.loss_purity.detach().mean().item()),
                    "D_R": float(terms.metrics.D_R.detach().mean().item()),
                    "D_G": float(terms.metrics.D_G.detach().mean().item()),
                    "D_B": float(terms.metrics.D_B.detach().mean().item()),
                    "O_R": float(terms.metrics.O_R.detach().mean().item()),
                    "O_G": float(terms.metrics.O_G.detach().mean().item()),
                    "O_B": float(terms.metrics.O_B.detach().mean().item()),
                }
            )
            if progress_hook is not None:
                progress_hook(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "step": int(step),
                        "n_steps": int(steps),
                        "loss_total": float(loss.detach().item()),
                        "loss_spec": float(terms.loss_spec.detach().mean().item()),
                        "loss_reg": float(terms.loss_reg.detach().mean().item()),
                        "loss_purity": float(terms.loss_purity.detach().mean().item()),
                    }
                )

            # Periodic Top-K snapshot for the dashboard.
            if (step % log_every == 0) or (step == steps - 1):
                K = min(topk, B)
                idx = torch.topk(-best_loss, k=K).indices
                seed16_topk = best_seed[idx].detach().cpu()
                struct_topk = struct_from_seed_nominal(seed16_topk.to(device), cfg=cfg).detach().cpu()
                plog.write_topk_snapshot(
                    step=step,
                    seed16_topk=seed16_topk.numpy().astype(np.float32),
                    struct128_topk=struct_topk.numpy().astype(np.uint8),
                    metrics_topk={"best_loss": best_loss[idx].detach().cpu().numpy().astype(np.float32)},
                )

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
