from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .artifacts import struct_from_seed_nominal
from .config import InverseDesignConfig
from .losses import robust_mc_loss, spectral_terms_from_rggb
from .progress_logger import ProgressLogger
from .seed import seed_from_araw
from .surrogate_interface import ForwardSurrogate


@dataclass(frozen=True)
class GARunResult:
    run_dir: Path
    topk_path: Path


def _run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}"


def run_inverse_ga(
    *,
    cfg: InverseDesignConfig,
    surrogate: ForwardSurrogate,
    out_root: str | Path = "data/candidates",
    progress_dir: str | Path = "data/progress",
    device: str | torch.device = "cpu",
    progress_hook: Callable[[dict], None] | None = None,
    snapshot_hook: Callable[[dict], None] | None = None,
) -> GARunResult:
    """Genetic algorithm optimization that writes the same progress artifacts as the Adam engine.

    Dashboard contract (same as Adam engine):
    - progress_dir/run_meta.json
    - progress_dir/metrics.jsonl (step field)
    - progress_dir/topk_step-<step>.npz (best-so-far)
    - progress_dir/topk_cur_step-<step>.npz (current generation)
    """
    device = torch.device(device)

    B = int(cfg.opt.n_start)  # population
    steps = int(cfg.opt.n_steps)  # generations
    topk = int(cfg.opt.topk)
    chunk = int(getattr(cfg.opt, "chunk_size", 0) or 0)
    if chunk <= 0:
        chunk = B

    elite = max(0, int(getattr(cfg.opt, "ga_elite", 8)))
    t_k = max(2, int(getattr(cfg.opt, "ga_tournament_k", 5)))
    alpha = float(getattr(cfg.opt, "ga_crossover_alpha", 0.5))
    mut_sigma = float(getattr(cfg.opt, "ga_mutation_sigma", 0.15))
    mut_p = float(getattr(cfg.opt, "ga_mutation_p", 0.2))

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / _run_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    plog = ProgressLogger(Path(progress_dir))
    plog.write_meta(
        {
            "ts_start": datetime.now(timezone.utc).isoformat(),
            "engine": "ga",
            "n_start": B,
            "n_steps": steps,
            "topk": topk,
            "robustness_samples": int(cfg.opt.robustness_samples),
            "log_every_n_steps": int(cfg.opt.log_every_n_steps),
            "seed_size": int(cfg.seed_size),
            "struct_size": int(cfg.struct_size),
            "random_seed": int(cfg.opt.random_seed),
            "device": str(device),
            "ga": {
                "elite": elite,
                "tournament_k": t_k,
                "crossover_alpha": alpha,
                "mutation_sigma": mut_sigma,
                "mutation_p": mut_p,
            },
            "generator": {
                "backend": str(cfg.generator.backend),
                "sigma_set": list(cfg.generator.sigma_set),
                "tau0": float(cfg.generator.tau0),
                "delta_tau": float(cfg.generator.delta_tau),
            },
            "spectra": {
                "n_channels": int(cfg.spectra.n_channels),
                "rgb_weights": dict(getattr(cfg.spectra, "rgb_weights", {"R": 1.0, "G": 1.0, "B": 1.0})),
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
    pop = torch.randn((B, cfg.seed_size, cfg.seed_size), generator=g, device=device, dtype=torch.float32)

    # Best-so-far pool (topk)
    bestK_araw = None
    bestK_loss = None

    log_every = max(1, int(cfg.opt.log_every_n_steps))

    def _eval_population(a_raw: torch.Tensor, *, step_seed: int) -> dict[str, torch.Tensor]:
        loss_total = torch.empty((a_raw.shape[0],), device=device, dtype=torch.float32)
        loss_spec = torch.empty_like(loss_total)
        loss_reg = torch.empty_like(loss_total)
        loss_fill = torch.empty_like(loss_total)
        loss_purity = torch.empty_like(loss_total)
        D_R = torch.empty_like(loss_total)
        D_G = torch.empty_like(loss_total)
        D_B = torch.empty_like(loss_total)
        O_R = torch.empty_like(loss_total)
        O_G = torch.empty_like(loss_total)
        O_B = torch.empty_like(loss_total)

        with torch.no_grad():
            for i0 in range(0, a_raw.shape[0], chunk):
                i1 = min(a_raw.shape[0], i0 + chunk)
                terms = robust_mc_loss(
                    cfg=cfg,
                    a_raw=a_raw[i0:i1],
                    surrogate_predict_fn=surrogate.predict,
                    step_seed=step_seed,
                )
                loss_total[i0:i1] = terms.loss_total.detach()
                loss_spec[i0:i1] = terms.loss_spec.detach()
                loss_reg[i0:i1] = terms.loss_reg.detach()
                loss_fill[i0:i1] = terms.loss_fill.detach()
                loss_purity[i0:i1] = terms.loss_purity.detach()
                D_R[i0:i1] = terms.metrics.D_R.detach()
                D_G[i0:i1] = terms.metrics.D_G.detach()
                D_B[i0:i1] = terms.metrics.D_B.detach()
                O_R[i0:i1] = terms.metrics.O_R.detach()
                O_G[i0:i1] = terms.metrics.O_G.detach()
                O_B[i0:i1] = terms.metrics.O_B.detach()

        return {
            "loss_total": loss_total,
            "loss_spec": loss_spec,
            "loss_reg": loss_reg,
            "loss_fill": loss_fill,
            "loss_purity": loss_purity,
            "D_R": D_R,
            "D_G": D_G,
            "D_B": D_B,
            "O_R": O_R,
            "O_G": O_G,
            "O_B": O_B,
        }

    def _tournament(losses: torch.Tensor) -> int:
        idx = torch.randint(0, int(losses.numel()), (t_k,), device=losses.device)
        best = idx[losses[idx].argmin()]
        return int(best.item())

    def _write_purity_debug(step: int, seed16_best: torch.Tensor, seed16_cur: torch.Tensor, loss_best0: float, loss_cur0: float) -> None:
        try:
            xb = struct_from_seed_nominal(seed16_best[:1].to(device), cfg=cfg)
            xc = struct_from_seed_nominal(seed16_cur[:1].to(device), cfg=cfg)
            with torch.no_grad():
                tb = surrogate.predict(xb)
                tc = surrogate.predict(xc)
                _mb, _rgbb, Ab = spectral_terms_from_rggb(
                    tb,
                    n_channels_target=int(cfg.spectra.n_channels),
                    band_ranges_nm=cfg.spectra.band_ranges_nm,
                    band_indices_30=getattr(cfg.spectra, "band_indices_30", None),
                    rgb_weights=getattr(cfg.spectra, "rgb_weights", None),
                )
                _mc, _rgbc, Ac = spectral_terms_from_rggb(
                    tc,
                    n_channels_target=int(cfg.spectra.n_channels),
                    band_ranges_nm=cfg.spectra.band_ranges_nm,
                    band_indices_30=getattr(cfg.spectra, "band_indices_30", None),
                    rgb_weights=getattr(cfg.spectra, "rgb_weights", None),
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
                    "loss0": float(loss_best0),
                },
                "cur": {
                    "A": Ac_np.tolist(),
                    "diag": [float(Ac_np[0, 0]), float(Ac_np[1, 1]), float(Ac_np[2, 2])],
                    "offdiag_mean": float((Ac_np.sum() - (Ac_np[0, 0] + Ac_np[1, 1] + Ac_np[2, 2])) / 6.0),
                    "loss0": float(loss_cur0),
                },
            }
            plog.write_json(name=f"purity_step-{int(step)}.json", payload=purity)
        except Exception:
            pass

    for step in range(steps):
        ev = _eval_population(pop, step_seed=int(cfg.opt.random_seed) + int(step))
        losses = ev["loss_total"]

        order = torch.argsort(losses)
        pop_sorted = pop[order]
        loss_sorted = losses[order]

        # Update best-so-far topk pool.
        if bestK_araw is None:
            bestK_araw = pop_sorted[: min(topk, B)].detach().clone()
            bestK_loss = loss_sorted[: min(topk, B)].detach().clone()
        else:
            cand_araw = torch.cat([bestK_araw, pop], dim=0)
            cand_loss = torch.cat([bestK_loss, losses], dim=0)
            o2 = torch.argsort(cand_loss)[: min(topk, cand_loss.numel())]
            bestK_araw = cand_araw[o2].detach().clone()
            bestK_loss = cand_loss[o2].detach().clone()

        # Metrics log (mean over population)
        with torch.no_grad():
            rec = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "step": int(step),
                "loss_total": float(loss_sorted.mean().item()),
                "loss_spec": float(ev["loss_spec"].mean().item()),
                "loss_reg": float(ev["loss_reg"].mean().item()),
                "loss_fill": float(ev["loss_fill"].mean().item()),
                "loss_purity": float(ev["loss_purity"].mean().item()),
                "D_R": float(ev["D_R"].mean().item()),
                "D_G": float(ev["D_G"].mean().item()),
                "D_B": float(ev["D_B"].mean().item()),
                "O_R": float(ev["O_R"].mean().item()),
                "O_G": float(ev["O_G"].mean().item()),
                "O_B": float(ev["O_B"].mean().item()),
            }
            plog.log_metrics(rec)
            if progress_hook is not None:
                progress_hook({"step": int(step), "n_steps": int(steps), "loss_total": float(rec["loss_total"])})

        if (step % log_every == 0) or (step == steps - 1):
            K = min(topk, B)
            # Current generation topk
            idx_cur = order[:K]
            a_cur = pop[idx_cur].detach().cpu()
            seed16_cur = seed_from_araw(a_cur.to(device)).detach().cpu()
            struct_cur = struct_from_seed_nominal(seed16_cur.to(device), cfg=cfg).detach().cpu()
            plog.write_topk_snapshot(
                step=step,
                seed16_topk=seed16_cur.numpy().astype(np.float32),
                struct128_topk=struct_cur.numpy().astype(np.uint8),
                metrics_topk={"cur_loss": loss_sorted[:K].detach().cpu().numpy().astype(np.float32)},
                prefix="topk_cur",
            )

            # Best-so-far topk
            assert bestK_araw is not None and bestK_loss is not None
            seed16_best = seed_from_araw(bestK_araw.to(device)).detach().cpu()
            struct_best = struct_from_seed_nominal(seed16_best.to(device), cfg=cfg).detach().cpu()
            out_npz = plog.write_topk_snapshot(
                step=step,
                seed16_topk=seed16_best.numpy().astype(np.float32),
                struct128_topk=struct_best.numpy().astype(np.uint8),
                metrics_topk={"best_loss": bestK_loss.detach().cpu().numpy().astype(np.float32)},
                prefix="topk",
            )
            if snapshot_hook is not None:
                snapshot_hook({"step": int(step), "n_steps": int(steps), "topk_npz": str(out_npz)})

            # Debug purity (top-1)
            _write_purity_debug(
                step=step,
                seed16_best=seed16_best,
                seed16_cur=seed16_cur,
                loss_best0=float(bestK_loss[0].item()),
                loss_cur0=float(loss_sorted[0].item()),
            )

        # Next generation (skip on last)
        if step == steps - 1:
            break

        # Elitism
        next_pop = [pop_sorted[: min(elite, B)].detach().clone()] if elite > 0 else []
        while sum(x.shape[0] for x in next_pop) < B:
            i1 = _tournament(losses)
            i2 = _tournament(losses)
            p1 = pop[i1 : i1 + 1]
            p2 = pop[i2 : i2 + 1]
            child = alpha * p1 + (1.0 - alpha) * p2
            if float(torch.rand((), device=device).item()) < mut_p:
                child = child + mut_sigma * torch.randn_like(child)
            next_pop.append(child)
        pop = torch.cat(next_pop, dim=0)[:B]

    # Final topk pack (best-so-far)
    assert bestK_araw is not None and bestK_loss is not None
    seed16_topk = seed_from_araw(bestK_araw.to(device)).detach().cpu()
    struct_topk = struct_from_seed_nominal(seed16_topk.to(device), cfg=cfg).detach().cpu()
    topk_path = run_dir / "topk_pack.npz"
    np.savez_compressed(
        topk_path,
        seed16_topk=seed16_topk.numpy().astype(np.float32),
        struct128_topk=struct_topk.numpy().astype(np.uint8),
        metric_best_loss=bestK_loss.detach().cpu().numpy().astype(np.float32),
    )
    return GARunResult(run_dir=run_dir, topk_path=topk_path)
