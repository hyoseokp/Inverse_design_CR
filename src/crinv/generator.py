from __future__ import annotations

from pathlib import Path

import torch
import yaml
import torch.nn.functional as F

from .config import GeneratorConfig
from .ops import gaussian_blur_2d, sample_sigma_tau, ste_threshold, upsample_bicubic_2d, upsample_bilinear_2d

_NN_CACHE: dict[tuple[str, str], torch.nn.Module] = {}
_SOFT_CIRCLE_KERNEL_CACHE: dict[tuple[str, str, int, float, float], torch.Tensor] = {}


def _load_yaml_dict(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        return {}
    return obj


def _get_generator_predictor(*, paths_yaml: str, device: torch.device) -> torch.nn.Module:
    """Load the 16->128 predictor checkpoint once per (ckpt_path, device)."""
    d = _load_yaml_dict(paths_yaml)
    ckpt = str(d.get("generator_predictor_checkpoint") or "").strip()
    if not ckpt:
        raise FileNotFoundError(
            "generator_predictor_checkpoint missing in paths.yaml; "
            "set it in configs/paths.yaml (git-ignored)."
        )
    ckpt_path = Path(ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"generator_predictor_checkpoint not found: {ckpt_path}")

    key = (str(ckpt_path.resolve()), str(device))
    if key in _NN_CACHE:
        return _NN_CACHE[key]

    from .generator_predictor import ResUNet16to128

    model = ResUNet16to128(base=64, groups=8, circular=True, enforce_diag_sym=True, return_logits=True)
    state = torch.load(ckpt_path, map_location="cpu")
    # Notebook saves model.state_dict()
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    _NN_CACHE[key] = model
    return model


def generate_u_and_binary(
    s16: torch.Tensor,
    *,
    sigma: float,
    tau: float,
    struct_size: int = 128,
    use_ste: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rule-based generator pipeline.

    Inputs:
      s16: [B,16,16] in [0,1] (symmetric seed)
    Outputs:
      x_hard: [B,128,128] (0/1 float)
      x_ste:  [B,128,128] (forward equals x_hard; backward passes grad from u)
      u:      [B,128,128] (blurred continuous field)
    """
    if s16.ndim != 3:
        raise ValueError(f"s16 must have shape [B,16,16], got {tuple(s16.shape)}")
    u0 = upsample_bilinear_2d(s16, out_hw=struct_size)
    u = gaussian_blur_2d(u0, sigma=sigma)
    x_hard, x_ste = ste_threshold(u, tau)
    if not use_ste:
        x_ste = x_hard.detach()
    return x_hard, x_ste, u


def generate_u_and_binary_cfg(
    *,
    cfg,
    s16: torch.Tensor,
    sigma: float,
    tau: float,
    struct_size: int = 128,
    use_ste: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate (x_hard, x_ste, u) using configured backend.

    - rule: bilinear upsample + gaussian blur + threshold
    - rule_mfs: bicubic upsample + gaussian blur + soft threshold + differentiable-ish MFS (morphology)
    - nn:   learned 16->128 predictor + threshold (sigma is ignored)
    """
    backend = getattr(getattr(cfg, "generator", None), "backend", "rule")
    if backend == "rule":
        return generate_u_and_binary(s16, sigma=sigma, tau=tau, struct_size=struct_size, use_ste=use_ste)

    if backend == "rule_mfs":
        if s16.ndim != 3:
            raise ValueError(f"s16 must have shape [B,16,16], got {tuple(s16.shape)}")
        if int(struct_size) != 128:
            raise ValueError("rule_mfs currently supports struct_size=128 only")

        # Match dataset script: symmetric seed is created by copying upper triangle.
        sym_mode = str(getattr(getattr(cfg, "generator", None), "sym_mode", "avg") or "avg")
        if sym_mode == "upper_copy":
            upper = torch.triu(s16)
            s16 = upper + torch.triu(upper, diagonal=1).transpose(-1, -2)
        elif sym_mode == "avg":
            # seed_from_araw already returns avg-sym; keep as-is.
            pass
        else:
            raise ValueError("generator.sym_mode must be 'avg' or 'upper_copy'")

        # Match dataset script closer: bicubic upsample + gaussian blur.
        u0 = upsample_bicubic_2d(s16, out_hw=struct_size)
        u = gaussian_blur_2d(u0, sigma=sigma)

        # Soft threshold -> probability field in [0,1] (keeps gradients).
        temp = float(getattr(getattr(cfg, "generator", None), "threshold_temp", 0.05) or 0.05)
        temp = max(1.0e-4, temp)
        p = torch.sigmoid((u - float(tau)) / temp)

        # Approximate enforce_mfs_final() via pooling-based morphology with circular padding (wrap).
        #
        # apply_buffer_logic(binary, min_size, buffer=1):
        #   radius = min_size/2
        #   core = erode(binary, radius)
        #   over = dilate(core, radius+buffer)
        #   out  = erode(over, buffer)
        #
        # enforce_mfs_final:
        #   pad wrap, iterate:
        #     solid: img = apply_buffer_logic(img)
        #     void:  img = ~apply_buffer_logic(~img)
        #   crop, then diagonal upper-tri OR symmetry.
        r = int(getattr(getattr(cfg, "generator", None), "mfs_radius_px", 5) or 5)
        iters = int(getattr(getattr(cfg, "generator", None), "mfs_iters", 4) or 4)
        kshape = str(getattr(getattr(cfg, "generator", None), "mfs_kernel", "soft_circle") or "soft_circle")
        beta = float(getattr(getattr(cfg, "generator", None), "mfs_soft_beta", 20.0) or 20.0)
        dist_scale = float(getattr(getattr(cfg, "generator", None), "mfs_dist_scale", 0.25) or 0.25)
        r = max(0, r)
        iters = max(0, iters)
        beta = max(1.0e-3, beta)
        dist_scale = max(0.0, dist_scale)

        def _pad_circ(x: torch.Tensor, pad: int) -> torch.Tensor:
            if pad <= 0:
                return x
            x4 = x.unsqueeze(1)  # (B,1,H,W)
            x4 = F.pad(x4, (pad, pad, pad, pad), mode="circular")
            return x4.squeeze(1)

        def _soft_circle_dilate(x: torch.Tensor, rad: int) -> torch.Tensor:
            if rad <= 0:
                return x
            # Memory-efficient logsumexp approximation via conv2d:
            # logsumexp(beta*(x + bias)) = log( sum exp(beta*x) * exp(beta*bias) )
            # y = (1/beta) * log(conv2d(exp(beta*x), w)) where w = exp(beta*bias) masked to a circle.
            #
            # This avoids materializing unfold patches (k*k*H*W).
            k = 2 * rad + 1
            key = (str(x.device), str(x.dtype), int(rad), float(beta), float(dist_scale))
            w = _SOFT_CIRCLE_KERNEL_CACHE.get(key)
            if w is None:
                yy, xx = torch.meshgrid(
                    torch.arange(-rad, rad + 1, device=x.device, dtype=torch.float32),
                    torch.arange(-rad, rad + 1, device=x.device, dtype=torch.float32),
                    indexing="ij",
                )
                dist2 = (xx * xx) + (yy * yy)
                mask = (dist2 <= float(rad * rad)).to(dtype=torch.float32)
                bias = (-float(dist_scale)) * dist2
                w2 = torch.exp(torch.clamp(float(beta) * bias, min=-80.0, max=0.0)) * mask
                w = w2.to(device=x.device, dtype=x.dtype).view(1, 1, k, k)
                _SOFT_CIRCLE_KERNEL_CACHE[key] = w

            x4 = _pad_circ(x, rad).unsqueeze(1)  # (B,1,H+2r,W+2r)
            # exp(beta*x) can overflow if beta is huge; clamp is cheap and safe here.
            xb = torch.clamp(float(beta) * x4, min=-80.0, max=50.0)
            ex = torch.exp(xb)
            z = F.conv2d(ex, w, stride=1, padding=0)  # (B,1,H,W)
            y = (1.0 / float(beta)) * torch.log(z + 1.0e-12)
            return y[:, 0]

        def _dilate(x: torch.Tensor, rad: int) -> torch.Tensor:
            if rad <= 0:
                return x
            if kshape == "soft_circle":
                return _soft_circle_dilate(x, rad)
            if kshape == "square":
                k = 2 * rad + 1
                x4 = _pad_circ(x, rad).unsqueeze(1)
                y4 = F.max_pool2d(x4, kernel_size=k, stride=1)
                return y4.squeeze(1)
            raise ValueError("generator.mfs_kernel must be 'square' or 'soft_circle'")

        def _erode(x: torch.Tensor, rad: int) -> torch.Tensor:
            if rad <= 0:
                return x
            return -_dilate(-x, rad)

        def _apply_buffer_logic(x: torch.Tensor, *, rad: int, buffer: int = 1) -> torch.Tensor:
            core = _erode(x, rad)
            over = _dilate(core, rad + int(buffer))
            out = _erode(over, int(buffer))
            return out

        # Wrap pad like the dataset script to reduce edge artifacts.
        pad = max(0, 2 * (2 * r))  # approx: min_size*2 = (2r)*2 = 4r
        p = _pad_circ(p, pad)

        for _ in range(iters):
            prev = p
            p = _apply_buffer_logic(p, rad=r, buffer=1)
            p = 1.0 - _apply_buffer_logic(1.0 - p, rad=r, buffer=1)
            # Cheap convergence for soft field: if no change at float precision, stop early.
            if torch.allclose(p, prev, atol=0.0, rtol=0.0):
                break

        # Crop back.
        if pad > 0:
            p = p[:, pad:-pad, pad:-pad]

        # Diagonal symmetry like: img = triu(img) | triu(img,1).T
        upper = torch.triu(p)
        p = torch.maximum(upper, upper.transpose(-1, -2))

        # Dataset stores binary = ~enforce_mfs_final(...); apply the same inversion.
        p = 1.0 - p

        # Use p as the differentiable field for STE thresholding.
        x_hard, x_ste = ste_threshold(p, 0.5)
        if not use_ste:
            x_ste = x_hard.detach()
        return x_hard, x_ste, p

    if s16.ndim != 3:
        raise ValueError(f"s16 must have shape [B,16,16], got {tuple(s16.shape)}")
    if int(struct_size) != 128:
        raise ValueError("nn generator currently supports struct_size=128 only")

    device = s16.device
    paths_yaml = str(getattr(getattr(cfg, "paths", None), "paths_yaml", "configs/paths.yaml"))
    model = _get_generator_predictor(paths_yaml=paths_yaml, device=device)

    x = s16.to(dtype=torch.float32).unsqueeze(1)  # (B,1,16,16)
    logits = model(x)  # (B,1,128,128)
    u = torch.sigmoid(logits[:, 0])  # (B,128,128) in [0,1]
    x_hard, x_ste = ste_threshold(u, tau)
    if not use_ste:
        x_ste = x_hard.detach()
    return x_hard, x_ste, u


def sample_robust_params(
    cfg: GeneratorConfig,
    *,
    n: int,
    seed: int | None = None,
    device=None,
    dtype=None,
) -> list[tuple[float, float]]:
    samples = sample_sigma_tau(
        sigma_set=cfg.sigma_set,
        tau0=cfg.tau0,
        delta_tau=cfg.delta_tau,
        n=n,
        seed=seed,
        device=device,
        dtype=dtype,
    )
    return [(s.sigma, s.tau) for s in samples]
