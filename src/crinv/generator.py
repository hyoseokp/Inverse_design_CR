from __future__ import annotations

from pathlib import Path

import torch
import yaml

from .config import GeneratorConfig
from .ops import gaussian_blur_2d, sample_sigma_tau, ste_threshold, upsample_bilinear_2d

_NN_CACHE: dict[tuple[str, str], torch.nn.Module] = {}


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
    - nn:   learned 16->128 predictor + threshold (sigma is ignored)
    """
    backend = getattr(getattr(cfg, "generator", None), "backend", "rule")
    if backend != "nn":
        return generate_u_and_binary(s16, sigma=sigma, tau=tau, struct_size=struct_size, use_ste=use_ste)

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
