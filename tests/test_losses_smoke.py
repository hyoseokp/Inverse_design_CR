from __future__ import annotations

import torch

from crinv.config import InverseDesignConfig
from crinv.losses import compute_loss_from_surrogate, robust_mc_loss


def _mock_predict(x_binary: torch.Tensor) -> torch.Tensor:
    # x_binary: [B,128,128]
    # Produce stable RGGB spectrum: [B,2,2,30] where values depend on mean(x).
    B = x_binary.shape[0]
    m = x_binary.mean(dim=(-1, -2), keepdim=False)  # [B]
    C = 30
    out = torch.zeros((B, 2, 2, C), dtype=x_binary.dtype, device=x_binary.device)
    out[:, 0, 0, :] = (0.2 + 0.1 * m).unsqueeze(-1)  # R
    out[:, 0, 1, :] = (0.3 + 0.1 * m).unsqueeze(-1)  # G1
    out[:, 1, 0, :] = (0.4 + 0.1 * m).unsqueeze(-1)  # G2
    out[:, 1, 1, :] = (0.5 + 0.1 * m).unsqueeze(-1)  # B
    return out


def test_compute_loss_no_nan():
    cfg = InverseDesignConfig()
    B = 2
    t = torch.rand(B, 2, 2, 30)
    u = torch.rand(B, 128, 128)
    terms = compute_loss_from_surrogate(cfg=cfg, t_rggb=t, u=u)
    assert torch.isfinite(terms.loss_total).all()
    assert torch.isfinite(terms.loss_spec).all()
    assert torch.isfinite(terms.loss_reg).all()


def test_robust_mc_loss_smoke_and_grad():
    cfg = InverseDesignConfig()
    cfg.opt.n_start = 3
    cfg.opt.robustness_samples = 3
    a_raw = torch.randn(cfg.opt.n_start, 16, 16, requires_grad=True)
    terms = robust_mc_loss(cfg=cfg, a_raw=a_raw, surrogate_predict_fn=_mock_predict, step_seed=123)
    assert torch.isfinite(terms.loss_total).all()
    terms.loss_total.mean().backward()
    assert a_raw.grad is not None
    assert torch.isfinite(a_raw.grad).all()
