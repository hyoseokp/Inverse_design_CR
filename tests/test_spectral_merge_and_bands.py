from __future__ import annotations

import torch

from crinv.spectral import band_averages_rgb, merge_rggb_to_rgb, wavelength_grid_nm


def test_merge_rggb_to_rgb_shapes():
    t = torch.zeros(2, 2, 30)
    rgb = merge_rggb_to_rgb(t)
    assert rgb.shape == (3, 30)

    tb = torch.zeros(4, 2, 2, 30)
    rgbb = merge_rggb_to_rgb(tb)
    assert rgbb.shape == (4, 3, 30)


def test_merge_rggb_to_rgb_values():
    # R=1, G1=2, G2=4, B=8
    t = torch.zeros(2, 2, 3)
    t[0, 0, :] = 1.0
    t[0, 1, :] = 2.0
    t[1, 0, :] = 4.0
    t[1, 1, :] = 8.0
    rgb = merge_rggb_to_rgb(t)
    assert torch.allclose(rgb[0], torch.ones(3) * 1.0)
    assert torch.allclose(rgb[1], torch.ones(3) * 3.0)
    assert torch.allclose(rgb[2], torch.ones(3) * 8.0)


def test_band_averages_deterministic():
    # Create RGB spectra with constant values per color; band averages should match constants.
    C = 30
    wl = wavelength_grid_nm(C)
    rgb = torch.stack(
        [
            torch.full((C,), 0.1),  # R
            torch.full((C,), 0.2),  # G
            torch.full((C,), 0.3),  # B
        ],
        dim=0,
    )
    bands = {"B": (400.0, 500.0), "G": (500.0, 600.0), "R": (600.0, 700.0)}
    m = band_averages_rgb(rgb, wl_nm=wl, band_ranges_nm=bands)
    assert torch.isfinite(m.D_R) and torch.isfinite(m.O_R)
    assert torch.allclose(m.D_R, torch.tensor(0.1), atol=1e-6)
    assert torch.allclose(m.D_G, torch.tensor(0.2), atol=1e-6)
    assert torch.allclose(m.D_B, torch.tensor(0.3), atol=1e-6)
    # O_b is mean of other two constants.
    assert torch.allclose(m.O_R, torch.tensor(0.25), atol=1e-6)
    assert torch.allclose(m.O_G, torch.tensor(0.2), atol=1e-6)  # mean(0.1,0.3)
    assert torch.allclose(m.O_B, torch.tensor(0.15), atol=1e-6)

