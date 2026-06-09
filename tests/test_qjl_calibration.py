"""
Tests for the qjl_scale override and codebook.calibrate_qjl_scale.

Covers four gates from the research notes:
  1. Default qjl_scale matches the historical sqrt(π/2)/d.
  2. Overriding qjl_scale takes effect on dequantize / attention_score.
  3. Overriding centroids/boundaries replaces the codebook entry-for-entry.
  4. calibrate_qjl_scale on iid-Gaussian residuals matches the analytic
     MSE-min α = sqrt(2/π) / (1 + 2/π) ≈ 0.4875 (closed form, finite d).
  5. calibrate_qjl_scale on a retuned codebook reduces residual L2 error
     vs the default α.
"""

import math
import torch
import pytest

from turboquant.codebook import (
    get_codebook_tensors,
    compute_lloyd_max_codebook,
    calibrate_qjl_scale,
)
from turboquant.quantizer import TurboQuantMSE, TurboQuantProd
from turboquant.rotation import generate_qjl_matrix


DIM = 64
SEED = 7
DEVICE = torch.device("cpu")


def _make_prod(qjl_scale=None, centroids=None, boundaries=None):
    return TurboQuantProd(
        dim=DIM,
        bits=3,
        device=DEVICE,
        seed=SEED,
        qjl_scale=qjl_scale,
        centroids=centroids,
        boundaries=boundaries,
    )


def test_default_qjl_scale_is_sqrt_pi_2_over_dim():
    q = _make_prod()
    assert q.qjl_scale == pytest.approx(math.sqrt(math.pi / 2.0) / DIM)


def test_override_qjl_scale_takes_effect():
    q = _make_prod(qjl_scale=0.4875)
    assert q.qjl_scale == pytest.approx(0.4875 / DIM)


def test_override_qjl_scale_changes_dequantized_residual_path():
    torch.manual_seed(0)
    x = torch.randn(8, DIM)

    q_default = _make_prod()
    q_override = _make_prod(qjl_scale=0.4875)

    x_d = q_default.dequantize(q_default.quantize(x))
    x_o = q_override.dequantize(q_override.quantize(x))

    # Same MSE stage, different α → outputs must differ.
    assert not torch.allclose(x_d, x_o)


def test_override_centroids_replaces_codebook():
    centroids, boundaries = get_codebook_tensors(DIM, 2, DEVICE)
    centroids_perturbed = centroids * 1.5
    # Re-derive boundaries from the perturbed centroids (preserve sort).
    sorted_c, _ = centroids_perturbed.sort()
    interior = 0.5 * (sorted_c[:-1] + sorted_c[1:])
    bounds_new = torch.cat(
        [torch.tensor([-1.0]), interior, torch.tensor([1.0])]
    ).to(centroids.dtype)

    mse = TurboQuantMSE(
        dim=DIM, bits=2, device=DEVICE, seed=SEED,
        centroids=sorted_c, boundaries=bounds_new,
    )
    assert torch.allclose(mse.centroids, sorted_c)
    assert torch.allclose(mse.boundaries, bounds_new)


def test_override_centroids_requires_both():
    centroids, _ = get_codebook_tensors(DIM, 2, DEVICE)
    with pytest.raises(ValueError, match="provided together"):
        TurboQuantMSE(dim=DIM, bits=2, device=DEVICE, centroids=centroids)


def test_override_centroids_size_check():
    centroids, boundaries = get_codebook_tensors(DIM, 2, DEVICE)
    with pytest.raises(ValueError, match="centroids has"):
        TurboQuantMSE(
            dim=DIM, bits=2, device=DEVICE,
            centroids=centroids[:2], boundaries=boundaries,
        )


def test_calibrate_qjl_scale_mse_min_on_gaussian():
    """Closed-form α* ≈ sqrt(2/π) / (1 + 2/π) ≈ 0.4875 at finite d."""
    torch.manual_seed(1)
    n_samples = 2000
    r = torch.randn(n_samples, DIM)
    S = generate_qjl_matrix(DIM, DEVICE, dtype=torch.float32, seed=SEED + 1000)

    alpha = calibrate_qjl_scale(r, S)

    analytic = math.sqrt(2.0 / math.pi) / (1.0 + 2.0 / math.pi)  # ≈ 0.4875
    # Empirical agrees within ~2% on 2k samples at d=64.
    assert abs(alpha - analytic) < 0.02, (
        f"got {alpha:.4f}, analytic {analytic:.4f}"
    )


def test_calibrate_reduces_l2_error_vs_default_alpha():
    """
    On synthetic residuals shaped to violate the iid-Gaussian assumption
    (heavier tails), MSE-min α must reduce the aggregate L2 error of the
    *codec* reconstruction r̂ = (α/d)·‖r‖·Sᵀ·sign(S·r). This is the exact
    objective calibrate_qjl_scale minimizes; the test guards against typos
    in the closed form.
    """
    torch.manual_seed(2)
    n_samples = 1000
    # Laplace-shaped residuals (heavier tails than Gaussian).
    u = torch.rand(n_samples, DIM) - 0.5
    r = -torch.sign(u) * torch.log1p(-2 * u.abs())  # Laplace(0, 1)
    r = r * 0.1  # match typical residual magnitude

    S = generate_qjl_matrix(DIM, DEVICE, dtype=torch.float32, seed=SEED + 1000)
    proj = r @ S.T
    signs = torch.where(proj > 0, 1.0, -1.0)
    g = signs @ S
    r_norm = r.norm(dim=-1, keepdim=True)

    alpha_default = math.sqrt(math.pi / 2.0)
    alpha_cal = calibrate_qjl_scale(r, S)

    err_default = (r - (alpha_default / DIM) * r_norm * g).pow(2).sum(dim=-1)
    err_cal = (r - (alpha_cal / DIM) * r_norm * g).pow(2).sum(dim=-1)

    # MSE-min by construction — must beat the default on the aggregate.
    assert err_cal.sum() < err_default.sum()
