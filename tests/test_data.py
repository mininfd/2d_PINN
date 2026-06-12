"""Unit tests: the analytic data generator must satisfy the physics it claims."""

import numpy as np
import pytest
from scipy.special import hankel1

from src import data as D


def fd_residual(point, f, h=1e-4):
    """Finite-difference Helmholtz residual (Delta P + k^2 P) at `point`."""
    k = 2 * np.pi * f / D.C
    offsets = np.array([[0, 0], [h, 0], [-h, 0], [0, h], [0, -h]])
    pts = point + offsets
    p = D.pressure(pts, np.array([f]))[0]
    lap = (p[1] + p[2] + p[3] + p[4] - 4 * p[0]) / h ** 2
    return lap + k ** 2 * p[0], k, p[0]


@pytest.mark.parametrize("f", [100.0, 1000.0, 8000.0])
def test_field_satisfies_helmholtz(f):
    res, k, p0 = fd_residual(np.array([0.3, 0.7]), f)
    rel = abs(res) / (k ** 2 * abs(p0))
    assert rel < 1e-4, f"relative Helmholtz residual {rel:.2e} at {f} Hz"


def test_superposition_matches_manual_sum():
    pts = np.array([[0.2, 0.4], [0.9, 0.1]])
    f = np.array([500.0, 4000.0])
    k = 2 * np.pi * f / D.C
    expected = np.zeros((2, 2), dtype=complex)
    for xs, ys, amp in D.SOURCES:
        r = np.hypot(pts[:, 0] - xs, pts[:, 1] - ys)
        expected += amp * (1j / 4) * hankel1(0, np.outer(k, r))
    np.testing.assert_allclose(D.pressure(pts, f), expected, rtol=1e-12)


def test_sources_are_outside_domain():
    for xs, ys, _ in D.SOURCES:
        assert not (0 <= xs <= 1 and 0 <= ys <= 1)


def test_frequencies():
    f = D.frequencies()
    assert f[0] == 50.0 and f[-1] == 8000.0
    assert np.allclose(np.diff(f), 10.0)
    assert len(f) == 796


def test_grids():
    assert D.mic_grid(8).shape == (64, 2)
    assert D.mic_grid(4).shape == (16, 2)
    g = D.eval_grid()
    assert g.shape == (1089, 2)
    assert g.min() == 0.0 and g.max() == 1.0


def test_pressure_shape_and_finite():
    p = D.pressure(D.mic_grid(4), D.frequencies()[:5])
    assert p.shape == (5, 16)
    assert np.all(np.isfinite(p))
