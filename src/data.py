"""Analytic 2D sound field data: exterior point sources via the 2D Green's function.

The field in the measurement region [0,1]^2 is a superposition of three exterior
monopoles (direct sound + two reflection images), each radiating the free-field
2D Green's function G(r) = (i/4) * H0^(1)(k r), which satisfies the homogeneous
Helmholtz equation away from the source.
"""

import numpy as np
from scipy.special import hankel1

C = 343.0  # speed of sound [m/s]

# (x_s, y_s, amplitude)
SOURCES = np.array([
    [-0.5, 0.5, 1.0],   # direct
    [-0.5, -0.5, 0.7],  # image across y=0
    [-0.5, 1.5, 0.7],   # image across y=1
])

F_MIN, F_MAX, F_STEP = 50.0, 8000.0, 10.0


def frequencies() -> np.ndarray:
    """Frequency bins 50..8000 Hz in 10 Hz steps (796 bins)."""
    return np.arange(F_MIN, F_MAX + F_STEP / 2, F_STEP)


def wavenumber(freqs: np.ndarray) -> np.ndarray:
    return 2.0 * np.pi * np.asarray(freqs) / C


def pressure(points: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Complex pressure at `points` [N,2] for each frequency.

    Returns complex array of shape [F, N].
    """
    points = np.atleast_2d(points)
    k = wavenumber(freqs)  # [F]
    p = np.zeros((len(k), len(points)), dtype=np.complex128)
    for xs, ys, amp in SOURCES:
        r = np.hypot(points[:, 0] - xs, points[:, 1] - ys)  # [N]
        p += amp * (1j / 4.0) * hankel1(0, np.outer(k, r))
    return p


def grid(n: int) -> np.ndarray:
    """n x n grid covering [0,1]^2 inclusive, returned as [n*n, 2]."""
    g = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(g, g, indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], axis=1)


def mic_grid(n_per_side: int) -> np.ndarray:
    return grid(n_per_side)


def eval_grid() -> np.ndarray:
    return grid(33)
