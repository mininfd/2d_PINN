"""Model tests: the Herglotz basis must satisfy the Helmholtz PDE exactly."""

import torch

from src.models import make_model
from src.train import helmholtz_residual


def test_herglotz_satisfies_helmholtz_exactly():
    torch.manual_seed(0)
    model = make_model("herglotz", n_dirs=16, width=32, depth=2).double()
    k_min, k_max = model.k_min, model.k_max
    xy = torch.rand(8, 2, dtype=torch.float64)
    kp = torch.rand(8, 1, dtype=torch.float64) * (k_max - k_min) + k_min
    # tolerance: cancellation of terms of size ~k^2 leaves float64 noise ~1e-8
    r = helmholtz_residual(model, xy, kp, k_min, k_max)
    assert r.abs().max().item() < 1e-6


def test_all_models_forward_shape():
    x = torch.randn(5, 3)
    for name in ("tanh", "siren", "mmlp"):
        assert make_model(name, width=16, depth=2)(x).shape == (5, 2)
    assert make_model("herglotz", n_dirs=8, width=16, depth=2)(x).shape == (5, 2)
