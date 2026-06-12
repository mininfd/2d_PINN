"""Model tests: physics-exact architectures must reproduce the physics."""

import numpy as np
import torch

from src import data as D
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
    assert make_model("psource", n_src=4)(x.clamp(-1, 1)).shape == (5, 2)


def test_psource_matches_analytic_greens():
    """With the true source layout and modulation off, PointSourceNet must
    reproduce the scipy Hankel-function field."""
    torch.manual_seed(0)
    model = make_model("psource", n_src=3)
    with torch.no_grad():
        model.pos.copy_(torch.tensor(D.SOURCES[:, :2], dtype=torch.float32))
        model.amp.zero_()
        model.amp[:, 0] = torch.tensor(D.SOURCES[:, 2], dtype=torch.float32)
        model.mod[-1].weight.zero_()
        model.mod[-1].bias.zero_()

    pts = np.array([[0.2, 0.4], [0.9, 0.1]])
    f = np.array([500.0, 4000.0])
    expected = D.pressure(pts, f)

    k_all = D.wavenumber(D.frequencies())
    k_min, k_max = k_all.min(), k_all.max()
    kf = D.wavenumber(f)
    rows = [[2 * p[0] - 1, 2 * p[1] - 1,
             2 * (kf[i] - k_min) / (k_max - k_min) - 1]
            for i in range(len(f)) for p in pts]
    out = model(torch.tensor(rows, dtype=torch.float32)).detach().numpy()
    got = (out[:, 0] + 1j * out[:, 1]).reshape(2, 2)
    np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-4)


def test_psource_gradients_flow():
    model = make_model("psource", n_src=4)
    x = torch.rand(8, 3) * 2 - 1
    loss = model(x).pow(2).sum() + model.coef_penalty(x)
    loss.backward()
    assert torch.isfinite(model.pos.grad).all()
    assert torch.isfinite(model.amp.grad).all()
    assert model.amp.grad.abs().sum() > 0
