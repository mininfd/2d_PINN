"""Network architectures: tanh MLP (baseline), SIREN, modified MLP (Wang et al. 2021).

All models map (x_norm, y_norm, k_norm) in [-1,1]^3 to (Re P, Im P) of the
per-frequency-normalized pressure.
"""

import math

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Plain tanh MLP."""

    def __init__(self, in_dim=3, width=256, depth=5, out_dim=2):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SineLayer(nn.Module):
    def __init__(self, in_dim, out_dim, omega0, is_first):
        super().__init__()
        self.omega0 = omega0
        self.linear = nn.Linear(in_dim, out_dim)
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_dim
            else:
                bound = math.sqrt(6.0 / in_dim) / omega0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega0 * self.linear(x))


class Siren(nn.Module):
    """SIREN (Sitzmann et al. 2020)."""

    def __init__(self, in_dim=3, width=256, depth=5, out_dim=2, omega0=30.0,
                 omega0_hidden=30.0):
        super().__init__()
        layers = [SineLayer(in_dim, width, omega0, is_first=True)]
        for _ in range(depth - 1):
            layers += [SineLayer(width, width, omega0_hidden, is_first=False)]
        final = nn.Linear(width, out_dim)
        with torch.no_grad():
            bound = math.sqrt(6.0 / width) / omega0_hidden
            final.weight.uniform_(-bound, bound)
        layers += [final]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ModifiedMLP(nn.Module):
    """Modified MLP with two encoder streams U, V mixed at every layer
    (Wang, Teng & Perdikaris 2021)."""

    def __init__(self, in_dim=3, width=256, depth=5, out_dim=2):
        super().__init__()
        self.act = nn.Tanh()
        self.enc_u = nn.Linear(in_dim, width)
        self.enc_v = nn.Linear(in_dim, width)
        self.input = nn.Linear(in_dim, width)
        self.hidden = nn.ModuleList(
            [nn.Linear(width, width) for _ in range(depth - 1)])
        self.out = nn.Linear(width, out_dim)

    def forward(self, x):
        u = self.act(self.enc_u(x))
        v = self.act(self.enc_v(x))
        h = self.act(self.input(x))
        for layer in self.hidden:
            z = self.act(layer(h))
            h = (1.0 - z) * u + z * v
        return self.out(h)


class HerglotzNet(nn.Module):
    """Physics-exact plane-wave (Herglotz) basis with k-dependent coefficients.

    P(x, k) = sum_j c_j(k) * exp(i k d_j . (x - x0)),  d_j on the unit circle.
    Every basis function satisfies the homogeneous Helmholtz equation exactly,
    so the model needs only the data loss; the PDE is built into the
    architecture. The complex coefficients c_j(k) come from a SIREN over k
    (the coefficients of exterior sources oscillate rapidly in k, which a
    tanh MLP cannot represent).

    Takes the same normalized input (x_norm, y_norm, k_norm) in [-1,1]^3 as
    the other models; k_min/k_max of the global frequency range are baked in
    to undo the normalization internally.
    """

    def __init__(self, n_dirs=256, width=256, depth=3, omega0=30.0,
                 omega0_hidden=30.0):
        super().__init__()
        from . import data as D
        k = D.wavenumber(D.frequencies())
        self.k_min, self.k_max = float(k.min()), float(k.max())
        self.n_dirs = n_dirs
        ang = torch.arange(n_dirs, dtype=torch.float32) * (2 * math.pi / n_dirs)
        self.register_buffer("dirs", torch.stack([torch.cos(ang),
                                                  torch.sin(ang)]))  # [2, J]
        self.coef = Siren(in_dim=1, width=width, depth=depth,
                          out_dim=2 * n_dirs, omega0=omega0,
                          omega0_hidden=omega0_hidden)

    def forward(self, x):
        xy = (x[:, :2] + 1.0) * 0.5 - 0.5  # physical coords relative to center
        k = (x[:, 2:3] + 1.0) * 0.5 * (self.k_max - self.k_min) + self.k_min
        phase = k * (xy @ self.dirs)       # [N, J]
        c = self.coef(x[:, 2:3])           # [N, 2J]
        a, b = c[:, :self.n_dirs], c[:, self.n_dirs:]
        cosp, sinp = torch.cos(phase), torch.sin(phase)
        norm = math.sqrt(self.n_dirs)
        re = (a * cosp - b * sinp).sum(-1) / norm
        im = (a * sinp + b * cosp).sum(-1) / norm
        return torch.stack([re, im], dim=-1)


def make_model(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name in ("tanh", "mlp", "baseline"):
        return MLP(**kwargs)
    if name == "siren":
        return Siren(**kwargs)
    if name in ("mmlp", "modified_mlp"):
        return ModifiedMLP(**kwargs)
    if name in ("herglotz", "hnet"):
        return HerglotzNet(**kwargs)
    raise ValueError(f"unknown model: {name}")
