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


def make_model(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name in ("tanh", "mlp", "baseline"):
        return MLP(**kwargs)
    if name == "siren":
        return Siren(**kwargs)
    if name in ("mmlp", "modified_mlp"):
        return ModifiedMLP(**kwargs)
    raise ValueError(f"unknown model: {name}")
