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

    def coef_penalty(self, x):
        """Mean squared Herglotz coefficient (Tikhonov / minimum-norm reg).

        The minimum-l2-norm plane-wave fit is equivalent to kernel
        interpolation with the J0 Bessel kernel, the standard prior for
        sound-field reconstruction from sparse mics.
        """
        c = self.coef(x[:, 2:3])
        return (c ** 2).mean()


class KScaledSiren(nn.Module):
    """SIREN over (k·x, k·y, k_norm): first-layer atoms are random plane
    waves sin(W·kx), so the network's intrinsic curvature tracks k² across
    the whole band and the k²-normalized Helmholtz residual stays O(1) —
    the failure mode of the plain SIREN PINN (huge residual at low k from
    k-independent curvature) is removed by construction.
    """

    def __init__(self, width=256, depth=5, out_dim=2, omega0=3.0,
                 omega0_hidden=1.0):
        super().__init__()
        from . import data as D
        k = D.wavenumber(D.frequencies())
        self.k_min, self.k_max = float(k.min()), float(k.max())
        self.net = Siren(in_dim=3, width=width, depth=depth, out_dim=out_dim,
                         omega0=omega0, omega0_hidden=omega0_hidden)

    def forward(self, x):
        xy = (x[:, :2] + 1.0) * 0.5 - 0.5  # physical coords around center
        k = (x[:, 2:3] + 1.0) * 0.5 * (self.k_max - self.k_min) + self.k_min
        # with omega0=3 and first-layer |W| <= 1/3, atoms sin(3·W·k·x) reach
        # spatial frequencies up to k — exactly the Helmholtz wave scale
        feat = torch.cat([k * xy, x[:, 2:3]], dim=1)
        return self.net(feat)


class KPlaneMMLP(nn.Module):
    """Sine plane-wave feature layer on k-scaled coordinates + ModifiedMLP body.

    Combines the two ingredients that individually helped PDE-loss training:
    KScaledSiren's curvature matching (first-layer atoms sin(W·kx) have
    curvature ∝ k², keeping the k²-normalized Helmholtz residual O(1) at all
    frequencies) and the ModifiedMLP body (the only PDE-loss network that
    partially learned with plain inputs).
    """

    def __init__(self, n_feat=128, width=256, depth=4, omega0=3.0):
        super().__init__()
        from . import data as D
        k = D.wavenumber(D.frequencies())
        self.k_min, self.k_max = float(k.min()), float(k.max())
        self.feat = SineLayer(2, n_feat, omega0, is_first=True)
        self.body = ModifiedMLP(in_dim=n_feat + 1, width=width, depth=depth)

    def forward(self, x):
        xy = (x[:, :2] + 1.0) * 0.5 - 0.5  # physical coords around center
        k = (x[:, 2:3] + 1.0) * 0.5 * (self.k_max - self.k_min) + self.k_min
        f = self.feat(k * xy)
        return self.body(torch.cat([f, x[:, 2:3]], dim=1))


class _BesselJ0(torch.autograd.Function):
    """torch.special.bessel_j0 with the (missing) backward J0'(x) = -J1(x)."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.special.bessel_j0(x)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return -g * torch.special.bessel_j1(x)


class _BesselY0(torch.autograd.Function):
    """torch.special.bessel_y0 with the (missing) backward Y0'(x) = -Y1(x)."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.special.bessel_y0(x)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return -g * torch.special.bessel_y1(x)


class PointSourceNet(nn.Module):
    """Equivalent-source model: M monopoles with learnable exterior positions.

    P(x, k) = m(k) * sum_j a_j * (i/4) H0^(1)(k |x - q_j|), with learnable
    positions q_j (kept outside the domain by a penalty), frequency-flat
    complex amplitudes a_j, and one shared smooth complex modulation m(k)
    (a small MLP) that absorbs the per-frequency data normalization. Each
    monopole satisfies the Helmholtz equation exactly away from its source,
    so only the data loss is needed. The frequency-flat amplitude is the
    broadband prior that makes very sparse arrays (16 mics) identifiable:
    each mic's response over k encodes the distances to the sources.
    """

    def __init__(self, n_src=16, ring_radius=1.3, mod_width=64, margin=0.15):
        super().__init__()
        self.margin = margin
        ang = torch.arange(n_src, dtype=torch.float32) * (2 * math.pi / n_src)
        pos = 0.5 + ring_radius * torch.stack([torch.cos(ang), torch.sin(ang)],
                                              dim=1)
        self.pos = nn.Parameter(pos)                      # [M, 2]
        self.amp = nn.Parameter(0.1 * torch.randn(n_src, 2))  # [M, 2] Re/Im
        self.mod = nn.Sequential(
            nn.Linear(1, mod_width), nn.Tanh(),
            nn.Linear(mod_width, mod_width), nn.Tanh(),
            nn.Linear(mod_width, 2))
        from . import data as D
        k = D.wavenumber(D.frequencies())
        self.k_min, self.k_max = float(k.min()), float(k.max())

    def forward(self, x):
        xy = (x[:, :2] + 1.0) * 0.5                       # physical [0,1]^2
        k = (x[:, 2:3] + 1.0) * 0.5 * (self.k_max - self.k_min) + self.k_min
        r = torch.cdist(xy, self.pos)                     # [N, M]
        kr = k * r
        # G = (i/4) H0 = (-Y0 + i J0) / 4
        g_re = -_BesselY0.apply(kr) / 4.0
        g_im = _BesselJ0.apply(kr) / 4.0
        a_re, a_im = self.amp[:, 0], self.amp[:, 1]       # [M]
        p_re = (a_re * g_re - a_im * g_im).sum(-1)        # [N]
        p_im = (a_re * g_im + a_im * g_re).sum(-1)
        m = self.mod(x[:, 2:3])                           # [N, 2]
        m_re, m_im = 1.0 + m[:, 0], m[:, 1]
        return torch.stack([m_re * p_re - m_im * p_im,
                            m_re * p_im + m_im * p_re], dim=-1)

    def coef_penalty(self, x):
        """Keeps sources at least `margin` outside the unit square."""
        clamped = self.pos.clamp(0.0, 1.0)
        d = (self.pos - clamped).norm(dim=1)
        return 100.0 * torch.relu(self.margin - d).pow(2).sum()


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
    if name in ("psource", "esm"):
        return PointSourceNet(**kwargs)
    if name == "ksiren":
        return KScaledSiren(**kwargs)
    if name == "kpmmlp":
        return KPlaneMMLP(**kwargs)
    raise ValueError(f"unknown model: {name}")
