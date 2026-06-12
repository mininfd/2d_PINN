"""PINN training & evaluation for broadband 2D Helmholtz sound-field interpolation.

The network predicts the per-frequency-normalized complex pressure
P(x,y,f) / s(f), where s(f) is the RMS of |P| over the training mics
(derivable from measurements only). Since the Helmholtz equation is linear
and s depends only on f, the normalized field satisfies the same PDE in (x,y).

Loss = MSE_data + pde_weight * MSE_pde, with the PDE residual normalized by k^2
so that all frequencies contribute on a comparable scale.
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .models import make_model

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _k_norm(k, k_min, k_max):
    return 2.0 * (k - k_min) / (k_max - k_min) - 1.0


def helmholtz_residual(model, xy, k_phys, k_min, k_max):
    """k^2-normalized Helmholtz residual at collocation points.

    xy: [N,2] in [0,1]^2 (requires_grad not needed by caller), k_phys: [N,1].
    """
    x = xy[:, 0:1].clone().requires_grad_(True)
    y = xy[:, 1:2].clone().requires_grad_(True)
    inp = torch.cat([2 * x - 1, 2 * y - 1, _k_norm(k_phys, k_min, k_max)], dim=1)
    u = model(inp)  # [N,2]
    res = []
    for ch in range(2):
        gx, gy = torch.autograd.grad(u[:, ch].sum(), [x, y], create_graph=True)
        uxx = torch.autograd.grad(gx.sum(), x, create_graph=True)[0]
        uyy = torch.autograd.grad(gy.sum(), y, create_graph=True)[0]
        res.append(uxx + uyy + k_phys ** 2 * u[:, ch:ch + 1])
    return torch.cat(res, dim=1) / k_phys ** 2


def train_pinn(model, mic_xy, freqs, p_mic, *, steps=20000, lr=2e-3,
               data_batch=8192, colloc_batch=4096, pde_weight=1.0,
               device="cuda", seed=0, log_every=1000, log=print):
    """Train a PINN on mic measurements. Returns per-frequency scale s [F]."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    k = D.wavenumber(freqs)  # [F]
    k_min, k_max = float(k.min()), float(k.max())

    # per-frequency normalization from mic data only
    scale = np.sqrt(np.mean(np.abs(p_mic) ** 2, axis=1))  # [F]
    p_n = p_mic / scale[:, None]  # [F, M]

    F, M = p_n.shape
    xy = np.broadcast_to(mic_xy[None, :, :], (F, M, 2)).reshape(-1, 2)
    kk = np.repeat(k, M)[:, None]
    tgt = np.stack([p_n.real.ravel(), p_n.imag.ravel()], axis=1)

    xy_t = torch.tensor(xy, dtype=torch.float32, device=device)
    k_t = torch.tensor(kk, dtype=torch.float32, device=device)
    inp_data = torch.cat([2 * xy_t - 1, _k_norm(k_t, k_min, k_max)], dim=1)
    tgt_t = torch.tensor(tgt, dtype=torch.float32, device=device)
    n_data = inp_data.shape[0]

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps,
                                                       eta_min=lr * 1e-2)
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, n_data, (min(data_batch, n_data),), device=device)
        pred = model(inp_data[idx])
        loss_data = torch.mean((pred - tgt_t[idx]) ** 2)

        cxy = torch.rand(colloc_batch, 2, device=device)
        ck = torch.empty(colloc_batch, 1, device=device).uniform_(k_min, k_max)
        r = helmholtz_residual(model, cxy, ck, k_min, k_max)
        loss_pde = torch.mean(r ** 2)

        loss = loss_data + pde_weight * loss_pde
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if step % log_every == 0 or step == 1:
            log(f"step {step:6d}  loss {loss.item():.3e}  "
                f"data {loss_data.item():.3e}  pde {loss_pde.item():.3e}  "
                f"({(time.time() - t0) / step * 1000:.0f} ms/step)")
    return scale


@torch.no_grad()
def evaluate_nmse(model, scale, freqs, device="cuda", batch=131072):
    """Broadband NMSE [dB] on the 33x33 held-out grid.

    Returns (nmse_db, per_freq_nmse_db [F]).
    """
    pts = D.eval_grid()
    p_true = D.pressure(pts, freqs)  # [F, N] complex
    k = D.wavenumber(freqs)
    k_min, k_max = float(k.min()), float(k.max())

    F, N = p_true.shape
    xy = np.broadcast_to(pts[None, :, :], (F, N, 2)).reshape(-1, 2)
    kk = np.repeat(k, N)[:, None]
    preds = []
    model = model.to(device).eval()
    for i in range(0, F * N, batch):
        xy_t = torch.tensor(xy[i:i + batch], dtype=torch.float32, device=device)
        k_t = torch.tensor(kk[i:i + batch], dtype=torch.float32, device=device)
        inp = torch.cat([2 * xy_t - 1, _k_norm(k_t, k_min, k_max)], dim=1)
        preds.append(model(inp).cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    p_hat = (pred[:, 0] + 1j * pred[:, 1]).reshape(F, N) * scale[:, None]

    err2 = np.abs(p_hat - p_true) ** 2
    ref2 = np.abs(p_true) ** 2
    nmse_db = 10 * np.log10(err2.sum() / ref2.sum())
    per_freq = 10 * np.log10(err2.sum(axis=1) / ref2.sum(axis=1))
    return float(nmse_db), per_freq


def append_summary(row: dict, path=None):
    path = Path(path) if path else RESULTS / "summary.csv"
    path.parent.mkdir(exist_ok=True)
    fields = ["model", "mics", "steps", "nmse_db", "wall_min", "notes"]
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def run_experiment(model_name, n_mics_side, *, steps=20000, notes="",
                   model_kwargs=None, train_kwargs=None, save_ckpt=None,
                   summary_path=None, log=print):
    """End-to-end: build data -> train -> evaluate -> record. Returns result dict."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    freqs = D.frequencies()
    mic_xy = D.mic_grid(n_mics_side)
    p_mic = D.pressure(mic_xy, freqs)

    model = make_model(model_name, **(model_kwargs or {}))
    t0 = time.time()
    scale = train_pinn(model, mic_xy, freqs, p_mic, steps=steps,
                       device=device, log=log, **(train_kwargs or {}))
    nmse_db, per_freq = evaluate_nmse(model, scale, freqs, device=device)
    wall_min = (time.time() - t0) / 60.0

    n_mics = n_mics_side ** 2
    log(f"[{model_name} {n_mics} mics] NMSE = {nmse_db:.2f} dB "
        f"({wall_min:.1f} min)")
    RESULTS.mkdir(exist_ok=True)
    if save_ckpt:
        torch.save({"model": model.state_dict(), "scale": scale,
                    "model_name": model_name, "n_mics_side": n_mics_side},
                   RESULTS / save_ckpt)
    append_summary({"model": model_name, "mics": n_mics, "steps": steps,
                    "nmse_db": f"{nmse_db:.2f}", "wall_min": f"{wall_min:.1f}",
                    "notes": notes}, path=summary_path)
    return {"model": model_name, "mics": n_mics, "nmse_db": nmse_db,
            "per_freq_nmse_db": per_freq, "wall_min": wall_min,
            "net": model, "scale": scale}
