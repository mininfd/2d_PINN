"""Sound-field snapshots: true vs. predicted Re(P) on the 33x33 grid.

Usage: python experiments/visualize.py --ckpt results/best_model.pt
Writes results/field_snapshots.png and results/nmse_vs_freq.png.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import data as D
from src.models import make_model
from src.train import RESULTS, evaluate_nmse


def predict(model, scale, freqs, device):
    pts = D.eval_grid()
    k = D.wavenumber(D.frequencies())
    k_min, k_max = float(k.min()), float(k.max())
    kf = D.wavenumber(freqs)
    out = []
    model = model.to(device).eval()
    with torch.no_grad():
        for i, f in enumerate(freqs):
            xy = torch.tensor(pts, dtype=torch.float32, device=device)
            kk = torch.full((len(pts), 1), float(kf[i]), device=device)
            kn = 2 * (kk - k_min) / (k_max - k_min) - 1
            inp = torch.cat([2 * xy - 1, kn], dim=1)
            p = model(inp).cpu().numpy()
            out.append((p[:, 0] + 1j * p[:, 1]) * scale[i])
    return np.array(out)  # [F, N]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/best_model.pt")
    ap.add_argument("--freqs", type=float, nargs="+",
                    default=[500.0, 2000.0, 6000.0])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ROOT / args.ckpt, map_location=device, weights_only=False)
    model = make_model(ck["model_name"])
    model.load_state_dict(ck["model"])
    scale_full = ck["scale"]

    all_freqs = D.frequencies()
    idx = [int(np.argmin(np.abs(all_freqs - f))) for f in args.freqs]
    freqs = all_freqs[idx]
    scale = scale_full[idx]

    pts = D.eval_grid()
    p_true = D.pressure(pts, freqs)
    p_hat = predict(model, scale, freqs, device)

    n = 33
    fig, axes = plt.subplots(2, len(freqs), figsize=(4 * len(freqs), 7.5),
                             constrained_layout=True)
    for j, f in enumerate(freqs):
        vmax = np.abs(p_true[j].real).max()
        for row, (field, label) in enumerate(
                [(p_true[j], "true"), (p_hat[j], "PINN")]):
            ax = axes[row, j]
            im = ax.imshow(field.real.reshape(n, n).T, origin="lower",
                           extent=[0, 1, 0, 1], cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax)
            ax.set_title(f"{label}  Re(P)  {f:.0f} Hz")
            fig.colorbar(im, ax=ax, shrink=0.8)
    out1 = RESULTS / "field_snapshots.png"
    fig.savefig(out1, dpi=150)
    print(f"wrote {out1}")

    nmse_db, per_freq = evaluate_nmse(model, scale_full, all_freqs,
                                      device=device)
    fig2, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(all_freqs, per_freq)
    ax.axhline(nmse_db, color="r", ls="--",
               label=f"broadband {nmse_db:.1f} dB")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("NMSE [dB]")
    ax.legend()
    ax.grid(alpha=0.3)
    out2 = RESULTS / "nmse_vs_freq.png"
    fig2.savefig(out2, dpi=150)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
