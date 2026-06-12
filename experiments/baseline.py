"""Baseline: tanh-MLP PINN, 64 mics. Writes results/baseline.log and summary.csv."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.train import RESULTS, run_experiment


def main():
    RESULTS.mkdir(exist_ok=True)
    log_path = RESULTS / "baseline.log"
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(str(msg))

    res = run_experiment("tanh", 8, steps=20000, notes="baseline tanh-MLP",
                         save_ckpt="baseline_model.pt", log=log)

    pf = res["per_freq_nmse_db"]
    lines.append("")
    lines.append(f"BASELINE NMSE [dB] = {res['nmse_db']:.2f}")
    lines.append(f"config: tanh-MLP 5x256, 64 mics (8x8), 20000 steps, "
                 f"Adam cosine lr 2e-3, pde_weight 1.0")
    lines.append(f"per-freq NMSE: best {pf.min():.1f} dB, worst {pf.max():.1f} dB, "
                 f"median {np.median(pf):.1f} dB")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {log_path}")


if __name__ == "__main__":
    main()
