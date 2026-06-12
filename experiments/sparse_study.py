"""Sparse-mic study: best model at {64, 36, 16} mics -> results/sparse_study.csv."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import RESULTS, run_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--pde-weight", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--reg", type=float, default=0.0)
    ap.add_argument("--curriculum", action="store_true")
    ap.add_argument("--omega0", type=float, default=None)
    ap.add_argument("--omega0-hidden", type=float, default=None)
    args = ap.parse_args()

    model_kwargs = {}
    if args.omega0 is not None:
        model_kwargs["omega0"] = args.omega0
    if args.omega0_hidden is not None:
        model_kwargs["omega0_hidden"] = args.omega0_hidden

    out = RESULTS / "sparse_study.csv"
    for side in (8, 6, 4):
        run_experiment(args.model, side, steps=args.steps,
                       notes=f"sparse study {side}x{side}",
                       model_kwargs=model_kwargs,
                       train_kwargs={"pde_weight": args.pde_weight,
                                     "lr": args.lr,
                                     "reg_weight": args.reg,
                                     "k_curriculum": args.curriculum},
                       save_ckpt=f"sparse_{args.model}_{side}x{side}.pt",
                       summary_path=out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
