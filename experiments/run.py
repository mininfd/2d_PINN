"""Generic experiment runner: python experiments/run.py --model siren --mics 8."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import run_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="tanh | siren | mmlp")
    ap.add_argument("--mics", type=int, default=8, help="mics per side")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--pde-weight", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--grad-clip", type=float, default=None)
    ap.add_argument("--omega0", type=float, default=None,
                    help="SIREN first-layer omega0")
    ap.add_argument("--omega0-hidden", type=float, default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--ckpt", default=None, help="checkpoint filename")
    ap.add_argument("--summary", default=None, help="summary csv path")
    args = ap.parse_args()

    model_kwargs = {}
    if args.omega0 is not None:
        model_kwargs["omega0"] = args.omega0
    if args.omega0_hidden is not None:
        model_kwargs["omega0_hidden"] = args.omega0_hidden

    run_experiment(args.model, args.mics, steps=args.steps, notes=args.notes,
                   model_kwargs=model_kwargs,
                   train_kwargs={"pde_weight": args.pde_weight, "lr": args.lr,
                                 "grad_clip": args.grad_clip},
                   save_ckpt=args.ckpt, summary_path=args.summary)


if __name__ == "__main__":
    main()
