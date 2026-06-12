#!/usr/bin/env bash
# Full pipeline: tests -> baseline -> SIREN -> mMLP -> Herglotz -> ESM (best)
# -> sparse study -> figures. Regenerates results/ from scratch.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-/c/Projects/venv/Scripts/python.exe}"

"$PY" -m pytest tests -x -q

rm -f results/summary.csv results/sparse_study.csv

# baselines / NN-only PINNs (PDE as a loss)
"$PY" experiments/baseline.py
"$PY" experiments/run.py --model siren --lr 1e-4 --pde-weight 0.01 \
    --grad-clip 1.0 --curriculum --ckpt siren_64.pt \
    --notes "SIREN pde_w=0.01 clip lr=1e-4 curriculum"
"$PY" experiments/run.py --model mmlp --ckpt mmlp_64.pt --notes "modified MLP"

# physics-exact architectures (PDE satisfied by construction, data loss only)
"$PY" experiments/run.py --model herglotz --lr 5e-4 --pde-weight 0 \
    --ckpt herglotz_64.pt --notes "Herglotz J=256, data-only"
"$PY" experiments/run.py --model psource --mics 8 --lr 5e-4 --pde-weight 0 \
    --reg 1.0 --curriculum --steps 40000 --ckpt best_model.pt \
    --notes "ESM M=16 + k-curriculum 40k (best)"

# sparse-mic study with the best model
"$PY" experiments/sparse_study.py --model psource --pde-weight 0 --lr 5e-4 \
    --reg 1.0 --curriculum --steps 40000

"$PY" experiments/visualize.py --ckpt results/best_model.pt

echo "ALL DONE"
