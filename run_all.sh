#!/usr/bin/env bash
# Full pipeline: tests -> baseline -> SIREN -> mMLP -> sparse study -> figures.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-/c/Projects/venv/Scripts/python.exe}"

"$PY" -m pytest tests -x -q

"$PY" experiments/baseline.py
"$PY" experiments/run.py --model siren --ckpt best_model.pt --notes "SIREN"
"$PY" experiments/run.py --model mmlp --notes "modified MLP"

"$PY" experiments/sparse_study.py --model siren
"$PY" experiments/visualize.py --ckpt results/best_model.pt

echo "ALL DONE"
