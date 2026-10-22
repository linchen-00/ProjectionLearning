#!/usr/bin/env bash
# Run every experiment in the supplement sequentially.
# Outputs are written next to each model under trained_models/<exp>/.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY=${PYTHON:-python}

if [ ! -d experiments ]; then
  echo "No experiments/ directory found."
  echo "The trained outputs are available under trained_models/."
  exit 0
fi

$PY experiments/01_2d_circle.py
$PY experiments/02_k_torus.py
$PY experiments/03_torus_1d_images.py
$PY experiments/04_torus_circle_union.py
$PY experiments/05_torus_256d.py
# 06: cached multi-seed evaluation. Pass --force-eval directly to
# experiments/06_mnist_multiseed.py to recompute metrics.
$PY experiments/06_mnist_multiseed.py
$PY experiments/07_capacity_sweep.py
echo "All experiments finished."
