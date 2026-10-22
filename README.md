# Projection Learning on Manifolds [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MiDFBP05REl0mNaKOWu8JP0Ufr97qbms?usp=sharing)

Projection learning trains maps that project noisy ambient samples back onto low-dimensional data manifolds. This repository builds a PyTorch library for projection models, together with experiment scripts for synthetic manifolds, image-valued manifolds, and MNIST.

The repository contains:

- `src/projlearn/`: reusable projection-learning components, including
  manifolds, models, losses, training loops, sampling, evaluation, and
  visualization utilities.
- `experiments/`: scripts for the circle, torus, image-manifold, torus-union, high-dimensional torus, MNIST, and model-capacity experiments.
- `requirements.txt`, `pyproject.toml`, and `run_all.sh`: installation and
  experiment-running utilities.

For a guided introduction and an interactive playground, open the
[Google Colab Tutorial](https://colab.research.google.com/drive/1MiDFBP05REl0mNaKOWu8JP0Ufr97qbms?usp=sharing).

## 1. Code Structure

```text
ProjectionLearning/
├── experiments/
│   ├── 01_2d_circle.py
│   ├── 02_k_torus.py
│   ├── 03_torus_1d_images.py
│   ├── 04_torus_circle_union.py
│   ├── 05_torus_256d.py
│   ├── 06_mnist_multiseed.py
│   └── 07_capacity_sweep.py
├── src/
│   └── projlearn/
├── pyproject.toml
├── requirements.txt
└── run_all.sh
```

Generated checkpoints, cached metrics, figures, and reports are written under `trained_models/` during local runs. That directory is ignored by Git.

## 2. Installation

Python 3.10 or later is recommended.

```bash
git clone https://github.com/linchen-00/ProjectionLearning.git
cd ProjectionLearning

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install .
```


## 3. Pretrained Models

Pretrained models and generated experiment results are not available in Git. They can be downloaded from
[Google Drive](https://drive.google.com/drive/folders/1BBvd7rodzt04CBB1fjHc0RHlpFsGWHMQ?usp=sharing).

Place the downloaded `trained_models/` directory at the repository root:

```text
ProjectionLearning/
└── trained_models/
```

## 4. Quickstart

### Use Cached Results

After installing the package and downloading `trained_models/`, run an
experiment script from the repository root. For example:

```bash
python3 experiments/01_2d_circle.py
```

Most scripts reuse cached checkpoints or metrics when they are present and regenerate missing results locally when needed.

### Train From Scratch

To retrain a single experiment, use the corresponding force flag. For example:

```bash
python3 experiments/01_2d_circle.py --force-retrain
```

To run all experiments in sequence:

```bash
bash run_all.sh
```

### Build a New Projection Experiment

The `projlearn` package can be used directly in a new script or notebook:

```python
import torch

from projlearn.manifolds import circle_2d
from projlearn.models import ProjectionNet
from projlearn.training import train_projection_algorithm1

samples = circle_2d(N=50, ambient_dim=2)
model = ProjectionNet(ambient_dim=2)

train_projection_algorithm1(
    model,
    samples,
    delta=0.5,
    n_pretrain=300,
    n_project=1000,
    lr_pretrain=5e-3,
    lr_project=1e-3,
    num_noisy=50,
    r_inner=0.25,
    c2_mode="exact",
)

with torch.no_grad():
    noisy_samples = samples + 0.5 * torch.randn_like(samples)
    projected_once = model(noisy_samples)    # apply the projection map once
    projected_twice = model(projected_once)  # apply the projection map twice
```

The second application is optional, but it can further refine samples because a composition of projection maps can still be used as a projection map.

## 5. Practical Guidance

- Algorithm 1 trains a projection map at a single noise radius. It is a good default for low-dimensional manifolds and simple experiments.
- For models trained with Algorithm 1, applying the learned projection map a small number of times can improve sampling quality.
- Algorithm 2 trains projection maps across decreasing noise radii, giving a coarse-to-fine projection procedure. It is recommended for high-dimensional data such as images.
- All experiments use the adaptive C2 loss for projection training. It combines mean squared Jacobian-vector penalties at `T(x)` and `x`, starting with equal weights and rebalancing them after each epoch.
- If the Jacobian-vector penalty in the objective is expensive, use `c2_mode="fd"` for a finite-difference approximation or `c2_mode="subsample"` to evaluate on the random subset. Use `c2_mode="exact"` for exact Jacobian-vector products on the full batch.
- If a noisy batch does not fit in memory, set `loss_chunk_size` to process its points in smaller groups while keeping one optimizer update per batch.

## 6. Citation

Coming soon.
