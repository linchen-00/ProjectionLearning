# Projection Learning on Manifolds

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MiDFBP05REl0mNaKOWu8JP0Ufr97qbms?usp=sharing)

Projection learning trains maps that project noisy ambient samples back
onto low-dimensional data manifolds. This repository builds a small PyTorch
library for projection models, together with experiment scripts for synthetic
manifolds, image-valued manifolds, and MNIST.

The repository contains:

- `src/projlearn/`: reusable projection-learning components, including
  manifolds, models, losses, training loops, sampling, evaluation, and
  visualization utilities.
- `experiments/`: scripts for the circle, torus, image-manifold, torus-union,
  high-dimensional torus, and MNIST experiments.
- `requirements.txt`, `pyproject.toml`, and `run_all.sh`: installation and
  experiment-running utilities.

For a guided introduction and an interactive playground, open the
[Google Colab tutorial](https://colab.research.google.com/drive/1MiDFBP05REl0mNaKOWu8JP0Ufr97qbms?usp=sharing).

## 1. Code Structure

```text
ProjectionLearning/
├── experiments/
│   ├── 01_2d_circle.py
│   ├── 02_k_torus.py
│   ├── 03_torus_1d_images.py
│   ├── 04_torus_circle_union.py
│   ├── 05_torus_256d.py
│   └── 06_mnist_multiseed.py
├── src/
│   └── projlearn/
├── pyproject.toml
├── requirements.txt
└── run_all.sh
```

Generated checkpoints, cached metrics, figures, and reports are written under
`trained_models/` during local runs. That directory is ignored by Git.

## 2. Installation

Python 3.10 or later is recommended.

```bash
git clone https://github.com/linchen-00/ProjectionLearning.git
cd ProjectionLearning

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install .
```

If your system uses `python3` instead of `python`, replace the command
accordingly.

## 3. Pretrained Models

Pretrained models and generated experiment artifacts are not tracked in Git.
They can be downloaded from
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
python experiments/01_2d_circle.py --no-show
```

Most scripts reuse cached checkpoints or metrics when they are present and
regenerate missing artifacts locally when needed.

### Train From Scratch

To retrain a single experiment, use the corresponding force flag. For example:

```bash
python experiments/01_2d_circle.py --force-retrain --no-show
```

To rerun every experiment in sequence:

```bash
bash run_all.sh
```

For headless servers:

```bash
MPLBACKEND=Agg bash run_all.sh
```

### Build a New Projection Experiment

The `projlearn` package can be used directly in a new script or notebook:

```python
import torch

from projlearn.manifolds import circle_2d
from projlearn.models import ProjectionNet
from projlearn.training import train_projection_algorithm1

samples = circle_2d(N=128, ambient_dim=2)
model = ProjectionNet(ambient_dim=2)

train_projection_algorithm1(
    model,
    samples,
    delta=0.5,
    n_pretrain=200,
    n_project=800,
    lr_pretrain=5e-3,
    lr_project=1e-3,
    num_noisy=100,
    r_inner=0.25,
)

with torch.no_grad():
    noisy_samples = samples + 0.5 * torch.randn_like(samples)
    projected_samples = model(noisy_samples)
```

Use the experiment scripts as templates for larger studies, custom manifolds,
multi-stage projection maps, and comparisons with diffusion baselines.

## 5. Citation

Coming soon.
