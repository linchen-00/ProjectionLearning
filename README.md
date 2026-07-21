# Paper Supplement — Projection Learning on Manifolds

This repository contains the supplementary source code for the paper:
experiment scripts, the projection-learning library (`src/projlearn/`), a
tutorial notebook, and packaging and dependency files. Trained models,
checkpoints, downloaded datasets, generated figures, cached metrics, experiment
results, and machine-specific files are intentionally excluded from version
control because they are large, generated, downloaded, or local to a particular
environment.

Generated models and experimental outputs can be recreated locally by running
the corresponding experiment scripts. Datasets and external evaluation
dependencies may require separate downloading or setup as described below.

## 1. Folder layout

```
paper_supplement_final/
├── README.md
├── requirements.txt
├── pyproject.toml
├── run_all.sh                     # one-shot runner for all experiments
├── src/
│   └── projlearn/                 # the library (manifolds, losses, training, …)
├── experiments/
│   ├── 01_2d_circle.py            # 2D circle projection vs diffusion
│   ├── 02_k_torus.py              # k-torus n-vs-d MIE scaling
│   ├── 03_torus_1d_images.py      # circle in image space
│   ├── 04_torus_circle_union.py   # torus and circle in R^3
│   ├── 05_torus_256d.py           # 2-torus in R^256, multi-seed
│   └── 06_mnist_multiseed.py      # MNIST multi-seed evaluation
└── trained_models/                # generated locally; not included in Git
```

The repository does not include `trained_models/`, `MNIST/`,
`capsule-net-pytorch/`, or `capsnet_ckpt/`.  The MNIST experiment can download
or clone those caches when they are absent.  All experiments use file-relative paths
(`Path(__file__).resolve()`) to locate the supplement root.

## 2. Environment

- Python 3.10 or later required (developed on 3.11).
- Create a fresh virtual environment and install requirements:

```bash
cd paper_supplement_final
python -m venv .venv   (or python3 -m venv .venv)
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install .
```

> **Important.** On some systems, Python 3 is invoked as `python3` rather than
> `python`; replace the command accordingly. The supplement's `projlearn`
> library lives in `src/projlearn/` and is installed as a package via
> `pip install .` as shown above.

Some entries in `requirements.txt` are only used by specific experiments:

- `POT` — used by `05_torus_256d.py` for Wasserstein evaluation.
- `gdown` — only needed by `06_mnist_multiseed.py` to *auto-download* the
  CapsNet checkpoint from Google Drive.
- `psutil` — used by `05_torus_256d.py` for the Wasserstein RAM check.

## 3. How to run each experiment

Recommended: run scripts from the supplement root. Both of the following
invocations work and resolve `projlearn` to the installed
`src/projlearn/` package:

```bash
# (a) from the supplement root (recommended)
cd paper_supplement_final
python experiments/01_2d_circle.py

# (b) from inside experiments/ — also works
cd paper_supplement_final/experiments
python 01_2d_circle.py
```

Outputs are written under `trained_models/`, with one subdirectory per
experiment. These directories are created locally and are ignored by Git.

```bash
# 1. Experiment 01: 2D circle, projection vs diffusion
python experiments/01_2d_circle.py
# Outputs: trained_models/01_2d_circle/
# Flags: --force-retrain, --force-retrain-projection,
#        --force-retrain-diffusion, --no-show

# 2. Experiment 02: k-torus sample complexity across dimensions
python experiments/02_k_torus.py
# Outputs: trained_models/02_k_torus/
# Flag: --force-train

# 3. Experiment 03: 1-torus represented as 8x8 image data
python experiments/03_torus_1d_images.py
# Outputs: trained_models/03_torus_1d/
# Flag: --force-retrain

# 4. Experiment 04: union of a torus surface and a circle in R^3
python experiments/04_torus_circle_union.py
# Outputs: trained_models/04_torus_circle_union/
# Flag: --force-retrain

# 5. Experiment 05: 2-torus embedded in R^256, evaluated across seeds
python experiments/05_torus_256d.py
# Outputs: trained_models/05_torus_256d/
# Flags: --force-retrain, --force-recompute-wasserstein

# 6. Experiment 06: multi-seed MNIST evaluation for projection and diffusion
python experiments/06_mnist_multiseed.py
# Outputs:
#   trained_models/06_mnist/all_digits_multiseed/
#   trained_models/06_mnist/diffusion_multiseed/
#   trained_models/06_mnist/figures/
# Variants:
python experiments/06_mnist_multiseed.py --only projection
python experiments/06_mnist_multiseed.py --only diffusion
# Flags: --force-eval, --force-retrain, --force-retrain-projection,
#        --force-retrain-diffusion, --no-show
```

Approximate computing resources for uncached runs on a single NVIDIA T4
(16 GB) are as follows. These estimates are based on the default hyperparameters
in the scripts. On a fresh clone, runs may train models or regenerate missing
artifacts locally.

**Experiment 01.** Training the 2D projection MLP uses 3,000 total projection
epochs (500 pretraining + 2,500 projection), and the ScoreNetVector diffusion
baseline uses 10,000 epochs. Including generation of the 1,000-sample,
10,000-step reverse-ODE snapshots for the figure, this should take
approximately 2 minutes on one T4 with less than 100 MB of peak GPU memory.

**Experiment 02.** Each projection-model training uses 2,000 total epochs
(500 pretraining + 1,500 projection). A full `--force-train` run performs 28
Part A trainings, one for each sample-size-search attempt recorded as a data row
in `trial_data.csv` after the header. Part B then performs 10 more
ambient-256 trainings: 5 torus dimensions times 2 noise scales
(`delta = 1` and `delta = 16 / sqrt(d)`) recorded in `partb_results.csv`. On one
T4, this is estimated at approximately 45-90 minutes of wall-clock time, with peak
GPU memory around 300-500 MB during the largest ambient-256 minibatches.

**Experiment 03.** A full `--force-retrain` run trains the 5,000-epoch
projection model and the 50,000-epoch vector diffusion model, including
diffusion metric callbacks that sample 5,000 points every 1,000 epochs. On one
T4, this is estimated at approximately 25-40 minutes of wall-clock time, with
about 400-600 MB of peak GPU memory.

**Experiment 04.** A full `--force-retrain` run trains the two-stage
torus-plus-circle projection model for 5,000 total projection iterations and
then regenerates the visualization samples. On one T4, this is estimated at
approximately 5-10 minutes of wall-clock time, with less than 100 MB of peak GPU
memory.

**Experiment 05.** A full `--force-retrain` run trains five seeds of the
ambient-256 two-torus projection model, each for 2,400 total projection epochs
(400 pretraining + 2,000 projection), and then runs the saved metric/plot
pipeline. By default, the script loads
`trained_models/05_torus_256d/wasserstein.pt` if it exists locally. Otherwise,
follow the behavior implemented in the script to compute or regenerate the
required output. On one T4, training is estimated at approximately 4-8 minutes
per seed, or 20-40 minutes for all five seeds, with about 300-500 MB of peak GPU
memory during training. Wasserstein recomputation is CPU/RAM intensive.

**Experiment 06.** A full `--force-retrain` run uses five MNIST seeds for the
projection cascade and five seeds for the VE-SDE diffusion baseline. Per seed,
the projection cascade uses 450 total dataloader epochs across three stages
((100 pretraining + 50 projection) x 3), and the diffusion baseline uses
300 dataloader epochs followed by 500-step PC sampling of 10,000 generated
images. On one T4, the projection cascade is estimated at approximately
1-1.5 hours per seed with about 10-12 GB of peak GPU memory, while the diffusion
baseline is estimated at approximately 10 minutes of training plus about
10 minutes of sampling per seed with about 2-4 GB of peak GPU memory. The
complete forced run of both methods is therefore estimated at roughly
7-10 GPU-hours, plus the metric and plotting passes.

Or run everything in sequence:

```bash
bash run_all.sh
```

In a headless terminal, use a non-interactive Matplotlib backend:

```bash
MPLBACKEND=Agg bash run_all.sh
```

## 4. Output locations

- Checkpoints, cached metrics, reports, and generated figures are written under
  the corresponding experiment directory inside `trained_models/`.
- Running the corresponding experiment, or `run_all.sh`, regenerates local
  artifacts under `trained_models/`.
- Re-running an experiment overwrites or refreshes outputs in its corresponding
  directory under `trained_models/`.
- For example, `06_mnist_multiseed.py` writes formatted reports to
  `multiseed_report.txt`, and `05_torus_256d.py` writes stdout to
  `trained_models/05_torus_256d/output.txt`.

## 5. Data and caches

The MNIST experiment can download MNIST through `torchvision`, clone the CapsNet
model code, and download/extract the CapsNet checkpoint when those cache
directories are absent.

- **`trained_models/06_mnist/{all_digits_multiseed,diffusion_multiseed}/`**
  — local output directories for 5 random seeds × {projection cascade,
  diffusion model}. Forced re-evaluation can regenerate deterministic MNIST
  splits and, if generated diffusion samples are missing, sample from a locally
  trained or downloaded `diff_model.pt`.

## 6. Reproducibility

Generated artifacts under `trained_models/` are local reproduction outputs and
are not tracked by Git.  The experiment scripts set deterministic seeds (`torch.manual_seed`,
`numpy.random.seed`) before non-deterministic steps.  With identical PyTorch
versions and CPU/MPS/CUDA backend, numerical outputs match the paper to ~4-5
significant digits. Results in the paper were produced with `torch==2.4`, but
any 2.x release should reproduce them up to floating-point determinism.

## 7. Troubleshooting

- `ModuleNotFoundError: projlearn` — the install step was
  likely skipped. Run `pip install .` from the supplement root and
  retry.
- `FileNotFoundError: trained_models/…`. The scripts resolve paths relative to
  their own location (`__file__`), so the working directory does not matter.
  Run the relevant experiment or training command to create the required
  `trained_models/` directory and checkpoint files locally. Also verify that any
  required datasets or external evaluation dependencies have been installed or
  downloaded.
- CapsNet metrics missing in an MNIST rerun. They are skipped automatically
  when the CapsNet model cannot be loaded; the warning is informational. The
  non-CapsNet metric (pixel-space memorisation score) is unaffected.
