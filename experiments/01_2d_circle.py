"""Experiment 01: 2D circle, projection vs. diffusion.

What this experiment does
-------------------------
This script trains or loads two models on 2D circle data:
1. a projection model learned with Algorithm 1
2. a score-based diffusion model

It then saves one combined figure with:
1. diffusion samples near reverse-ODE time ``t = 1e-3``
2. diffusion samples near reverse-ODE time ``t = 1e-6``
3. projection-model outputs on points sampled from ``M_delta``

Outputs save to ``trained_models/01_2d_circle/``

CLI
---
Run with:
``python experiments/01_2d_circle.py``

Flags:
``--force-retrain`` to retrain both models
``--force-retrain-projection`` to retrain only the projection model
``--force-retrain-diffusion`` to retrain only the diffusion model
``--no-show`` to save figures without opening a plot window
"""


import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from projlearn.manifolds import circle_2d
from projlearn.models import ProjectionNet, ScoreNetVector
from projlearn.sampling import (
    make_sde_fns,
    tubular_nbh,
)
from projlearn.training import train_diffusion, train_projection_algorithm1

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")


# ── Hyper-parameters ────────────────────────────────────────────────────────
data_seed = 2
train_seed = 30
eval_seed = 123

N = 40
delta = 0.7
ambient_dim = 2
num_points_per_samples = 250

# Projection training (two-phase, cosine schedule each)
proj_phase1_epochs = 500
proj_phase2_epochs = 2500
proj_lr1 = 6e-3
proj_lr2 = 1e-3

# Diffusion training
diffusion_sigma = 25
diffusion_train_epochs = 10000
diffusion_train_M = 100
diffusion_train_lr = 1e-3

# Diffusion sampling for plots
diffusion_batch_size = 1000
diffusion_num_steps = 10000
diffusion_eps = 1e-6
desired_times = [1e-3, 1e-6]

_ROOT = Path(__file__).resolve().parent.parent
model_dir = _ROOT / "trained_models" / "01_2d_circle"
output_dir = model_dir
model_dir.mkdir(parents=True, exist_ok=True)

projection_ckpt = model_dir / "projection_model.pt"
diffusion_ckpt = model_dir / "diffusion_model.pt"


# ── Data ────────────────────────────────────────────────────────────────────
torch.manual_seed(data_seed)
samples = circle_2d(N, ambient_dim=ambient_dim)


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
})


# ── Train / load helpers ────────────────────────────────────────────────────
def train_projection_model():
    print("=" * 60)
    print("Training projection_model (Phase 1: C1+C3 cosine, "
          "Phase 2: C1+C2+C3 cosine)")
    print("=" * 60)
    torch.manual_seed(train_seed)
    model = ProjectionNet(ambient_dim).to(device)
    train_projection_algorithm1(
        model, samples, delta,
        n_pretrain=proj_phase1_epochs, n_project=proj_phase2_epochs,
        lr_pretrain=proj_lr1, lr_project=proj_lr2,
        num_noisy=num_points_per_samples, r_inner=delta / 2,
        loss_pretrain={"C1": 1, "C3": 1},
        loss_project={"C1": 1, "C2": 0.5, "C3": 0.5},
        scheduler_pretrain="cosine", scheduler_project="cosine",
        print_every=200, device=device,
    )
    torch.save(model.state_dict(), projection_ckpt)
    print(f"Saved {projection_ckpt}")
    return model.cpu().eval()


def train_diffusion_model(marginal_prob_std_fn):
    print("=" * 60)
    print("Training diffusion_model (score-based SDE, "
          f"{diffusion_train_epochs} epochs)")
    print("=" * 60)
    torch.manual_seed(train_seed)
    model = ScoreNetVector(ambient_dim, marginal_prob_std_fn).to(device)
    train_diffusion(model, samples.to(device), marginal_prob_std_fn,
                    n_epochs=diffusion_train_epochs,
                    M=diffusion_train_M, lr=diffusion_train_lr)
    torch.save(model.state_dict(), diffusion_ckpt)
    print(f"Saved {diffusion_ckpt}")
    return model.eval()


def load_projection_model():
    model = ProjectionNet(ambient_dim)
    model.load_state_dict(torch.load(
        projection_ckpt, map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_diffusion_model(marginal_prob_std_fn):
    model = ScoreNetVector(ambient_dim, marginal_prob_std_fn).to(device)
    model.load_state_dict(torch.load(
        diffusion_ckpt, map_location=device, weights_only=True))
    model.eval()
    return model


# ── Plotting helpers (unchanged) ────────────────────────────────────────────
@torch.no_grad()
def sample_diffusion_snapshots(score_model, ambient_dim, marginal_prob_std_fn,
                               diffusion_coeff_fn, desired_ts, batch_size=5000,
                               num_steps=10000, device="cpu", eps=1e-6,
                               ode_last_frac=1.0):
    """Sample reverse ODE and capture states nearest the requested times."""
    score_model.eval()
    t1 = torch.ones(batch_size, device=device)
    x = torch.randn(batch_size, ambient_dim, device=device) * \
        marginal_prob_std_fn(t1)[:, None]

    time_steps = torch.linspace(1.0, eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    snapshot_indices = {}
    for requested_t in desired_ts:
        target_t = torch.tensor(
            requested_t, dtype=time_steps.dtype, device=device)
        idx = int(torch.argmin(torch.abs(time_steps - target_t)).item())
        snapshot_indices[idx] = requested_t

    snapshots = {}
    actual_times = {}

    for i, ts in enumerate(time_steps):
        batch_t = torch.ones(batch_size, device=device) * ts
        g = diffusion_coeff_fn(batch_t)
        drift = (g ** 2)[:, None] * score_model(x, batch_t)

        if i >= int((1.0 - ode_last_frac) * num_steps):
            x = x + 0.5 * drift * step_size
        else:
            x = x + drift * step_size + \
                torch.sqrt(step_size) * g[:, None] * torch.randn_like(x)

        if i in snapshot_indices:
            requested_t = snapshot_indices[i]
            snapshots[requested_t] = x.detach().cpu().clone()
            actual_times[requested_t] = ts.item()

    return snapshots, actual_times


def fmt_time(t):
    if t >= 1e-2:
        return rf"$t = {t:g}$"
    exp = round(math.log10(t))
    return rf"$t = 10^{{{exp}}}$"


def style_axis(ax, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#888888")
    ax.grid(True, color="#cccccc", linewidth=0.6, linestyle="--", alpha=0.8)
    ax.tick_params(axis="both", which="both",
                   bottom=False, left=False, top=False, right=False,
                   labelbottom=False, labelleft=False)


def _draw_diffusion_ax(ax, gen_samples, t_label=None):
    ax.scatter(gen_samples[:, 0], gen_samples[:, 1], s=14,
               color="#1b6fab", edgecolors="none", alpha=0.85)
    ax.scatter(samples[:, 0], samples[:, 1], s=80, marker="^",
               color="red", edgecolors="none", zorder=5)
    if t_label is not None:
        ax.text(0.97, 0.95, fmt_time(t_label), transform=ax.transAxes,
                ha="right", va="top", fontsize=18)


@torch.no_grad()
def _draw_projection_ax(ax, model, M_delta, num_lines=500):
    dev = next(model.parameters()).device
    mapped = model(model(M_delta.to(dev))).cpu()
    M_delta_cpu = M_delta.cpu()

    n = len(M_delta_cpu)
    line_idx = torch.randperm(n)[:min(num_lines, n)]

    M_delta_sub = M_delta_cpu[line_idx]
    mapped_sub = mapped[line_idx]

    for i in range(len(line_idx)):
        ax.plot([M_delta_sub[i, 0], mapped_sub[i, 0]],
                [M_delta_sub[i, 1], mapped_sub[i, 1]],
                color="gray", linewidth=0.6, linestyle="--", alpha=0.5)

    ax.scatter(M_delta_sub[:, 0], M_delta_sub[:, 1],
               s=1, color="gray", alpha=1,
               label=r"$\mathcal{M}_\delta$")
    ax.scatter(mapped[:, 0], mapped[:, 1],
               s=14, color="#1b6fab", edgecolors="none",
               label=r"Generated samples")
    ax.scatter(samples[:, 0], samples[:, 1],
               s=80, marker="^", color="red", edgecolors="none", zorder=5,
               label="training samples")


def plot_combined_figure(diffusion_snapshots, actual_times, projection_model,
                         M_delta, samples, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))

    for ax, requested_t in zip(axes[:2], desired_times):
        _draw_diffusion_ax(ax, diffusion_snapshots[requested_t], requested_t)
        style_axis(ax)

    _draw_projection_ax(axes[2], projection_model, M_delta)
    style_axis(axes[2])

    fig.tight_layout(pad=0.4, rect=[0, 0, 1, 0.91])

    p0 = axes[0].get_position()
    p1 = axes[1].get_position()
    p2 = axes[2].get_position()
    top_y = p0.y1
    gap = 1.0 - top_y

    arrow_y = top_y + gap * 0.32
    label_y = top_y + gap * 0.78

    axes[0].annotate(
        "",
        xy=(p1.x1, arrow_y),
        xytext=(p0.x0, arrow_y),
        xycoords="figure fraction",
        textcoords="figure fraction",
        arrowprops=dict(arrowstyle="-|>", lw=2.0, color="black",
                        mutation_scale=22),
        annotation_clip=False,
    )
    fig.text((p0.x0 + p1.x1) / 2, label_y,
             "Reverse diffusion ODE: time decreases",
             ha="center", va="center", fontsize=25)
    fig.text((p2.x0 + p2.x1) / 2, top_y + gap * 0.75,
             "Projection model",
             ha="center", va="center", fontsize=25, linespacing=1.3)

    fig.savefig(save_path, dpi=600, bbox_inches="tight")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force-retrain", action="store_true",
                    help="Retrain both models even if checkpoints exist.")
    ap.add_argument("--force-retrain-projection", action="store_true")
    ap.add_argument("--force-retrain-diffusion", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    retrain_proj = (args.force_retrain or args.force_retrain_projection
                    or not projection_ckpt.exists())
    retrain_diff = (args.force_retrain or args.force_retrain_diffusion
                    or not diffusion_ckpt.exists())

    marginal_prob_std_fn, diffusion_coeff_fn = make_sde_fns(
        sigma=diffusion_sigma)

    if retrain_proj:
        projection_model = train_projection_model()
    else:
        print(f"Found {projection_ckpt}; loading.")
        projection_model = load_projection_model()

    if retrain_diff:
        score_model = train_diffusion_model(marginal_prob_std_fn)
    else:
        print(f"Found {diffusion_ckpt}; loading.")
        score_model = load_diffusion_model(marginal_prob_std_fn)

    # Diffusion plotting samples (independent of training)
    torch.manual_seed(eval_seed)
    snapshots, actual_times = sample_diffusion_snapshots(
        score_model, ambient_dim, marginal_prob_std_fn, diffusion_coeff_fn,
        desired_times,
        batch_size=diffusion_batch_size,
        num_steps=diffusion_num_steps,
        device=device, eps=diffusion_eps, ode_last_frac=1.0,
    )

    M_delta = tubular_nbh(samples, delta, ambient_dim,
                          num_points_per_sample=25, device='cpu')

    combined_path = output_dir / "01_projdiff_combined.png"
    plot_combined_figure(snapshots, actual_times, projection_model,
                         M_delta, samples, combined_path)

    print("Saved outputs:")
    print(f"  {combined_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
