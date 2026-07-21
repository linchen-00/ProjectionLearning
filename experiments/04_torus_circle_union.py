"""Experiment 04: union of a torus surface and a circle in R^3.

What this experiment does
-------------------------
This script samples a disconnected 3D manifold made from:
the union of a torus surface and a circle curve

It trains one projection model with Algorithm 1 at a single tubular radius,
then applies the trained model to noisy tubular-neighborhood samples to produce
generated manifold points.

Outputs save to trained_models/04_torus_circle_union/

CLI
---
Run with:
``python experiments/04_torus_circle_union.py``

Flag:
``--force-retrain`` to retrain even when a checkpoint already exists
"""


import argparse
import copy
from pathlib import Path

from matplotlib.ticker import FormatStrFormatter
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import torch

from projlearn.manifolds import torus_and_circle_3d
from projlearn.models import ProjectionNet
from projlearn.sampling import tubular_nbh
from projlearn.training import train_projection_algorithm1
from projlearn.utils import second_neighbor_stats
from projlearn.visualization import plot_3d

# Device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force-retrain", action="store_true",
                    help="Retrain from scratch even if a checkpoint exists.")
    args = ap.parse_args()

    # Single-radius comparison. Geometry and sample counts match Experiment 04.
    params = {
        "N1": 600,
        "N2": 100,
        "R1": 1.0,
        "R2": 0.4,
        "Rc": 1.0,
        "ambient_dim": 3,
        "delta": 0.25,
        "num_points_per_samples": 10,
        "num_points_per_sample_viz": 50,
        "lr_pretrain": 6e-3,
        "lr_projection": 1e-3,
        "n_pretrain": 3000,
        "n_projection": 2000,
        "num_projection_applications": 4,
        "loss1": {"C1": 1.0, "C3": 1.0},
        "loss2": {"C1": 1.0, "C2": 1.0, "C3": 1.0},
    }
    params["r_inner"] = params["delta"] / 2

    _ROOT = Path(__file__).resolve().parent.parent
    save_dir = _ROOT / "trained_models" / "04_torus_circle_union"
    save_dir.mkdir(parents=True, exist_ok=True)

    seed = 2
    samples, manifold = run_one_experiment(seed, params, save_dir,
                                           force_retrain=args.force_retrain)
    save_views(
        manifold,
        samples,
        save_dir,
        show_plot=True,
    )

    print(f"Saved checkpoint and figures to {save_dir}")


def distance_to_union(points, params):
    """Exact pointwise distance to the torus-circle union."""
    rho = torch.linalg.norm(points[:, :2], dim=1)
    d_torus = (
        torch.sqrt((rho - params["R1"]) ** 2 + points[:, 2] ** 2)
        - params["R2"]
    ).abs()
    circle_radial = torch.sqrt(
        (points[:, 0] - params["Rc"]) ** 2 + points[:, 2] ** 2
    )
    d_circle = torch.sqrt(
        (circle_radial - params["Rc"]) ** 2 + points[:, 1] ** 2
    )
    return torch.minimum(d_torus, d_circle)


def apply_projection(model, points, applications):
    out = points
    for _ in range(applications):
        out = model(out)
    return out


def train_torus_circle_model(model, samples, params):
    """Run single-radius Algorithm 1."""
    train_projection_algorithm1(
        model, samples, params["delta"],
        n_pretrain=params["n_pretrain"],
        n_project=params["n_projection"],
        lr_pretrain=params["lr_pretrain"],
        lr_project=params["lr_projection"],
        num_noisy=params["num_points_per_samples"],
        r_inner=params["r_inner"],
        loss_pretrain=params["loss1"],
        loss_project=params["loss2"],
        scheduler_pretrain="constant", scheduler_project="cosine",
        print_every=100, device=device,
    )
    return model


def _crop_to_square(path, pad=40):
    """Crop near-white border, add uniform padding, then pad to a square."""
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)
    # Pixels that are NOT near-white (threshold 240 on each RGB channel)
    not_white = ~(
        (arr[..., 0] > 240) & (arr[..., 1] > 240) & (arr[..., 2] > 240)
    )
    rows = np.any(not_white, axis=1)
    cols = np.any(not_white, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    H, W = arr.shape[:2]
    rmin = max(0, rmin - pad)
    rmax = min(H - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(W - 1, cmax + pad)
    cropped = img.crop((cmin, rmin, cmax + 1, rmax + 1))
    # Pad shorter dimension to make image square
    w, h = cropped.size
    side = max(w, h)
    square = Image.new("RGBA", (side, side), (255, 255, 255, 255))
    square.paste(cropped, ((side - w) // 2, (side - h) // 2))
    square.convert("RGB").save(path)


def save_views(manifold, samples, save_dir, show_plot=False):
    """Save 3D, side, and top views for the single-delta comparison."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
    })
    # View 1
    fig1 = plot_3d(
        manifold, samples,
        view=(35, -105), show=False, show_axis_lines=True, show_legend=False,
        figsize=(8, 8)
    )
    ax = fig1.axes[0]
    if len(ax.collections) > 1:
        generated = ax.collections[1]
        if hasattr(generated, "set_depthshade"):
            generated.set_depthshade(False)
        generated.set_alpha(0.5)
        generated.set_facecolor("#1b6fab")
        generated.set_edgecolor("#1b6fab")
    ax.set_facecolor("white")
    # Light grid lines on semi-transparent panes.
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.xaxis._axinfo["grid"]["color"] = (0.75, 0.75, 0.75, 1.0)
    ax.yaxis._axinfo["grid"]["color"] = (0.75, 0.75, 0.75, 1.0)
    ax.zaxis._axinfo["grid"]["color"] = (0.75, 0.75, 0.75, 1.0)
    ax.xaxis._axinfo["grid"]["linewidth"] = 0.5
    ax.yaxis._axinfo["grid"]["linewidth"] = 0.5
    ax.zaxis._axinfo["grid"]["linewidth"] = 0.5
    ax.grid(True)
    # Keep visible axes with light styling suitable for paper figures.
    ax.tick_params(labelsize=10, pad=1)
    ax.set_xlabel("$x_1$", fontsize=12, labelpad=4)
    ax.set_ylabel("$x_2$", fontsize=12, labelpad=4)
    ax.set_zlabel("$x_3$", fontsize=12, labelpad=4)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    zticks = np.linspace(ax.get_zlim()[0], ax.get_zlim()[1], 6)
    ax.set_zticks(zticks)
    rx = ax.get_xlim()[1] - ax.get_xlim()[0]
    ry = ax.get_ylim()[1] - ax.get_ylim()[0]
    rz = ax.get_zlim()[1] - ax.get_zlim()[0]
    ax.set_box_aspect((rx, ry, rz))
    fig1_path = Path(save_dir) / "torus_circle_3d.png"
    fig1.savefig(fig1_path, dpi=600, bbox_inches="tight")
    _crop_to_square(fig1_path)
    print(f"Saved figure: {fig1_path}")

    # View 2: side view rendered in 2D (x-z) so axes do not overlap.
    manifold_cpu = manifold.detach().cpu() if isinstance(
        manifold, torch.Tensor) else manifold
    samples_cpu = samples.detach().cpu() if isinstance(
        samples, torch.Tensor) else samples
    fig2, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        manifold_cpu[:, 0], manifold_cpu[:, 2],
        color="#1b6fab", s=2, alpha=0.5, label="Generated Samples"
    )
    ax.scatter(
        samples_cpu[:, 0], samples_cpu[:, 2],
        color="red", s=6, label="Training Samples"
    )
    ax.set_facecolor("white")
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_3$", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#E5E5E5", linewidth=0.5)
    legend = ax.legend(loc="upper left", fontsize=12, frameon=True)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.9)
    fig2_path = Path(save_dir) / "torus_circle_side.png"
    fig2.savefig(fig2_path, dpi=600, bbox_inches="tight")
    _crop_to_square(fig2_path)
    print(f"Saved figure: {fig2_path}")

    # View 3: top view rendered in 2D (x-y) to avoid 3D label overlap.
    fig3, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        manifold_cpu[:, 0], manifold_cpu[:, 1],
        color="#1b6fab", s=2, alpha=0.5, label="Generated Samples"
    )
    ax.scatter(
        samples_cpu[:, 0], samples_cpu[:, 1],
        color="red", s=6, label="Training Samples"
    )
    ax.set_facecolor("white")
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#E5E5E5", linewidth=0.5)
    legend = ax.legend(loc="upper right", fontsize=12, frameon=True)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.9)
    fig3_path = Path(save_dir) / "torus_circle_top.png"
    fig3.savefig(fig3_path, dpi=600, bbox_inches="tight")
    _crop_to_square(fig3_path)
    print(f"Saved figure: {fig3_path}")

    if show_plot:
        plt.figure(fig3.number)
        plt.show()

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)


def save_trained_models(ckpt_path, model, samples, M_delta, manifold, params,
                        final_mies):
    """Save model weights and a full experiment checkpoint.

    Saving happens before plotting so an interrupted plot window does not
    discard training results.
    """
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "samples": samples.cpu(),
            "M_delta": M_delta.cpu(),
            "manifold": manifold.cpu(),
            "model_final": model.state_dict(),
            "params": params,
            "final_mies": final_mies,
        },
        ckpt_path,
    )

    print(f"Saved checkpoint: {ckpt_path}")


def run_one_experiment(seed, base_params, save_dir, force_retrain=False):
    """Train or load one checkpointed experiment and return (samples, manifold)."""
    torch.manual_seed(seed)
    params = copy.deepcopy(base_params)
    params["seed"] = seed

    ckpt_path = Path(save_dir) / "checkpoint.pt"
    if ckpt_path.exists() and not force_retrain:
        print(f"Loading checkpoint from {ckpt_path} (skip training)")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Merge checkpoint params over current defaults so new keys from main
        # (such as num_points_per_sample_viz) are retained.
        ckpt_params = ckpt.get("params", {})
        if ckpt_params is None:
            ckpt_params = {}
        params = {**params, **ckpt_params}
        samples = ckpt["samples"].cpu()
        model = ProjectionNet(params["ambient_dim"]).cpu()
        model.load_state_dict(ckpt["model_final"])

        # Recompute samples so current visualization density is respected.
        viz_npps = params["num_points_per_sample_viz"]
        M_delta = tubular_nbh(
            samples,
            params["delta"],
            params["ambient_dim"],
            num_chosen_samples=None,
            num_points_per_sample=viz_npps,
            device="cpu", seed=seed + 20_000,
        )
        with torch.no_grad():
            manifold = apply_projection(
                model, M_delta, params["num_projection_applications"]
            )
        mie = distance_to_union(manifold, params).mean().item()
        print(f"Loaded single-delta model: MIE={mie:.6f}")
        return samples, manifold

    # Data
    N = params["N1"] + params["N2"]
    params["N"] = N
    samples = torus_and_circle_3d(
        N1=params["N1"], N2=params["N2"],
        R1=params["R1"], R2=params["R2"], Rc=params["Rc"]
    )
    avg_2nn, max_2nn = second_neighbor_stats(samples)

    print(
        f"Torus+Circle: N1={params['N1']}, N2={params['N2']}, "
        f"N={N}, ambient_dim={params['ambient_dim']}, delta={params['delta']}"
    )
    print(f"Average distance to 2nd nearest neighbor: {avg_2nn:.4f}")
    print(f"Maximum distance to 2nd nearest neighbor: {max_2nn:.4f}")

    # Model
    model = ProjectionNet(params["ambient_dim"]).to(device)
    model = train_torus_circle_model(model, samples, params)

    # Generate manifold samples for plotting (matches notebook expression)
    viz_npps = params["num_points_per_sample_viz"]
    M_delta = tubular_nbh(
        samples,
        params["delta"],
        params["ambient_dim"],
        num_chosen_samples=None,
        num_points_per_sample=viz_npps,
        device="cpu", seed=seed + 20_000,
    )

    model = model.cpu()
    with torch.no_grad():
        generated = {
            n: apply_projection(model, M_delta, n)
            for n in (1, 2, 4)
        }
    final_mies = {
        f"mie_{n}": distance_to_union(generated[n], params).mean().item()
        for n in (1, 2, 4)
    }
    manifold = generated[params["num_projection_applications"]]
    print(
        "Final exact MIE: "
        + ", ".join(f"x{n}={final_mies[f'mie_{n}']:.6f}" for n in (1, 2, 4))
    )

    save_trained_models(
        ckpt_path, model, samples, M_delta, manifold, params, final_mies,
    )
    return samples, manifold


if __name__ == "__main__":
    main()
