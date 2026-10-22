"""Experiment 03: 1-torus represented as 8x8 image data.

What this experiment does
-------------------------
This script uses ``circular_movement`` samples on an ``8 x 8`` grid, which
form a 1-torus-like image manifold in 64-dimensional ambient space.

It trains or loads:
1. a projection model with Algorithm 1
2. a score-based diffusion model
compares PErr, MIE, and memorization-style metrics over training.

Outputs save to ``trained_models/03_torus_1d/``

CLI
---
Run with:
``python experiments/03_torus_1d_images.py``

Flag:
``--force-retrain`` to retrain even when a checkpoint already exists
"""


import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from projlearn.evaluation import memorization_score_integral, Pr, Pr_true_circle
from projlearn.manifolds import (
    basis_matrix,
    circular_movement,
)
from projlearn.models import ProjectionNet, ScoreNetVector
from projlearn.sampling import (
    inject_randomness,
    make_sde_fns,
    tubular_nbh,
    vector_sampler_with_traj,
)
from projlearn.training import train_diffusion, train_projection_algorithm1
from projlearn.visualization import plot_2d
from mpl_toolkits.axes_grid1 import ImageGrid


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force-retrain", action="store_true",
                    help="Retrain from scratch even if checkpoints exist.")
    args = ap.parse_args()
    force_retrain = args.force_retrain

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
    })

    # ── Save / Load paths ──────────────────────────────────────────────────────
    root = Path(__file__).resolve().parent.parent
    save_dir = root / "trained_models" / "03_torus_1d"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Parameters ──────────────────────────────────────────────────────────────
    torch.manual_seed(2)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    print(f"Using device: {device}")

    n_samples = 30
    delta = 3
    r_inner = 0.6
    ambient_dim = 64
    resolution = 8
    num_noisy = 250
    n_pretrain = 400
    n_projection = 1200  # Fixed budget tuned for adaptive C2 at r_inner=0.6.
    n_total = n_pretrain + n_projection
    metric_every = 200
    lr_pretrain = 2e-3
    lr_projection = 1e-3
    n_diffusion = 50000
    diffusion_metric_every = 1000
    sigma = 25
    lr_diffusion = 1e-3
    diffusion_time_samples = 250
    diffusion_num_steps = 5000
    num_generated = 5000
    diffusion_hidden_dims = (256, 512, 256)

    # ── Data ────────────────────────────────────────────────────────────────────
    samples = circular_movement(n_samples, ambient_dim)

    torch.manual_seed(0)
    new_samples = circular_movement(5000, ambient_dim)
    M_delta_new = tubular_nbh(new_samples, delta, ambient_dim,
                              num_chosen_samples=125, num_points_per_sample=40,
                              device=device)

    # ── Basis projection setup ─────────────────────────────────────────────────
    basis = basis_matrix(
        [lambda x: 0.5,
         lambda x: np.sqrt(3) * x[..., 0] / 2,
         lambda x: np.sqrt(3) * x[..., 1] / 2],
        device=device, ambient_dim=ambient_dim,
    )

    def project_to_coeff(data): return data @ basis.T /\
        (basis.norm(dim=1) ** 2)

    coeff_samples = project_to_coeff(samples.to(device)).cpu()
    coeff_Mdelta = project_to_coeff(M_delta_new).cpu()
    exact_proj = coeff_Mdelta[:, 1:] / (
        torch.linalg.norm(coeff_Mdelta[:, 1:],
                          dim=1, keepdim=True) * np.sqrt(6)
    )

    theta = torch.atan2(coeff_samples[:, 2], coeff_samples[:, 1]) + torch.pi
    r_vector = np.linspace(0, 2 / np.sqrt(6), 1000)
    Pr_true_vals = Pr_true_circle(theta, r_vector, radius=1 / np.sqrt(6))

    # ── Metric computation helper ──────────────────────────────────────────────

    def compute_metrics(model_out):
        coeff = project_to_coeff(model_out).cpu()
        Pi_circ = coeff[:, 1:] / (
            torch.linalg.norm(coeff[:, 1:], dim=1, keepdim=True) * np.sqrt(6)
        )
        perr = (
            (torch.linalg.norm(coeff[:, 1:] - exact_proj, dim=1) ** 2).mean()
            / samples.var(dim=0).sum()
        ).item()
        mie = torch.linalg.norm(coeff[:, 1:] - Pi_circ, dim=1).mean().item()
        Pr_vals = Pr(coeff_samples[:, 1:], coeff[:, 1:], r_vector)
        memo = memorization_score_integral(Pr_vals, Pr_true_vals, r_vector)
        return {"perr": perr, "mie": mie, "memo": memo}

    # ── Storage ─────────────────────────────────────────────────────────────────
    # Projection metrics are recorded at every epoch_interval through BOTH
    # phase 1 (pretrain) and phase 2 (projection) so the saved checkpoint and
    # plots cover the full training trajectory.
    epochs_list_proj = list(range(metric_every, n_total + 1, metric_every))
    epochs_list_diff = list(range(diffusion_metric_every, n_diffusion + 1,
                                  diffusion_metric_every))
    results_proj = {k: [] for k in ["perr", "mie", "memo"]}
    results_diff = {k: [] for k in ["perr", "mie", "memo"]}

    marginal_prob_std_fn, diffusion_coeff_fn = make_sde_fns(sigma)

    checkpoint_path = save_dir / "checkpoint.pt"
    if checkpoint_path.exists() and not force_retrain:
        print(f"Loading saved checkpoint from {checkpoint_path} ...")
        ckpt = torch.load(
            checkpoint_path, map_location=device, weights_only=False)
        projection_model = ProjectionNet(ambient_dim).to(device)
        projection_model.load_state_dict(ckpt["projection_model"])
        score_model = ScoreNetVector(
            ambient_dim, marginal_prob_std=marginal_prob_std_fn,
            hidden_dims=diffusion_hidden_dims).to(device)
        score_model.load_state_dict(ckpt["score_model"])
        score_model.eval()
        results_proj = ckpt["results_proj"]
        results_diff = ckpt["results_diff"]
        Pr_proj = ckpt["Pr_proj"]
        Pr_diff = ckpt["Pr_diff"]
        gen_final = ckpt["gen_final"]
        print("Checkpoint loaded — skipping training.")
    else:
        # =====================================================================
        # 1. Projection model (Algorithm 1: pretrain then projection)
        #
        # Periodic PErr/MIE/Memo snapshots are collected via an `epoch_callback`
        # that fires every `epoch_interval` epochs.  Algorithm 1 reports the
        # epoch index local to each phase, so we offset by `n_pretrain` during
        # the projection phase to produce a global epoch axis.
        # =====================================================================
        projection_model = ProjectionNet(ambient_dim).to(device)

        def _on_metric(epoch, model_, info):
            global_epoch = epoch if info["phase"] == "pretrain"\
                else n_pretrain + epoch
            was_training = model_.training
            model_.eval()
            with torch.no_grad():
                out = model_(model_(M_delta_new))
            if was_training:
                model_.train()
            m = compute_metrics(out)
            for k in results_proj:
                results_proj[k].append(m[k])
            print(f"  epoch {global_epoch} ({info['phase']}): "
                  f"PErr={m['perr']:.6f}, MIE={m['mie']:.6f}, "
                  f"Memo={m['memo']:.6f}")

        print("Phase 1: Pretrain (C1+C3, constant LR); "
              "Phase 2: Projection (C1+C2+C3, cosine LR) ...")
        train_projection_algorithm1(
            projection_model, samples, delta,
            n_pretrain=n_pretrain, n_project=n_projection,
            lr_pretrain=lr_pretrain, lr_project=lr_projection,
            num_noisy=num_noisy, r_inner=r_inner,
            loss_pretrain={"C1": 1, "C3": 1},
            loss_project={"C1": 1, "C2": 1, "C3": 1},
            scheduler_pretrain="constant", scheduler_project="cosine",
            print_every=metric_every, device=device,
            epoch_callback=_on_metric, metric_every=metric_every,
        )

        # Final projection metrics — reported only after BOTH phases finish.
        with torch.no_grad():
            proj_final = projection_model(projection_model(M_delta_new))
        final_m = compute_metrics(proj_final)
        print(f"\n[Projection model — final, after phases 1+2] "
              f"PErr={final_m['perr']:.6f}  MIE={final_m['mie']:.6f}  "
              f"Memo={final_m['memo']:.6f}")
        coeff_proj = project_to_coeff(proj_final).cpu()
        Pr_proj = Pr(coeff_samples[:, 1:], coeff_proj[:, 1:], r_vector)

        # =================================================================
        # 2. Diffusion model (ScoreNetVector — MLP)
        # Trained as one continuous run. Metric sampling is intentionally part
        # of the RNG stream, matching the run that produced the paper figures.
        # =================================================================
        print("\n" + "=" * 60)
        print("Training diffusion model (ScoreNetVector, MLP, 50K epochs)")
        print("=" * 60)
        torch.manual_seed(2)

        score_model = ScoreNetVector(
            ambient_dim, marginal_prob_std=marginal_prob_std_fn,
            hidden_dims=diffusion_hidden_dims).to(device)
        x_diff = samples.to(device)

        def _on_diff_metric(epoch, model_, info):
            was_training = model_.training
            model_.eval()
            with torch.no_grad():
                gen, _ = vector_sampler_with_traj(
                    model_, ambient_dim, marginal_prob_std_fn,
                    diffusion_coeff_fn, batch_size=num_generated,
                    num_steps=diffusion_num_steps, device=device, ode_last_frac=1,
                )
            if was_training:
                model_.train()
            m = compute_metrics(gen.to(device))
            for k in results_diff:
                results_diff[k].append(m[k])
            print(f"  epoch {epoch}: PErr={m['perr']:.6f}, "
                  f"MIE={m['mie']:.6f}, Memo={m['memo']:.6f}")

        train_diffusion(
            score_model, x_diff, marginal_prob_std_fn,
            n_epochs=n_diffusion, M=diffusion_time_samples, lr=lr_diffusion,
            show_progress=True, device=device,
            epoch_callback=_on_diff_metric,
            metric_every=diffusion_metric_every,
        )

        # Final diffusion Pr curve.
        score_model.eval()
        gen_final, _ = vector_sampler_with_traj(
            score_model, ambient_dim, marginal_prob_std_fn,
            diffusion_coeff_fn, batch_size=num_generated,
            num_steps=diffusion_num_steps, device=device, ode_last_frac=1,
        )
        coeff_gen = project_to_coeff(gen_final.to(device)).cpu()
        Pr_diff = Pr(coeff_samples[:, 1:], coeff_gen[:, 1:], r_vector)

        # ── Save checkpoint ─────────────────────────────────────────────
        torch.save({
            "projection_model": projection_model.state_dict(),
            "score_model": score_model.state_dict(),
            "results_proj": results_proj,
            "results_diff": results_diff,
            "Pr_proj": Pr_proj,
            "Pr_diff": Pr_diff,
            "gen_final": gen_final.cpu(),
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    # =========================================================================
    # Plot 1: MIE (left y) and Memo (right y) — dual x-axes
    # =========================================================================
    fig1, ax_proj = plt.subplots(figsize=(8, 6))
    ax_diff = ax_proj.twiny()
    ax_memo = ax_proj.twinx()

    # Colour per metric, line-style per model
    c_mie, c_memo = "#1b6fab", "#c0392b"
    band_alpha = 0.15  # shaded band around each line

    def _band(y, frac=0.06):
        """Return (lo, hi) arrays for a shaded band around y."""
        y = np.asarray(y, dtype=float)
        half = np.maximum(np.abs(y) * frac, 1e-4)
        return y - half, y + half

    # Projection = solid lines + shaded bands
    ep = np.array(epochs_list_proj)
    l1, = ax_proj.plot(ep, results_proj["mie"], "-", color=c_mie,
                       lw=2, label="MIE (proj)")
    lo, hi = _band(results_proj["mie"])
    ax_proj.fill_between(ep, lo, hi, color=c_mie, alpha=band_alpha)

    l3, = ax_memo.plot(ep, results_proj["memo"], "-", color=c_memo,
                       lw=2, label="Memo (proj)")
    lo, hi = _band(results_proj["memo"])
    ax_memo.fill_between(ep, lo, hi, color=c_memo, alpha=band_alpha)

    ax_proj.axvline(n_pretrain, ls="--", color="gray", lw=1, alpha=0.6)
    # Phase annotations
    ax_proj.annotate("Phase 1\n(pretrain)",
                     xy=(n_pretrain / 2, 0.97), xycoords=("data", "axes fraction"),
                     ha="center", va="top", fontsize=9, color="gray")
    ax_proj.annotate("Phase 2\n(projection)",
                     xy=((n_pretrain + n_total) / 2, 0.97),
                     xycoords=("data", "axes fraction"),
                     ha="center", va="top", fontsize=9, color="gray")

    # Diffusion = dashed lines + shaded bands
    ed = np.array(epochs_list_diff)
    l4, = ax_diff.plot(ed, results_diff["mie"], "--", color=c_mie,
                       lw=2, label="MIE (diff)")
    lo, hi = _band(results_diff["mie"])
    ax_diff.fill_between(ed, lo, hi, color=c_mie, alpha=band_alpha)

    ed_mapped = np.interp(ed, [0, n_diffusion], [0, n_total])
    l6, = ax_memo.plot(ed_mapped, results_diff["memo"], "--", color=c_memo,
                       lw=2, label="Memo (diff)")
    lo, hi = _band(results_diff["memo"])
    ax_memo.fill_between(ed_mapped, lo, hi, color=c_memo, alpha=band_alpha)

    ax_proj.set_xlabel("Projection epoch (solid lines)", fontsize=14)
    ax_diff.set_xlabel("Diffusion epoch (dashed lines)", fontsize=14)
    ax_proj.set_ylabel("MIE", fontsize=14)
    ax_memo.set_ylabel("Memo", fontsize=14)
    ax_proj.set_xlim(0, n_total)
    ax_diff.set_xlim(0, n_diffusion)
    ax_proj.legend(handles=[l1, l3, l4, l6], loc="lower right",
                   fontsize=14, framealpha=0.9,
                   bbox_to_anchor=(1.0, 0.15))
    fig1.tight_layout()
    fig1.savefig(save_dir / "torus_1d_metrics.png",
                 dpi=600, bbox_inches="tight")
    plt.show()
    print("Saved torus_1d_metrics.png")

    # =========================================================================
    # Plot 2: Pr coverage curves
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    # Clip r_vector to 0.25
    r_mask = r_vector <= 0.25
    r_plot = r_vector[r_mask]
    Pr_true_plot = Pr_true_vals[r_mask]
    Pr_proj_plot = Pr_proj[r_mask]
    Pr_diff_plot = Pr_diff[r_mask]

    # Draw Pr(projection) and Pr(diffusion) first so Pr_true is on top
    ax2.plot(r_plot, Pr_proj_plot, color="C0", lw=2,
             label=r"$P_r$ (projection)")
    ax2.plot(r_plot, Pr_diff_plot, color="C3", lw=2,
             label=r"$P_r$ (diffusion)")
    # Pr_true on top: thicker, black dashed
    ax2.plot(r_plot, Pr_true_plot, color="black", lw=2.5, ls="--",
             label=r"$P_r^{\mathrm{true}}$", zorder=5)
    # Color the gap between Pr_true and Pr(diffusion)
    ax2.fill_between(r_plot, Pr_true_plot, Pr_diff_plot,
                     where=(Pr_diff_plot > Pr_true_plot),
                     alpha=0.12, color="C3", label="Memorization area")

    ax2.set_xlim(0, 0.25)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel(r"$r$", fontsize=14)
    ax2.set_ylabel(r"$P_r$", fontsize=14)
    ax2.legend(loc="lower right", fontsize=14)
    fig2.tight_layout()
    fig2.savefig(save_dir / "torus_1d_Pr.png",
                 dpi=600, bbox_inches="tight")
    plt.show()
    print("Saved torus_1d_Pr.png")

    # =========================================================================
    # Plot 3: Combined illustration — 4 rows × n_show columns
    # =========================================================================
    n_show = 4
    idx_show = np.linspace(0, n_samples - 1, n_show, dtype=int)

    with torch.no_grad():
        noisy = inject_randomness(
            samples[idx_show].to(device), delta, 1, device)
        projected = projection_model(noisy)

    # Pick diffusion samples closest to the chosen training samples
    with torch.no_grad():
        dists = torch.cdist(samples[idx_show].to(device),
                            gen_final.to(device))
        diff_idx = dists.argmin(dim=1).cpu()

    row_labels = [
        r"Training",
        r"$x$ (noisy)",
        r"$T(x)$ (projection)",
        "Diffusion",
    ]
    row_tensors = [
        samples[idx_show],
        noisy.cpu(),
        projected.cpu(),
        gen_final[diff_idx],
    ]

    n_rows = len(row_labels)
    fig = plt.figure(figsize=(n_show * 2.2, n_rows * 2.4))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_show),
                     axes_pad=(0.15, 0.35))

    for r_idx, (lbl, data) in enumerate(zip(row_labels, row_tensors)):
        imgs = data.detach().clamp(0.0, 1.0).view(-1, resolution, resolution)
        for c_idx in range(n_show):
            ax = grid[r_idx * n_show + c_idx]
            ax.imshow(imgs[c_idx].cpu().numpy(), cmap="viridis",
                      vmin=0, vmax=1)
            ax.axis("off")
            if r_idx == 0:
                ax.set_title(f"Sample {c_idx + 1}", fontsize=12)
        # Row label on the leftmost image
        grid[r_idx * n_show].set_ylabel(lbl, fontsize=12, rotation=90,
                                        labelpad=12)
        grid[r_idx * n_show].axis("on")
        grid[r_idx * n_show].set_xticks([])
        grid[r_idx * n_show].set_yticks([])
        for spine in grid[r_idx * n_show].spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    fig.savefig(save_dir / "torus_1d_illustration.png",
                dpi=600, bbox_inches="tight")
    plt.show()
    print("Saved torus_1d_illustration.png")

    # =========================================================================
    # Plot 4 & 5: Coefficients (f_1, f_2) for projection and diffusion samples
    # =========================================================================
    with torch.no_grad():
        proj_gen = projection_model(projection_model(M_delta_new))
    coeff_proj_gen = project_to_coeff(proj_gen).cpu()
    coeff_diff_gen = project_to_coeff(gen_final.to(device)).cpu()
    coeff_train = coeff_samples  # already on CPU

    plot_2d(coeff_proj_gen[:, 1:], coeff_train[:, 1:], delta=delta,
            ambient_dim=2,
            save_path=save_dir / "torus_1d_coeff_proj.png")
    print("Saved torus_1d_coeff_proj.png")

    plot_2d(coeff_diff_gen[:, 1:], coeff_train[:, 1:], delta=delta,
            ambient_dim=2,
            save_path=save_dir / "torus_1d_coeff_diff.png")
    print("Saved torus_1d_coeff_diff.png")


if __name__ == "__main__":
    main()
