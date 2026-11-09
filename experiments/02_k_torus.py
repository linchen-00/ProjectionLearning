"""Experiment 02: k-torus sample complexity across dimensions.

What this experiment does
-------------------------
Part A:
For each torus dimension ``d in {1, 2, 3, 4, 5}``, it works in ambient
dimension ``2d`` with fixed ``delta = 1`` and searches for the smallest sample
size ``n*`` such that MIE< 0.02.

Part B:
Using the ``n*`` values from Part A, it trains projection models for the same
torus dimensions embedded in ``R^256`` and compares two noise scales:
``delta = 1`` and ``delta = 16 / sqrt(d)``.

Outputs save to ``trained_models/02_k_torus/``

CLI
---
Run with:
``python experiments/02_k_torus.py``

Flag:
``--force-train`` to ignore cached checkpoints and retrain from scratch
``--evaluate-checkpoints`` to evaluate the saved native and scaled-256D
models without training or replacing the historical paper result caches.
"""

import argparse
import csv
import gc
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from projlearn.evaluation import dist_torus_embedded
from projlearn.manifolds import torus, torus_embedded
from projlearn.models import ProjectionNet
from projlearn.sampling import tubular_nbh
from projlearn.training import train_projection_algorithm1

# ── Paths ────────────────────────────────────────────────────────────────────
root = Path(__file__).resolve().parent.parent
save_dir = root / "trained_models" / "02_k_torus"

# ── Device ───────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# ── Experiment parameters ────────────────────────────────────────────────────
torus_dims = [1, 2, 3, 4, 5]
mie_threshold = 0.02
growth_factor = 1.2
n_step = 30
max_attempts = 50
n_pretrain = 500
n_projection = 1500
lr_pretrain = 5e-3
lr_projection = 1e-3
batch_size = 512
num_noisy = 100

# ── Plot style ───────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":     "serif",
    "font.size":       12,
    "axes.labelsize":  15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "lines.linewidth": 1.8,
    "lines.markersize": 7,
})


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force-train", action="store_true",
                    help="Retrain from scratch even if checkpoints exist.")
    ap.add_argument("--evaluate-checkpoints", action="store_true",
                    help="Evaluate saved native/scaled models without training; "
                         "leave historical result caches unchanged.")
    args = ap.parse_args()
    if args.force_train and args.evaluate_checkpoints:
        ap.error("--force-train and --evaluate-checkpoints cannot be combined")
    return args


# ── Training helper ───────────────────────────────────────────────────────────
def train_torus_model(n, delta, ambient_dim, sampler=None, use_dataloader=False):
    """Train a ProjectionNet on a torus manifold and return (samples, manifold, model)."""
    torch.manual_seed(3)
    samples = sampler(n) if sampler is not None else torus(
        n, ambient_dim=ambient_dim)
    model = ProjectionNet(ambient_dim).to(device)

    train_projection_algorithm1(
        model, samples, delta,
        n_pretrain=n_pretrain, n_project=n_projection,
        lr_pretrain=lr_pretrain, lr_project=lr_projection,
        num_noisy=num_noisy, r_inner=delta / 2,
        loss_pretrain={"C1": 1, "C3": 1},
        loss_project={"C1": 1, "C2": 0.5, "C3": 0.5},
        scheduler_pretrain="constant", scheduler_project="constant",
        batch_size=batch_size, use_dataloader=use_dataloader,
        c2_balance=1.0,  # Paper protocol: original C2, fixed s=1.
        show_progress=True, verbose=False, device=device,
    )

    M_delta = tubular_nbh(samples, delta, ambient_dim,
                          num_points_per_sample=num_noisy)
    model_cpu = model.cpu()
    with torch.no_grad():
        manifold = model_cpu(model_cpu(M_delta))
    return samples.cpu(), manifold.cpu(), model_cpu


def load_torus_checkpoint(d, *, embedded=False):
    """Load saved weights and anchors; never train or substitute metric caches.

    The scaled checkpoints are replacement runs at the paper settings,
    not the unavailable weights that produced the historical paper numbers.
    """
    if embedded:
        path = save_dir / f"torus_scaled_d{d}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = (checkpoint["model_state_dict"] if "model_state_dict" in checkpoint
                 else checkpoint["state_dict"])
        samples = checkpoint["samples"].cpu()
        ambient = 256
    else:
        path = save_dir / f"torus_d{d}.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        progress = torch.load(save_dir / "checkpoint.pt", map_location="cpu",
                              weights_only=False)
        n = progress["results_min_n"][torus_dims.index(d)]
        # Original native checkpoints contain bare weights. Their anchors
        # are regenerated using the original data sampler and seed.
        with torch.random.fork_rng():
            torch.manual_seed(3)
            samples = torus(n, ambient_dim=2*d).cpu()
        ambient = 2*d
    model = ProjectionNet(ambient)
    model.load_state_dict(state, strict=True)
    return model.eval(), samples


@torch.no_grad()
def evaluate_checkpoints():
    """Fresh MIE from saved weights, separate from historical paper metrics."""
    rows = []
    for d in torus_dims:
        row = {"d": d}
        for embedded, label in ((False, "native"), (True, "scaled256")):
            model, samples = load_torus_checkpoint(d, embedded=embedded)
            ambient = 256 if embedded else 2*d
            delta = 16/math.sqrt(d) if embedded else 1.0
            noisy = tubular_nbh(samples, delta, ambient,
                                num_points_per_sample=num_noisy, seed=3)
            generated = model(model(noisy))
            mie, _ = dist_torus_embedded(
                generated, torus_dim=d, ambient_dim=ambient,
                threshold=mie_threshold)
            row[label] = float(mie)
        rows.append(row)
    print("Fresh saved-model evaluation (two applications of T).")
    print("Scaled models are replacement runs; these are not the historical paper MIEs.")
    print(" d       native MIE    scaled-256D MIE")
    for row in rows:
        print(f" {row['d']}       {row['native']:.6f}          {row['scaled256']:.6f}")
    return rows


def save_embedded_checkpoint(d, samples, model, delta, *, scaled):
    """Keep embedded weights as well as metrics for future runs."""
    name = f"torus_{'scaled' if scaled else 'delta1'}_d{d}.pt"
    torch.save({
        "model_state_dict": model.state_dict(), "samples": samples.cpu(),
        "config": {"d": d, "n": len(samples), "ambient_dim": 256,
                   "delta": delta, "r_inner": delta/2, "seed": 3,
                   "c2_balance": 1.0, "n_pretrain": n_pretrain,
                   "n_projection": n_projection, "lr_pretrain": lr_pretrain,
                   "lr_projection": lr_projection, "num_noisy": num_noisy,
                   "batch_size": batch_size, "evaluation_applications": 2},
    }, save_dir / name)


# ═════════════════════════════════════════════════════════════════════════════
# PART A — minimal n* search
# ═════════════════════════════════════════════════════════════════════════════

def run_partA(force_train: bool):
    print("\n" + "=" * 70)
    print("PART A — n* search, delta=1, ambient=2d")
    print("=" * 70)
    print(f"Using device: {device}")

    checkpoint_path = save_dir / "checkpoint.pt"

    # ── Load path ─────────────────────────────────────────────────────────────
    if checkpoint_path.exists() and not force_train:
        ckpt = torch.load(checkpoint_path, map_location="cpu",
                          weights_only=False)
        results_min_n = ckpt["results_min_n"]
        all_trials = ckpt["all_trials"]
        completed_d = ckpt["completed_d"]
        next_start_n = ckpt["next_start_n"]
        print(f"Loaded checkpoint from {checkpoint_path.relative_to(root)}")

        missing = [d for d in torus_dims if d not in completed_d]
        if not missing:
            print(f"  All dimensions completed: {completed_d}")
            print(f"  n* values: {results_min_n}")
            return results_min_n, all_trials
    else:
        results_min_n = []
        all_trials = []
        completed_d = []
        next_start_n = n_step
        if force_train:
            print("--force-train: starting Part A from scratch")

    # ── Train loop ────────────────────────────────────────────────────────────
    for d in torus_dims:
        if d in completed_d:
            print(f"\n[d={d}] already completed, skipping")
            continue

        ambient_dim = 2 * d
        delta_d = 1
        n = max(next_start_n, n_step)
        found = None

        print(f"\n[d={d}] delta={delta_d} (fixed), ambient={ambient_dim}, "
              f"starting at n={n}")

        for attempt in range(max_attempts):
            use_dl = ambient_dim >= 8
            samples, manifold, model = train_torus_model(
                n, delta_d, ambient_dim, use_dataloader=use_dl)
            mie_val, is_good = dist_torus_embedded(
                manifold, torus_dim=d, ambient_dim=ambient_dim,
                threshold=mie_threshold)
            all_trials.append((d, n, mie_val))
            print(f"  attempt {attempt+1}: n={n}  MIE={mie_val:.6f}  "
                  f"{'PASS' if is_good else 'FAIL'}")

            if is_good:
                found = n
                torch.save(model.state_dict(),
                           save_dir / f"torus_d{d}.pt")
                del samples, manifold, model
                break

            n = max(n + n_step, int(math.ceil(n * growth_factor)))
            del samples, manifold, model
            gc.collect()

        results_min_n.append(found)
        next_start_n = found if found is not None else n
        completed_d.append(d)
        print(f"  => min n = {found}")

        torch.save({
            "results_min_n": results_min_n,
            "all_trials":    all_trials,
            "next_start_n":  next_start_n,
            "completed_d":   completed_d,
        }, checkpoint_path)
        print(f"  Checkpoint saved "
              f"({len(completed_d)}/{len(torus_dims)} dims done)")

    print(f"\nPart A complete — n* values: {results_min_n}")
    return results_min_n, all_trials


def plot_partA(results_min_n: list, all_trials: list) -> None:
    """Produce k_torus_n_vs_dim_mie.png and save trial_data.csv."""
    d_arr = np.array(torus_dims, dtype=float)
    n_vals = np.array(
        [v if v is not None else np.nan for v in results_min_n], dtype=float)
    mask = np.isfinite(n_vals)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.semilogy(d_arr[mask], n_vals[mask], "o-", color="#1f77b4",
                label=rf"$\mathrm{{MIE}} \leq {mie_threshold}$")

    if mask.sum() >= 2:
        C_fit, b_fit = np.polyfit(d_arr[mask], np.log(n_vals[mask]), 1)
        x_ref = np.linspace(d_arr[mask].min(), d_arr[mask].max(), 200)
        y_ref = np.exp(b_fit + np.log(2)) * np.exp(C_fit * x_ref)
        ax.semilogy(x_ref, y_ref, "--", color="#c0392b",
                    label=rf"$n \propto \exp({C_fit:.2f}\,d)$")

    ax.set_xlabel(r"Intrinsic dimension $d$")
    ax.set_ylabel(r"Number of training samples $n$")
    ax.set_xticks(torus_dims)
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.10, top=0.96)

    out = save_dir / "k_torus_n_vs_dim_mie.png"
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print(f"Saved {out.relative_to(root)}")

    # Trial data CSV
    csv_path = save_dir / "trial_data.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["d", "n", "mie"])
        w.writerows(all_trials)
    print(f"Saved {csv_path.relative_to(root)}")


# ═════════════════════════════════════════════════════════════════════════════
# PART B — ambient-256 comparison
# ═════════════════════════════════════════════════════════════════════════════

def run_partB(results_min_n: list, force_train: bool) -> dict:
    print("\n" + "=" * 70)
    print("PART B — ambient-256: delta=1 vs delta=16/sqrt(d)")
    print("=" * 70)

    ambient_256 = 256
    partb_ckpt_path = save_dir / "partb_checkpoint.pt"

    # ── Load path ─────────────────────────────────────────────────────────────
    if partb_ckpt_path.exists() and not force_train:
        ckpt = torch.load(partb_ckpt_path, map_location="cpu",
                          weights_only=False)
        partb_results = ckpt["partb_results"]
        partb_completed = ckpt["partb_completed_d"]
        print(f"Loaded checkpoint from {partb_ckpt_path.relative_to(root)}")

        missing = [d for d in torus_dims if d not in partb_completed]
        if not missing:
            print("Reporting historical cached metrics. Use --evaluate-checkpoints "
                  "for fresh MIE from the saved replacement scaled models.")
            _print_partB_table(partb_results, results_min_n)
            _save_partB_artifacts(partb_results)
            return partb_results
    else:
        partb_results = {}
        partb_completed = []
        if force_train:
            print("--force-train: starting Part B from scratch")

    # ── Train loop ────────────────────────────────────────────────────────────
    def make_sampler(torus_d, amb):
        def sampler(n):
            return torus_embedded(n, torus_dim=torus_d, ambient_dim=amb)
        return sampler

    for i, d in enumerate(torus_dims):
        n_star = results_min_n[i]
        if n_star is None:
            print(f"\n[Part B d={d}] No n* from Part A, skipping.")
            continue
        if d in partb_completed:
            r = partb_results[d]
            print(f"\n[Part B d={d}] already completed — "
                  f"delta=1 amb MIE={r['delta1_amb_mie']:.6f}, "
                  f"delta=scaled amb MIE={r['delta_scaled_amb_mie']:.6f}")
            continue

        sampler_256 = make_sampler(d, ambient_256)
        delta_scaled = 16.0 / math.sqrt(d)
        print(f"\n[Part B d={d}] n*={n_star}, ambient=256")

        # ── delta = 1 ─────────────────────────────────────────────────────────
        print("  Training delta=1 ...")
        samples_1, manifold_1, model_1 = train_torus_model(
            n_star, 1.0, ambient_256, sampler=sampler_256, use_dataloader=True)
        save_embedded_checkpoint(d, samples_1, model_1, 1.0, scaled=False)
        d1_amb_mie, d1_amb_pass = dist_torus_embedded(
            manifold_1, torus_dim=d, ambient_dim=ambient_256,
            threshold=mie_threshold)
        d1_emb_mie, _ = dist_torus_embedded(
            manifold_1[:, :2*d], torus_dim=d, ambient_dim=2*d,
            threshold=mie_threshold)
        print(f"    amb MIE={d1_amb_mie:.6f} ({'PASS' if d1_amb_pass else 'FAIL'}), "
              f"emb MIE={d1_emb_mie:.6f}")
        del manifold_1, samples_1, model_1
        gc.collect()

        # ── delta = 16/sqrt(d) ────────────────────────────────────────────────
        print(f"  Training delta=16/sqrt({d})={delta_scaled:.4f} ...")
        samples_2, manifold_2, model_2 = train_torus_model(
            n_star, delta_scaled, ambient_256,
            sampler=sampler_256, use_dataloader=True)
        save_embedded_checkpoint(d, samples_2, model_2, delta_scaled, scaled=True)
        ds_amb_mie, ds_amb_pass = dist_torus_embedded(
            manifold_2, torus_dim=d, ambient_dim=ambient_256,
            threshold=mie_threshold)
        ds_emb_mie, _ = dist_torus_embedded(
            manifold_2[:, :2*d], torus_dim=d, ambient_dim=2*d,
            threshold=mie_threshold)
        print(f"    amb MIE={ds_amb_mie:.6f} ({'PASS' if ds_amb_pass else 'FAIL'}), "
              f"emb MIE={ds_emb_mie:.6f}")
        del manifold_2, samples_2, model_2
        gc.collect()

        partb_results[d] = {
            "n_star":                n_star,
            "delta1":                1.0,
            "delta1_amb_mie":        d1_amb_mie,
            "delta1_emb_mie":        d1_emb_mie,
            "delta1_amb_pass":       bool(d1_amb_pass),
            "delta_scaled":          delta_scaled,
            "delta_scaled_amb_mie":  ds_amb_mie,
            "delta_scaled_emb_mie":  ds_emb_mie,
            "delta_scaled_amb_pass": bool(ds_amb_pass),
        }
        partb_completed.append(d)

        torch.save({
            "partb_results":     partb_results,
            "partb_completed_d": partb_completed,
        }, partb_ckpt_path)
        print(f"  Part B checkpoint saved "
              f"({len(partb_completed)}/{len(torus_dims)} dims done)")

    _print_partB_table(partb_results, results_min_n)
    _save_partB_artifacts(partb_results)


def _print_partB_table(partb_results: dict, results_min_n: list) -> None:
    print(f"\n{'d':>3} {'n*':>6} {'delta_sc':>9}  "
          f"{'amb_MIE(d=1)':>13} {'emb_MIE(d=1)':>13}  "
          f"{'amb_MIE(sc)':>12} {'emb_MIE(sc)':>12}")
    print("-" * 80)
    for i, d in enumerate(torus_dims):
        if d not in partb_results:
            continue
        r = partb_results[d]
        print(f"  {d:>1}  {results_min_n[i]:>6}  {r['delta_scaled']:>9.4f}  "
              f"  {r['delta1_amb_mie']:>11.6f}  {r['delta1_emb_mie']:>11.6f}  "
              f"  {r['delta_scaled_amb_mie']:>10.6f}  {r['delta_scaled_emb_mie']:>10.6f}")


def _save_partB_artifacts(partb_results: dict) -> None:
    json_path = save_dir / "partb_results.json"
    with open(json_path, "w") as f:
        json.dump({str(d): partb_results[d]
                  for d in partb_results}, f, indent=2)
    print(f"\nSaved {json_path.relative_to(root)}")

    header = ["d", "n_star", "delta1",
              "delta1_amb_mie", "delta1_emb_mie", "delta1_amb_pass",
              "delta_scaled",
              "delta_scaled_amb_mie", "delta_scaled_emb_mie", "delta_scaled_amb_pass"]
    rows = []
    for d in torus_dims:
        if d not in partb_results:
            continue
        r = partb_results[d]
        rows.append([d, r["n_star"], r["delta1"],
                     r["delta1_amb_mie"], r["delta1_emb_mie"], r["delta1_amb_pass"],
                     r["delta_scaled"],
                     r["delta_scaled_amb_mie"], r["delta_scaled_emb_mie"],
                     r["delta_scaled_amb_pass"]])
    csv_path = save_dir / "partb_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Saved {csv_path.relative_to(root)}")

# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    args = parse_args()
    if args.evaluate_checkpoints:
        return evaluate_checkpoints()
    save_dir.mkdir(parents=True, exist_ok=True)

    results_min_n, all_trials = run_partA(args.force_train)
    plot_partA(results_min_n, all_trials)

    run_partB(results_min_n, args.force_train)

    print("\n" + "=" * 70)
    print("Experiment complete.")
    print(f"  Part A n* values : {results_min_n}")
    print(f"  Outputs in       : {save_dir.relative_to(root)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
