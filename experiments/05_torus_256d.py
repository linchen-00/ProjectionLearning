"""Experiment 05: 2-torus embedded in R^256.

What this experiment does
-------------------------
This script samples a 2-torus embedded in ``R^256`` and trains one projection
model per seed with Algorithm 1. It uses one tubular radius, one inner radius,
and equal C1/C2/C3 weights in the projection phase.

For each seed, it saves:
1. the trained projection checkpoint
2. MIE and PErr-style metrics
3. 3D coordinate views and recovered angular-coordinate plots

When ``wasserstein.pt`` is present, the script also loads and reports the
saved per-seed Wasserstein distances.

Outputs save to ``trained_models/05_torus_256d/``

CLI
---
Run with:
``python experiments/05_torus_256d.py``

Flags:
``--force-retrain`` to retrain even when checkpoints already exist
``--force-recompute-wasserstein`` to recompute bounded Wasserstein distances
"""

import argparse
import copy
import itertools
import time
from pathlib import Path

import psutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from projlearn.evaluation import projection_error
from projlearn.manifolds import (
    angles_torus_embedded,
    dist_torus_embedded,
    proj_torus_embedded,
    torus_embedded,
)
from projlearn.models import ProjectionNet
from projlearn.sampling import tubular_nbh
from projlearn.training import train_projection_algorithm1
from projlearn.utils import second_neighbor_stats
from projlearn.visualization import plot_3d, plot_3d_with_lines


ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = ROOT / "trained_models" / "05_torus_256d"
WASSERSTEIN_METRICS = [
    "W(orig,orig)",
    "W(true,true)",
    "W(gen,true)",
    "W(orig,true)",
    "W(gen,orig)",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[2, 3, 4, 5, 6])
    parser.add_argument("--n-pretrain", type=int, default=400)
    parser.add_argument("--n-projection", type=int, default=2000)
    parser.add_argument("--lr-pretrain", type=float, default=5e-3)
    parser.add_argument("--lr-projection", type=float, default=1e-3)
    parser.add_argument("--r-inner", type=float, default=1.75)
    parser.add_argument("--applications", type=int, default=4)
    parser.add_argument(
        "--run-tag", default="",
        help="Save an isolated tuning run in a subdirectory.",
    )
    parser.add_argument(
        "--force-recompute-wasserstein", action="store_true",
        help=(
            "Recompute Wasserstein distances with a bounded subsample. "
            "By default the script only loads wasserstein.pt if it exists."
        ),
    )
    parser.add_argument(
        "--wasserstein-subsample", type=int, default=5000,
        help=(
            "Maximum generated/true points per set for local Wasserstein "
            "recomputation. Original sample sets remain at N=480."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = SAVE_DIR / args.run_tag if args.run_tag else SAVE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / "output.txt"
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")

    common_params = {
        "N": 480,
        "delta": 7,
        "r_inner": args.r_inner,
        "torus_dim": 2,
        "ambient_dim": 256,
        "num_noisy": 50,
        "n_pretrain": args.n_pretrain,
        "n_projection": args.n_projection,
        "lr_pretrain": args.lr_pretrain,
        "lr_projection": args.lr_projection,
        "loss_pretrain": {"C1": 1, "C3": 1},
        "loss_projection": {"C1": 1, "C2": 1, "C3": 1},
        "applications": args.applications,
    }

    lines = [
        f"device={device}",
        f"seeds={args.seeds}",
        f"parameters={common_params}",
    ]
    metric_keys = [
        "mie", "perr", "projection_error",
        "median_displacement", "median_displacement_first4",
    ]
    all_metrics = {key: [] for key in metric_keys}
    generated_by_seed = {}
    samples_by_seed = {}

    for seed in args.seeds:
        params = {"seed": seed, **common_params}
        checkpoint_path = save_dir / f"seed{seed}_ckpt.pt"
        if checkpoint_path.exists() and not args.force_retrain:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            samples = checkpoint["samples"]
            model = checkpoint["model_final"].to(device).eval()
            params = checkpoint["params"]
            print(f"Loaded checkpoint: {checkpoint_path}")
        else:
            torch.manual_seed(seed)
            samples = torus_embedded(
                params["N"], torus_dim=params["torus_dim"],
                ambient_dim=params["ambient_dim"], density_params=[0.0, 0.5],
            )
            avg_2nn, max_2nn = second_neighbor_stats(samples)
            model = ProjectionNet(params["ambient_dim"]).to(device)
            started = time.perf_counter()
            result = train_projection_algorithm1(
                model, samples, params["delta"],
                n_pretrain=params["n_pretrain"],
                n_project=params["n_projection"],
                lr_pretrain=params["lr_pretrain"],
                lr_project=params["lr_projection"],
                num_noisy=params["num_noisy"],
                r_inner=params["r_inner"],
                loss_pretrain=params["loss_pretrain"],
                loss_project=params["loss_projection"],
                scheduler_pretrain="constant", scheduler_project="cosine",
                print_every=100, device=device,
            )
            checkpoint = {
                "seed": seed,
                "samples": samples.cpu(),
                "model_pretrain": copy.deepcopy(
                    result["pretrain_snapshot"]
                ).cpu(),
                "model_final": copy.deepcopy(model).cpu(),
                "params": params,
                "wall_time": time.perf_counter() - started,
                "avg_2nn": avg_2nn,
                "max_2nn": max_2nn,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        M_delta = tubular_nbh(
            samples, params["delta"], params["ambient_dim"],
            num_chosen_samples=None, num_points_per_sample=50,
            device=device, seed=seed,
        )
        chosen = apply_projection(
            model, M_delta, params["applications"]
        ).cpu()
        generated_by_seed[seed] = chosen.cpu()
        samples_by_seed[seed] = samples.cpu()
        values = metrics_for(chosen, M_delta.cpu(), samples.cpu())
        for key in metric_keys:
            all_metrics[key].append(values[key])
        line = (
            f"Seed {seed}: MIE={values['mie']:.6f} "
            f"PErr={values['perr']:.6f} "
            f"ProjErr={values['projection_error']:.6f} "
            f"Med|T-x|={values['median_displacement']:.6f} "
            f"Med|T-x|_4={values['median_displacement_first4']:.6f}"
        )
        print(line)
        lines.append(line)
        save_figures(chosen, M_delta.cpu(), samples.cpu(), save_dir, seed)

    labels = {
        "mie": "MIE (avg dist to torus)",
        "perr": "PErr (normalized proj error)",
        "projection_error": "Proj Error (avg |Pi(x)-T(x)|)",
        "median_displacement": "Median |T(x)-x|",
        "median_displacement_first4": "Median |T(x)-x| (coords 0-3)",
    }
    lines.append("Averages over all seeds:")
    for key in metric_keys:
        mean = float(np.mean(all_metrics[key]))
        std = float(np.std(all_metrics[key]))
        lines.append(f"  {labels[key]}: {mean:.6f} (std={std:.6f})")
        print(lines[-1])

    wass_ckpt_path = save_dir / "wasserstein.pt"
    if wass_ckpt_path.exists() and not args.force_recompute_wasserstein:
        msg = f"\nLoading Wasserstein results from {wass_ckpt_path}"
        print(msg)
        lines.append(msg)
        wass_data = torch.load(
            wass_ckpt_path, map_location="cpu", weights_only=False
        )
        add_wasserstein_summary(wass_data, lines)
    elif args.force_recompute_wasserstein:
        if args.wasserstein_subsample <= 0:
            raise ValueError("--wasserstein-subsample must be positive")
        try:
            ram_gb = psutil.virtual_memory().total / 1024**3
            if ram_gb < 8:
                raise SystemExit(
                    f"Insufficient RAM: {ram_gb:.1f} GB detected for the "
                    "bounded Wasserstein computation."
                )

            seeds = sorted(generated_by_seed.keys())
            sets = {}
            for seed in seeds:
                sets[f"generated_s{seed}"] = subsample_rows(
                    generated_by_seed[seed].cpu(),
                    args.wasserstein_subsample,
                    seed + 10_000,
                )
                sets[f"original_s{seed}"] = samples_by_seed[seed].cpu()
                sets[f"original_ref_s{seed}"] = sample_torus_reference(
                    samples_by_seed[seed].shape[0], 30_000 + seed
                )

                n_total = sets[f"generated_s{seed}"].shape[0]
                sets[f"true_s{seed}"] = sample_torus_reference(
                    n_total, 40_000 + seed
                )
                sets[f"true_ref_s{seed}"] = sample_torus_reference(
                    n_total, 50_000 + seed
                )

            results_W, per_seed_results, summary = compute_wasserstein_per_seed(
                sets, seeds
            )

            torch.save(
                {
                    "results_W": results_W,
                    "per_seed_results": per_seed_results,
                    "summary": summary,
                    "seeds": seeds,
                    "metadata": {
                        "wasserstein_subsample": args.wasserstein_subsample,
                        "full_generated_points_per_seed": int(
                            generated_by_seed[seeds[0]].shape[0]
                        ),
                    },
                },
                wass_ckpt_path,
            )
            msg = f"\nSaved Wasserstein results to {wass_ckpt_path}"
            print(msg)
            lines.append(msg)
            add_wasserstein_summary(
                {
                    "results_W": results_W,
                    "per_seed_results": per_seed_results,
                    "summary": summary,
                    "seeds": seeds,
                },
                lines,
            )
        except ImportError:
            msg = "\nSkipping Wasserstein: `pip install POT` to enable."
            print(msg)
            lines.append(msg)
    else:
        msg = (
            f"\nWasserstein checkpoint not found at {wass_ckpt_path}; "
            "skipping local computation. Place wasserstein.pt in this "
            "directory or pass --force-recompute-wasserstein to include the "
            "summary."
        )
        print(msg)
        lines.append(msg)

    output_path.write_text("\n".join(lines) + "\n")
    print(f"Saved outputs to {save_dir}")


@torch.no_grad()
def apply_projection(model, points, applications):
    result = points
    for _ in range(applications):
        result = model(result)
    return result


def metrics_for(manifold, M_delta, samples):
    exact = proj_torus_embedded(M_delta, torus_dim=2, ambient_dim=256)
    return {
        "mie": dist_torus_embedded(
            manifold, torus_dim=2, ambient_dim=256
        ),
        "perr": projection_error(manifold, exact, samples),
        "projection_error": torch.linalg.norm(
            exact - manifold, dim=1
        ).mean().item(),
        "median_displacement": torch.linalg.norm(
            manifold - M_delta, dim=1
        ).median().item(),
        "median_displacement_first4": torch.linalg.norm(
            manifold[:, :4] - M_delta[:, :4], dim=1
        ).median().item(),
    }


def subsample_rows(points, max_points, seed):
    if points.shape[0] <= max_points:
        return points
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    index = torch.randperm(points.shape[0], generator=generator)[:max_points]
    return points[index]


def add_wasserstein_summary(wass_data, lines):
    def emit(text):
        print(text)
        lines.append(text)

    if "per_seed_results" in wass_data:
        seeds = wass_data["seeds"]
        per_seed = wass_data["per_seed_results"]
        summary = wass_data.get("summary", {})

        emit("\nPer-seed Wasserstein distances:")
        for seed in seeds:
            row = per_seed.get(seed, per_seed.get(str(seed)))
            if row is None:
                raise KeyError(f"Missing Wasserstein row for seed {seed}")
            emit(
                f"  Seed {seed}: "
                + "  ".join(
                    f"{name}={float(row[name]):.6f}"
                    for name in WASSERSTEIN_METRICS
                )
            )

        emit("\nWasserstein averages:")
        for name in WASSERSTEIN_METRICS:
            key = f"Avg {name}"
            value = summary.get(key)
            if value is None:
                vals = [
                    float(per_seed.get(seed, per_seed.get(str(seed)))[name])
                    for seed in seeds
                ]
                value = float(np.mean(vals))
            emit(f"  {key} = {float(value):.6f}")
        return

    results_W = wass_data["results_W"]
    seeds = wass_data["seeds"]

    def get_W(a, b):
        return results_W.get((a, b), results_W.get((b, a), float("nan")))

    def avg_pairs(pairs):
        pairs = list(pairs)
        return sum(get_W(*pair) for pair in pairs) / len(pairs)

    num_seeds = len(seeds)

    emit("\nOriginal vs Original:")
    for si, sj in itertools.combinations(seeds, 2):
        emit(f"  W(orig_s{si}, orig_s{sj}) = "
             f"{get_W(f'original_s{si}', f'original_s{sj}'):.6f}")

    emit("\nTrue vs True:")
    for i, j in itertools.combinations(range(1, num_seeds + 1), 2):
        emit(f"  W(true{i}, true{j}) = "
             f"{get_W(f'true{i}', f'true{j}'):.6f}")

    emit("\nGenerated vs True:")
    for seed in seeds:
        for true_idx in range(1, num_seeds + 1):
            emit(f"  W(gen_s{seed}, true{true_idx}) = "
                 f"{get_W(f'generated_s{seed}', f'true{true_idx}'):.6f}")

    emit("\nOriginal vs True:")
    for seed in seeds:
        for true_idx in range(1, num_seeds + 1):
            emit(f"  W(orig_s{seed}, true{true_idx}) = "
                 f"{get_W(f'original_s{seed}', f'true{true_idx}'):.6f}")

    emit("\nGenerated vs Original (same seed):")
    for seed in seeds:
        emit(f"  W(gen_s{seed}, orig_s{seed}) = "
             f"{get_W(f'generated_s{seed}', f'original_s{seed}'):.6f}")

    emit("\n" + "=" * 60)
    emit("Averages:")
    emit(f"  Avg W(orig,orig) = {avg_pairs((f'original_s{si}', f'original_s{sj}') for si, sj in itertools.combinations(seeds, 2)):.6f}")
    emit(f"  Avg W(true,true) = {avg_pairs((f'true{i}', f'true{j}') for i, j in itertools.combinations(range(1, num_seeds + 1), 2)):.6f}")
    emit(f"  Avg W(gen,true)  = {avg_pairs((f'generated_s{seed}', f'true{true_idx}') for seed in seeds for true_idx in range(1, num_seeds + 1)):.6f}")
    emit(f"  Avg W(orig,true) = {avg_pairs((f'original_s{seed}', f'true{true_idx}') for seed in seeds for true_idx in range(1, num_seeds + 1)):.6f}")
    emit(f"  Avg W(gen,orig)  = {avg_pairs((f'generated_s{seed}', f'original_s{seed}') for seed in seeds):.6f}")


def sample_torus_reference(num_points, seed):
    torch.manual_seed(seed)
    return torus_embedded(
        num_points, torus_dim=2, ambient_dim=256,
        density_params=[0.0, 0.5],
    ).cpu()


def compute_wasserstein_per_seed(sets, seeds):
    import ot

    results_W = {}

    def compute_pair(name_a, name_b):
        if (name_a, name_b) in results_W or (name_b, name_a) in results_W:
            return results_W.get(
                (name_a, name_b), results_W.get((name_b, name_a))
            )
        Xi = np.array(sets[name_a])
        Xj = np.array(sets[name_b])
        ai = np.ones(Xi.shape[0]) / Xi.shape[0]
        aj = np.ones(Xj.shape[0]) / Xj.shape[0]
        cost = ot.dist(Xi, Xj, metric="euclidean")
        value = ot.emd2(ai, aj, cost)
        results_W[(name_a, name_b)] = value
        del cost
        return value

    per_seed_results = {}
    for seed in seeds:
        per_seed_results[seed] = {
            "W(orig,orig)": compute_pair(
                f"original_s{seed}", f"original_ref_s{seed}"
            ),
            "W(true,true)": compute_pair(
                f"true_s{seed}", f"true_ref_s{seed}"
            ),
            "W(gen,true)": compute_pair(
                f"generated_s{seed}", f"true_s{seed}"
            ),
            "W(orig,true)": compute_pair(
                f"original_s{seed}", f"true_s{seed}"
            ),
            "W(gen,orig)": compute_pair(
                f"generated_s{seed}", f"original_s{seed}"
            ),
        }

    summary = {
        f"Avg {name}": float(
            np.mean([per_seed_results[seed][name] for seed in seeds])
        )
        for name in WASSERSTEIN_METRICS
    }
    return results_W, per_seed_results, summary


def darken_generated_points(fig, color="#1b6fab", alpha=0.85, size=3):
    ax = fig.axes[0]
    if len(ax.collections) >= 2:
        generated = ax.collections[1]
        generated.set_facecolor(color)
        generated.set_edgecolor(color)
        generated.set_alpha(alpha)
        generated.set_sizes([size])


def save_figures(manifold, M_delta, samples, save_dir, seed):
    fig = plot_3d(
        manifold, samples, view=(30, 120), view_coord=(0, 1, 2), show=False
    )
    darken_generated_points(fig)
    fig.savefig(save_dir / f"seed{seed}_3d_012.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    fig = plot_3d_with_lines(
        manifold, M_delta, samples, 7,
        ambient_dim=256, nparticles=400,
        view=(90, 0), view_coord=(2, 3, 4), show=False,
    )
    fig.savefig(save_dir / f"seed{seed}_3d_lines_234.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    fig = plot_3d(
        manifold, samples, view=(90, 0), view_coord=(2, 3, 4), show=False
    )
    fig.savefig(save_dir / f"seed{seed}_3d_234.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    projected = proj_torus_embedded(
        manifold, torus_dim=2, ambient_dim=256
    )
    alpha_generated = angles_torus_embedded(projected, torus_dim=2)
    alpha_original = angles_torus_embedded(samples, torus_dim=2)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(alpha_generated[:, 0], alpha_generated[:, 1],
               s=1.5, alpha=0.3, label="generated (projected)")
    ax.scatter(alpha_original[:, 0], alpha_original[:, 1],
               s=18, marker="x", c="red", label="original samples")
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, 2 * np.pi)
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / f"seed{seed}_angles.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
