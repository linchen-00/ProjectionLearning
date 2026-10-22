"""Experiment 07: model-capacity sweep on the image-valued circle.

What this experiment does
-------------------------
This script uses the same 8x8 image manifold and projection-training settings
as Experiment 03. It compares four model capacities across five training seeds
for two methods:
1. a projection model with Algorithm 1 and adaptive C2
2. a score-based diffusion model (VE-SDE)

Training, validation, and test geometries are fixed across seeds. For each
method and capacity, it selects the checkpoint with minimum validation MIE.

It saves:
1. per-seed checkpoints and metrics
2. a mean ± standard deviation table of MIE, PErr, and Memo
3. a figure of MIE and Memo versus the number of trainable parameters
Outputs save to ``trained_models/07_capacity_sweep/``

CLI
---
Run with:
``python experiments/07_capacity_sweep.py``

Flags:
``--force-retrain`` to retrain even when checkpoints already exist
``--method projection`` or ``--method vesde`` to run only one method
``--report-only`` to regenerate the table and figure from saved checkpoints
"""

from projlearn.training import train_diffusion, train_projection_algorithm1
from projlearn.sampling import (
    make_sde_fns, tubular_nbh, vector_sampler_with_traj,
)
from projlearn.models import ProjectionNet, ScoreNetVector
from projlearn.manifolds import basis_matrix, circular_movement
from projlearn.evaluation import (
    Pr, Pr_true_circle, memorization_score_integral, projection_error,
)
from torch import nn
import torch
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")


# ── Hyperparameters ──────────────────────────────────────────────────────────
root = Path(__file__).resolve().parent.parent
save_dir = root / "trained_models" / "07_capacity_sweep"
seeds = (2, 3, 4, 5, 6)
n_samples = 30
ambient_dim = 64
delta = 3.0
r_inner = 0.6
num_noisy = 250
n_pretrain = 400
n_projection = 1200
lr_pretrain = 2e-3
lr_projection = 1e-3
metric_every = 200
loss_chunk_size = 256  # Accumulate the full objective before each Adam update.
n_diffusion = 50000
lr_diffusion = 1e-3
diffusion_time_samples = 250
diffusion_metric_every = 1000
sigma = 25.0
num_generated = 5000
diffusion_num_steps = 5000

# Model capacities for the projection and VE-SDE comparisons.
projection_specs = {
    "tiny": {"widths": (64, 64), "residual": False},
    "small": {"widths": (100, 200, 100), "residual": False},
    "medium": {"widths": (256,) * 4, "residual": True},
    "large": {"widths": (512,) * 8, "residual": True},
}
score_specs = {
    "tiny": {"hidden_dims": (64, 128, 64), "embed_dim": 64},
    "small": {"hidden_dims": (128, 256, 128), "embed_dim": 128},
    "medium": {"hidden_dims": (256, 512, 256), "embed_dim": 256},
    "large": {"hidden_dims": (512, 1024, 512), "embed_dim": 256},
}
method_labels = {"projection": "Projection", "vesde": "VE-SDE"}
method_colors = {"projection": "#0072B2", "vesde": "#D55E00"}
method_markers = {"projection": "o", "vesde": "^"}


# ── Experiment ───────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class CapacityProjectionNet(nn.Module):
    def __init__(self, ambient_dim, widths, residual=False):
        super().__init__()
        if residual:
            width = widths[0]
            self.net = nn.Sequential(
                nn.Linear(ambient_dim, width), nn.SiLU(),
                *(ResidualBlock(width) for _ in range(len(widths) // 2)),
                nn.Linear(width, ambient_dim),
            )
        else:
            layers, current = [], ambient_dim
            for width in widths:
                layers.extend([nn.Linear(current, width), nn.SiLU()])
                current = width
            layers.append(nn.Linear(current, ambient_dim))
            self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CircleEvaluation:

    def __init__(self):
        torch.manual_seed(2)
        self.samples = circular_movement(n_samples, ambient_dim).cpu()
        torch.manual_seed(100000)
        anchors = circular_movement(5000, ambient_dim)
        self.validation = tubular_nbh(
            anchors, delta, ambient_dim, 25, 40, device="cpu",
        )
        torch.manual_seed(0)
        anchors = circular_movement(5000, ambient_dim)
        self.test = tubular_nbh(
            anchors, delta, ambient_dim, 125, 40, device="cpu",
        )
        self.basis = basis_matrix([
            lambda x: 0.5,
            lambda x: np.sqrt(3) * x[..., 0] / 2,
            lambda x: np.sqrt(3) * x[..., 1] / 2,
        ], ambient_dim=ambient_dim, device="cpu")
        self.basis_scale = self.basis.norm(dim=1).square()
        self.train_coeff = self.coeff(self.samples)
        theta = torch.atan2(
            self.train_coeff[:, 2], self.train_coeff[:, 1]) + torch.pi
        self.radii = np.linspace(0, 2 / np.sqrt(6), 200)
        self.pr_true = Pr_true_circle(theta, self.radii, radius=1 / np.sqrt(6))

    def coeff(self, x):
        return x.detach().cpu() @ self.basis.T / self.basis_scale

    def metrics(self, generated, inputs=None):
        coeff = self.coeff(generated)[:, 1:]
        radial = coeff / \
            (coeff.norm(dim=1, keepdim=True).clamp_min(1e-12) * np.sqrt(6))
        pr_gen = Pr(self.train_coeff[:, 1:], coeff, self.radii)
        result = {
            "mie": (coeff - radial).norm(dim=1).mean().item(),
            "memo": memorization_score_integral(pr_gen, self.pr_true, self.radii),
        }
        if inputs is not None:
            coeff_in = self.coeff(inputs)[:, 1:]
            exact = coeff_in / (
                coeff_in.norm(dim=1, keepdim=True).clamp_min(
                    1e-12) * np.sqrt(6)
            )
            result["perr"] = projection_error(coeff, exact, self.samples)
        return result


@torch.no_grad()
def generate_samples(model, method, points, seed, device):
    """Apply T twice, or call the library VE-SDE sampler with a fixed RNG seed."""
    model.eval()
    if method == "projection":
        return torch.cat([model(model(part.to(device))).cpu()
                          for part in points.split(4096)])

    # Hold validation/test sampling fixed without changing training randomness.
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps_rng = torch.mps.get_rng_state() if torch.device(device).type == "mps" else None
    try:
        torch.manual_seed(seed)
        marginal_std, diffusion_coeff = make_sde_fns(sigma)
        generated, _ = vector_sampler_with_traj(
            model, ambient_dim, marginal_std, diffusion_coeff,
            batch_size=num_generated, num_steps=diffusion_num_steps, device=device,
            record_steps=2, traj_k=1, ode_last_frac=1.0,
        )
        return generated.cpu()
    finally:
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        if mps_rng is not None:
            torch.mps.set_rng_state(mps_rng)


def run_capacity_model(method, architecture, seed, evaluation, device):
    """Use the library trainers; select a validation checkpoint for this sweep."""
    torch.manual_seed(seed)
    marginal_std, _ = make_sde_fns(sigma)
    if method == "projection":
        spec = projection_specs[architecture]
        widths = spec["widths"]
        if not spec["residual"] and len(widths) == 3 and widths[0] == widths[2]:
            model = ProjectionNet(ambient_dim, hidden=widths[0], mid=widths[1])
        else:
            model = CapacityProjectionNet(ambient_dim, **spec)
    else:
        model = ScoreNetVector(ambient_dim, marginal_std,
                               **score_specs[architecture])
    model = model.to(device)
    trace, best = [], {"mie": float("inf")}
    started = time.time()

    def validate(epoch, current_model, info):
        pretrain = info.get("phase") == "pretrain"
        if pretrain and epoch != n_pretrain:
            return
        update = 0 if pretrain else epoch
        was_training = current_model.training
        generated = generate_samples(
            current_model, method, evaluation.validation, 100002, device,
        )
        metrics = evaluation.metrics(generated)
        trace.append({"update": update, **metrics})
        if metrics["mie"] < best["mie"]:
            state = {key: value.detach().cpu().clone()
                     for key, value in current_model.state_dict().items()}
            best.update(mie=metrics["mie"], update=update, state=state)
        current_model.train(was_training)
        print(
            f"  {method}/{architecture}/seed{seed}: epoch={update} {metrics}", flush=True)

    if method == "projection":
        train_projection_algorithm1(
            model, evaluation.samples, delta,
            n_pretrain=n_pretrain, n_project=n_projection,
            lr_pretrain=lr_pretrain, lr_project=lr_projection,
            num_noisy=num_noisy, r_inner=r_inner,
            loss_pretrain={"C1": 1, "C3": 1},
            loss_project={"C1": 1, "C2": 1, "C3": 1},
            scheduler_pretrain="constant", scheduler_project="cosine",
            loss_chunk_size=loss_chunk_size, device=device,
            epoch_callback=validate, metric_every=metric_every,
        )
    else:
        train_diffusion(
            model, evaluation.samples, marginal_std, n_epochs=n_diffusion,
            M=diffusion_time_samples, lr=lr_diffusion, device=device,
            epoch_callback=validate, metric_every=diffusion_metric_every,
        )
        if not trace or trace[-1]["update"] != n_diffusion:
            validate(n_diffusion, model, {})

    model.load_state_dict(best["state"])
    generated = generate_samples(
        model, method, evaluation.test, 200002, device)
    metrics = evaluation.metrics(
        generated, evaluation.test if method == "projection" else None,
    )
    metrics.update(parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                   selected_update=best["update"], validation_mie=best["mie"],
                   wall_seconds=time.time() - started)
    row = {"method": method, "architecture": architecture, "seed": seed,
           "metrics": metrics, "trajectory": trace}
    return row, best["state"]


def training_config():
    """Reject cached results from a different objective or training budget."""
    names = ("n_samples", "ambient_dim", "delta", "r_inner", "num_noisy", "n_pretrain",
             "n_projection", "lr_pretrain", "lr_projection", "n_diffusion",
             "lr_diffusion", "diffusion_time_samples", "sigma", "num_generated",
             "diffusion_num_steps", "metric_every", "diffusion_metric_every",
             "loss_chunk_size", "projection_specs", "score_specs")
    config = {name: globals()[name] for name in names}
    config["c2"] = "adaptive_squared_jvp_epoch_balance_s0_0.5"
    return json.loads(json.dumps(config))


# ── Output: table and figure ──────────────────────────────────────────────────
def mean_std(values):
    values = np.asarray(values, dtype=float)
    return values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0


def format_metric(values, scientific=False, latex=False):
    mean, std = mean_std(values)
    pm = r"\pm" if latex else "±"
    if scientific and mean != 0:
        exponent = int(np.floor(np.log10(abs(mean))))
        factor = 10.0 ** exponent
        if latex:
            return rf"({mean / factor:.4f} {pm} {std / factor:.4f}) \times 10^{{{exponent}}}"
        return f"({mean / factor:.4f} {pm} {std / factor:.4f}) × 10^{exponent}"
    return f"{mean:.4f} {pm} {std:.4f}"


def write_outputs(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    table_rows, latex_rows, display_rows = [], [], []
    fig, axes = plt.subplots(1, 2, figsize=(7.75, 2.95))
    for method in method_labels:
        means = {"parameters": [], "mie": [], "memo": []}
        for architecture in projection_specs:
            group = [row for row in rows if row["method"] == method
                     and row["architecture"] == architecture]
            if not group:
                continue
            metrics = [row["metrics"] for row in group]
            counts = [m["parameters"] for m in metrics]
            if len(set(counts)) != 1:
                raise ValueError(
                    f"Parameter counts differ for {method}/{architecture}")
            label = [method_labels[method],
                     architecture.title(), f"{counts[0]:,}"]
            plain, latex = [], []
            for key in ("mie", "perr", "memo"):
                if key == "perr" and method == "vesde":
                    plain.append("–")
                    latex.append("--")
                else:
                    values = [m[key] for m in metrics]
                    plain.append(format_metric(
                        values, scientific=key == "perr"))
                    latex.append("$" + format_metric(values,
                                 scientific=key == "perr", latex=True) + "$")
            table_rows.append(label + plain)
            display_rows.append(label + [plain[0], latex[1], plain[2]])
            latex_rows.append(" & ".join(label + latex) + r" \\")
            means["parameters"].append(counts[0])
            for axis, key in zip(axes, ("mie", "memo")):
                values = [m[key] for m in metrics]
                means[key].append(float(np.mean(values)))
                axis.scatter(counts, values, marker=method_markers[method], s=24,
                             facecolor="white", edgecolor=method_colors[method],
                             linewidth=0.8, alpha=0.65, zorder=2)
        if means["parameters"]:
            latex_rows.append(r"\hline")
        for axis, key in zip(axes, ("mie", "memo")):
            axis.plot(means["parameters"], means[key], color=method_colors[method],
                      linewidth=1.25, alpha=0.75, zorder=1)
            axis.scatter(means["parameters"], means[key], marker=method_markers[method],
                         s=46, color=method_colors[method], edgecolor="white",
                         linewidth=0.65, zorder=3)
    for axis, key in zip(axes, ("MIE", "Memo")):
        axis.set_xscale("log")
        axis.set_xlabel("number of trainable parameters")
        axis.set_ylabel(key)
        axis.grid(True, which="major", alpha=0.22, linewidth=0.65)
        axis.grid(True, which="minor", axis="x", alpha=0.10, linewidth=0.45)
    axes[0].set_yscale("log")
    handles = [Line2D([0], [0], color=method_colors[m], marker=method_markers[m],
                      linewidth=1.25, label=method_labels[m]) for m in method_labels]
    seed_count = len({row["seed"] for row in rows})
    mean_label = "Five-seed mean" if seed_count == 5 else f"{seed_count}-seed mean"
    handles.extend([
        Line2D([0], [0], marker="o", linestyle="None", color="0.35",
               markerfacecolor="white", label="Individual seed"),
        Line2D([0], [0], marker="o", linestyle="None",
               color="0.25", label=mean_label),
    ])
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=4, fontsize=7.5, frameon=True)
    fig.subplots_adjust(left=0.08, right=0.98,
                        bottom=0.18, top=0.80, wspace=0.25)
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"capacity_comparison.{extension}", dpi=320, bbox_inches="tight")
    plt.close(fig)

    headers = ["Method", "Capacity", "Parameters", "MIE", "PErr", "Memo"]
    title = (f"{seed_count}-seed image-circle capacity sweep. Mean ± sample standard "
             "deviation at minimum-validation-MIE checkpoints; PErr only for projection.")
    lines = [title, "", "| " + " | ".join(headers) + " |",
             "|---|---|---:|---:|---:|---:|"]
    lines.extend("| " + " | ".join(row) + " |" for row in table_rows)
    (output_dir / "metrics_table.md").write_text("\n".join(lines) + "\n")
    latex = [r"\begin{tabular}{llrrrr}", r"\hline", " & ".join(headers) + r" \\", r"\hline",
             *latex_rows, r"\end{tabular}"]
    (output_dir / "metrics_table.tex").write_text("\n".join(latex) + "\n")
    table_fig, axis = plt.subplots(
        figsize=(11.5, max(1.8, 0.32 * (len(table_rows) + 2))))
    axis.axis("off")
    table = axis.table(cellText=display_rows, colLabels=headers, loc="center",
                       cellLoc="center", colWidths=[0.11, 0.09, 0.12, 0.19, 0.30, 0.19])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.45)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.visible_edges = "TB"
        else:
            group_end = r == len(
                table_rows) or table_rows[r - 1][0] != table_rows[r][0]
            cell.visible_edges = "B" if group_end else ""
    table_fig.savefig(output_dir / "metrics_table.png",
                      dpi=250, bbox_inches="tight")
    plt.close(table_fig)
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--method", choices=("both",
                        "projection", "vesde"), default="both")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(seeds))
    parser.add_argument("--capacities", nargs="+", choices=tuple(projection_specs),
                        default=list(projection_specs))
    parser.add_argument("--output-dir", type=Path, default=save_dir)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 9})
    device = ("cuda" if torch.cuda.is_available() else
              "mps" if torch.backends.mps.is_available() else "cpu")
    output_dir = args.output_dir
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = training_config()
    methods = tuple(method_labels) if args.method == "both" else (args.method,)
    evaluation = CircleEvaluation()
    rows = []
    for method in methods:
        for architecture in args.capacities:
            for seed in args.seeds:
                checkpoint_path = checkpoint_dir / \
                    f"{method}_{architecture}_seed{seed}.pt"
                if checkpoint_path.exists() and (args.report_only or not args.force_retrain):
                    saved = torch.load(
                        checkpoint_path, map_location="cpu", weights_only=False)
                    # Accept the previous uppercase checkpoint metadata as well.
                    saved_config = {key.lower(): value
                                    for key, value in saved["training_config"].items()}
                    aliases = {"n": "n_samples", "eval_count": "num_generated",
                               "sampling_steps": "diffusion_num_steps",
                               "projection_validate_every": "metric_every",
                               "diffusion_validate_every": "diffusion_metric_every",
                               "objective_chunk": "loss_chunk_size"}
                    saved_config = {aliases.get(key, key): value
                                    for key, value in saved_config.items()}
                    if saved_config != config:
                        raise ValueError(
                            f"Configuration changed: {checkpoint_path}; use --force-retrain")
                    row = saved["row"]
                else:
                    if args.report_only:
                        raise FileNotFoundError(
                            f"Missing requested run: {checkpoint_path}")
                    row, state = run_capacity_model(
                        method, architecture, seed, evaluation, device,
                    )
                    torch.save({"row": row, "model_state_dict": state,
                                "selection": "minimum_validation_mie", "training_config": config},
                               checkpoint_path)
                rows.append(row)
    write_outputs(rows, output_dir)
    print(f"Saved capacity results to {output_dir}")


if __name__ == "__main__":
    main()
