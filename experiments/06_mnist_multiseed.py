"""Experiment 06: multi-seed MNIST evaluation for projection and diffusion.

What this experiment does
-------------------------
This script runs a shared multi-seed evaluation on MNIST for two generative
approaches:
1. the projection cascade (IPA)
2. the diffusion model (VE-SDE / ScoreNet)

For each seed, it resolves or regenerates a deterministic 10,000-image MNIST
training split, makes sure both methods use the same split, then trains or
loads the required model artifacts and generated samples.

It evaluates each method with memorization and distribution metrics, including
pixel-space nearest-neighbour ratios, CapsNet-based FID and KID, and P_r
curves against both:
1. that seed's held-out MNIST training complement
2. the shared MNIST test set

Outputs save to ``trained_models/06_mnist/``

CLI
---
Run both methods:
``python experiments/06_mnist_multiseed.py``

Run only one method:
``python experiments/06_mnist_multiseed.py --only projection``
``python experiments/06_mnist_multiseed.py --only diffusion``

Flags:
``--force-eval`` to recompute metrics and plots from saved artifacts
``--force-retrain`` to retrain both projection and diffusion
``--force-retrain-projection`` to retrain only the projection pipeline
``--force-retrain-diffusion`` to retrain only the diffusion pipeline
``--no-show`` to save figures without opening plot windows
"""

# pylint: disable=too-many-locals,too-many-statements
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats as _scipy_stats
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST

from projlearn.evaluation import memorization_score_ratio  # noqa: E402
from projlearn.mnist_eval import (  # noqa: E402
    capsnet_embed_160,
    collect_embeddings_from_tensor,
    compose,
    compute_pr_curves,
    fid_from_stats,
    kid_from_embeddings,
    load_digit_models,
    stats_from_embeddings,
    try_setup_capsnet as _try_setup_capsnet,
)
from projlearn.models import ProjectionUNet, ScoreNet  # noqa: E402
from projlearn.sampling import (  # noqa: E402
    make_sde_fns, pc_sampler, tubular_nbh,
)
from projlearn.training import (  # noqa: E402
    train_diffusion, train_projection_algorithm2,
)

here = Path(__file__).resolve().parent
root = here.parent

# ── Paths / config ────────────────────────────────────────────────────────────
projection_dir = root / "trained_models" / "06_mnist" / "all_digits_multiseed"
diffusion_dir = root / "trained_models" / "06_mnist" / "diffusion_multiseed"
figures_dir = root / "trained_models" / "06_mnist" / "figures"
mnist_data_dir = str(root)

seeds = [0, 2, 3, 4, 6]
n_samples = 10_000
n_ref_heldout = 10_000   # how many of remain_samples to use as held-out ref
deltas = [20.0, 8.0, 4.0]
r_inners = [8.0, 4.0, 2.0]
delta_generation = 20.0
num_noisy_generation = 1            # noisy points per training sample at generation
channels = (16, 32, 64, 128)
n_pretrain = [100, 100, 100]
n_projection = [50, 50, 50]
# The checkpoint loader uses these historical phase-key names.
stage_specs = [{"iter1": pre, "iter2": project}
               for pre, project in zip(n_pretrain, n_projection)]

# Diffusion (VE-SDE) hyper-parameters — match the notebook
sigma = 25.0
n_diffusion = 300
diffusion_time_samples = 1
diffusion_batch_size = 64
lr_diffusion = 1e-4
diffusion_num_steps = 500
diffusion_snr = 0.16

# Projection-cascade training hyper-parameters
batch_size = 64
lr_pretrain = 2e-3
lr_projection = 5e-4
num_noisy = 50
loss_pretrain = {"C1": 1, "C3": 1}
loss_projection = {"C1": 1, "C2": 1, "C3": 1}

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# Per-seed helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get_mnist_subset(n_samples: int, seed: int):
    """Replicates notebook ``get_mnist_subset(digit=None, N, seed)``."""
    ds = MNIST(mnist_data_dir, train=True, download=True,
               transform=transforms.ToTensor())
    imgs = torch.stack([img.view(-1) for img, _ in ds], dim=0).float()
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(imgs), generator=g)
    n_eff = min(n_samples, len(imgs))
    return imgs[perm[:n_eff]].contiguous(), imgs[perm[n_eff:]].contiguous()


def _validate_split(split_file: Path) -> tuple[bool, str]:
    """Validate a split file; return (is_valid, message)."""
    try:
        ck = torch.load(split_file, map_location="cpu", weights_only=False)
    except Exception as e:
        return False, f"Cannot load: {e}"
    tr = ck.get("train_samples")
    rm = ck.get("remain_samples")
    if tr is None or rm is None:
        return False, "Missing keys 'train_samples' or 'remain_samples'"
    if tr.shape != (n_samples, 784):
        return False, f"train_samples shape {tuple(tr.shape)} != ({n_samples}, 784)"
    if rm.shape[1] != 784:
        return False, f"remain_samples has wrong feature dim {rm.shape[1]}"
    expected_remain = 60_000 - n_samples
    if rm.shape[0] != expected_remain:
        return (False,
                f"remain_samples size {rm.shape[0]} != expected {expected_remain}")
    return True, f"OK  train={tuple(tr.shape)}  remain={tuple(rm.shape)}"


def _load_split(seed_dir: Path, seed: int, *,
                force_regen: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Load (or create if missing) the per-seed (train, held-out) split.

    ``seed_dir`` is the directory where the split should live.  For diffusion
    seeds, callers pass the corresponding projection seed directory
    (``projection_dir / f"seed{s}"``), because projection and diffusion use the same
    10 k training samples for every seed.

    Priority once ``seed_dir`` is resolved:
      1. ``seed_dir / mnist_samples_split.pth``  — primary file
      2. Safety-net fallback: ``projection_dir / seed_dir.name / mnist_samples_split.pth``
         (only reached if the caller passes a non-projection directory)
      3. Deterministically regenerate from the MNIST training set and save the
         split in ``seed_dir`` (only when both 1+2 are absent, or when forced)

    Existing files are **never** overwritten unless ``force_regen=True``.
    """
    own_file = seed_dir / "mnist_samples_split.pth"
    proj_file = projection_dir / seed_dir.name / "mnist_samples_split.pth"

    candidate = None
    source_label = ""

    if not force_regen:
        if own_file.exists():
            ok, msg = _validate_split(own_file)
            if ok:
                candidate, source_label = own_file, "own"
            else:
                print(f"  [SPLIT WARN] {own_file}: {msg}")
        if candidate is None and proj_file.exists() and proj_file != own_file:
            ok, msg = _validate_split(proj_file)
            if ok:
                candidate, source_label = proj_file, "proj-fallback"
                print(f"  [SPLIT] seed{seed}: using projection split as fallback "
                      f"({proj_file})")
            else:
                print(f"  [SPLIT WARN] {proj_file}: {msg}")

    if candidate is None:
        print(f"  [SPLIT] seed{seed}: generating fresh split (N={n_samples})")
        train_s, remain_s = _get_mnist_subset(n_samples=n_samples, seed=seed)
        seed_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"train_samples": train_s.cpu(),
                    "remain_samples": remain_s.cpu()}, own_file)
        print(f"  [SPLIT] saved → {own_file}")
        return train_s.float(), remain_s.float()

    ck = torch.load(candidate, map_location="cpu", weights_only=False)
    train_s = ck["train_samples"].float()
    remain_s = ck["remain_samples"].float()
    print(f"  [SPLIT] seed{seed}: loaded ({source_label})  "
          f"train={tuple(train_s.shape)}  remain={tuple(remain_s.shape)}")
    return train_s, remain_s


def _projection_ckpts_complete(seed_dir: Path) -> bool:
    """All 2*K cascade snapshots present in seed_dir (either layout)?"""
    for i, spec in enumerate(stage_specs):
        for phase, ep in (("a", spec["iter1"]), ("b", spec["iter2"])):
            phase_name = "pretrain" if phase == "a" else "project"
            legacy = seed_dir / f"model{i}{phase}_epoch{ep}.pth"
            new = seed_dir / f"alg2_stage{i}_{phase_name}_epoch{ep}.pth"
            if not (legacy.exists() or new.exists()):
                return False
    return True


def _train_projection_cascade(seed: int, train_flat: torch.Tensor,
                              seed_dir: Path):
    seed_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [TRAIN proj cascade]  seed={seed}  → {seed_dir}")
    torch.manual_seed(seed)
    model = ProjectionUNet(image_size=28, channels=channels,
                           in_channels=1).to(device)
    samples_dev = train_flat.to(device)
    save_every = min(min(n_pretrain), min(n_projection))
    out = train_projection_algorithm2(
        model, samples_dev,
        deltas=list(deltas), r_inners=list(r_inners),
        n_pretrain=n_pretrain, n_project=n_projection,
        lr_pretrain=lr_pretrain, lr_project=lr_projection,
        num_points_per_samples=num_noisy,
        loss_pretrain=loss_pretrain, loss_project=loss_projection,
        scheduler_pretrain="constant", scheduler_project="constant",
        batch_size=batch_size, use_dataloader=True,
        print_every=20, show_progress=True, device=device,
        save_dir=str(seed_dir), save_every_epochs=save_every,
        save_prefix="alg2",
        c2_balance=1.0,  # Paper protocol: original C2 at every cascade stage.
    )
    print(f"  [DONE proj cascade]   seed={seed}")
    interleaved = []
    for pre, full in zip(out["pretrain_snapshots"], out["stage_snapshots"]):
        interleaved.append(pre)
        interleaved.append(full)
    return interleaved


def _train_diffusion_score_and_sample(seed: int, train_flat: torch.Tensor,
                                      seed_dir: Path) -> torch.Tensor:
    seed_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [TRAIN diffusion]     seed={seed}  → {seed_dir}")
    marginal_prob_std_fn, diffusion_coeff_fn = make_sde_fns(sigma=sigma)
    torch.manual_seed(seed)
    model = ScoreNet(marginal_prob_std=marginal_prob_std_fn).to(device)
    images = train_flat.view(-1, 1, 28, 28).to(device)
    train_diffusion(model, images, marginal_prob_std_fn,
                    n_epochs=n_diffusion, M=diffusion_time_samples, lr=lr_diffusion,
                    use_dataloader=True, batch_size=diffusion_batch_size, device=device)
    torch.save(model.state_dict(), seed_dir / "score_model.pth")
    print(f"  [DONE diffusion train] seed={seed}; "
          f"sampling {train_flat.shape[0]} images …")
    model.eval()
    n_total = train_flat.shape[0]
    bs = 100
    chunks = []
    torch.manual_seed(seed)
    for b in range((n_total + bs - 1) // bs):
        x = pc_sampler(model, marginal_prob_std_fn, diffusion_coeff_fn,
                       batch_size=bs, num_steps=diffusion_num_steps, snr=diffusion_snr,
                       device=device, eps=1e-3, image_size=28, channels=1,
                       show_progress=(b == 0))
        chunks.append(x.view(bs, -1).clamp(0.0, 1.0).cpu())
    gen = torch.cat(chunks, dim=0)[:n_total]
    torch.save(gen, seed_dir / "generated_samples.pt")
    return gen


def _sample_diffusion_from_checkpoint(seed: int, n_total: int,
                                      seed_dir: Path) -> torch.Tensor | None:
    """Load historical or newly trained diffusion weights and run the PC sampler."""
    ckpt_path = next((seed_dir / name for name in
                      ("score_model.pth", "diff_model.pt")
                      if (seed_dir / name).exists()), None)
    if ckpt_path is None:
        return None

    print(f"  [DIFFUSION] seed{seed}: sampling from {ckpt_path.name}")
    marginal_prob_std_fn, diffusion_coeff_fn = make_sde_fns(sigma=sigma)
    model = ScoreNet(marginal_prob_std=marginal_prob_std_fn).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"{ckpt_path} did not contain a state dict")
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    bs = 100
    chunks = []
    torch.manual_seed(seed)
    for b in range((n_total + bs - 1) // bs):
        batch_size = min(bs, n_total - b * bs)
        x = pc_sampler(model, marginal_prob_std_fn, diffusion_coeff_fn,
                       batch_size=batch_size, num_steps=diffusion_num_steps,
                       snr=diffusion_snr, device=device, eps=1e-3, image_size=28,
                       channels=1, show_progress=(b == 0))
        chunks.append(x.view(batch_size, -1).clamp(0.0, 1.0).cpu())
    del model
    gc.collect()
    return torch.cat(chunks, dim=0)


def _generate_projection(seed: int, train_flat: torch.Tensor,
                         seed_dir: Path, *,
                         force_retrain: bool = False) -> torch.Tensor:
    if force_retrain or not _projection_ckpts_complete(seed_dir):
        _train_projection_cascade(seed, train_flat, seed_dir)
    models = load_digit_models(seed_dir, stage_specs, channels, device)
    torch.manual_seed(seed)
    M_delta = tubular_nbh(train_flat, delta=delta_generation, ambient_dim=784,
                          num_chosen_samples=None, num_points_per_sample=num_noisy_generation)
    with torch.no_grad():
        gen = compose(models, M_delta.to(device)).cpu()
    del models
    gc.collect()
    return gen


def _load_diffusion_samples(seed_dir: Path, *, seed: int | None = None,
                            train_flat: torch.Tensor | None = None,
                            force_retrain: bool = False) -> torch.Tensor:
    """Load cached diffusion samples, sample ``diff_model.pt``, or retrain.

    Cache priority:
      1. ``generated_samples.pt`` when present and retraining is not forced.
      2. ``diff_model.pt`` when generated samples are missing and retraining is
         not forced.
      3. Train a fresh ScoreNet and save ``generated_samples.pt``.
    """
    cache = seed_dir / "generated_samples.pt"
    if not force_retrain and cache.exists():
        gen = torch.load(cache, map_location="cpu", weights_only=False).float()
        if gen.dim() == 4:
            gen = gen.view(gen.shape[0], -1)
        return gen.clamp(0., 1.)

    if seed is None or train_flat is None:
        raise ValueError(
            "seed and train_flat are required for per-seed diffusion")

    if not force_retrain:
        gen = _sample_diffusion_from_checkpoint(
            seed, train_flat.shape[0], seed_dir)
        if gen is not None:
            return gen.clamp(0., 1.)

    gen = _train_diffusion_score_and_sample(seed, train_flat, seed_dir)
    if gen.dim() == 4:
        gen = gen.view(gen.shape[0], -1)
    return gen.clamp(0., 1.)


# ─────────────────────────────────────────────────────────────────────────────
# Statistical reporting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _summary(values: np.ndarray):
    n = values.size
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values, ddof=1)) if n > 1 else 0.0
    ci = float(_scipy_stats.t.ppf(0.975, df=n - 1) *
               std / np.sqrt(n)) if n > 1 else 0.0
    return mean, std, ci, float(np.nanmin(values)), float(np.nanmax(values))


def _ttest_one_sample(values: np.ndarray):
    if np.allclose(values, values[0]):
        return float("nan"), float("nan")
    res = _scipy_stats.ttest_1samp(values, popmean=0.0, nan_policy="omit")
    return float(res.statistic), float(res.pvalue)


def _format_report(label: str, seeds: list[int], n_train: int,
                   per_seed: dict, hyper_line: str) -> str:
    metrics = ["fid_heldout", "kid_heldout", "fid_test", "kid_test",
               "memo_heldout", "memo_test", "memo_pixel"]
    pretty = {
        "fid_heldout":  "FID  (CapsNet, per-seed held-out)",
        "kid_heldout":  "KID  (CapsNet, per-seed held-out)",
        "fid_test":     "FID  (CapsNet, MNIST test)",
        "kid_test":     "KID  (CapsNet, MNIST test)",
        "memo_heldout": "Memo (CapsNet, per-seed held-out)",
        "memo_test":    "Memo (CapsNet, MNIST test)",
        "memo_pixel":   "Memo (pixel L2, k=1/3)",
    }
    n = len(seeds)
    t_crit = _scipy_stats.t.ppf(0.975, df=n - 1)
    L: list[str] = []
    bar = "=" * 92
    L.append(bar)
    L.append(f"Multi-seed Evaluation Report — {label}")
    L.append(f"Seeds       : {seeds}")
    L.append(f"N per seed  : {n_train}  |  {hyper_line}")
    L.append(
        f"n (# seeds) : {n}  |  t_{{0.025,{n-1}}} = {t_crit:.4f}  (95 % CI)")
    L.append(f"Held-out ref: first {n_ref_heldout} rows of per-seed remain_samples "
             f"(zero cross-seed contamination)")
    L.append(bar)
    L.append(f"  {'Metric':<36} {'Mean':>10} {'Std':>10} {'95 % CI ±':>10}"
             f" {'Min':>10} {'Max':>10}")
    L.append("  " + "-" * 90)
    for m in metrics:
        vals = np.asarray([per_seed[s][m] for s in seeds], dtype=np.float64)
        mu, sd, ci, lo, hi = _summary(vals)
        L.append(f"  {pretty[m]:<36} {mu:10.5f} {sd:10.5f} {ci:10.5f}"
                 f" {lo:10.5f} {hi:10.5f}")
    L.append("")
    L.append("One-sample t-test vs. 0  (H0: metric = 0):")
    L.append(f"  {'Metric':<36} {'t-stat':>10} {'p-value':>10}  sig.")
    L.append("  " + "-" * 66)
    for m in ("kid_heldout", "kid_test", "memo_heldout",
              "memo_test", "memo_pixel"):
        vals = np.asarray([per_seed[s][m] for s in seeds], dtype=np.float64)
        t, p = _ttest_one_sample(vals)
        sig = "n.s." if not np.isfinite(p) else (
            "***" if p < 0.001 else "**" if p < 0.01
            else "*" if p < 0.05 else "n.s.")
        L.append(f"  {pretty[m]:<36} {t:+10.3f} {p:10.4f}    {sig}")
    L.append("")
    L.append("Per-seed breakdown:")
    head = "  " + " ".join(f"{c:>13s}" for c in ("Seed", *metrics))
    L.append(head)
    L.append("  " + "-" * (len(head) - 2))
    for s in seeds:
        row = [f"{s:>13d}"] + [f"{per_seed[s][m]:>13.5f}" for m in metrics]
        L.append("  " + " ".join(row))
    L.append(bar)
    return "\n".join(L)


# ─────────────────────────────────────────────────────────────────────────────
# Figure helpers
# ─────────────────────────────────────────────────────────────────────────────

pr_rcparams = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
}


def _save_aggregate_pr_test_both(pr_proj: dict, pr_diff: dict,
                                 save_path: Path):
    """Single figure: aggregate test-reference P_r for BOTH models.

    pr_proj / pr_diff: seed -> (radii, pr_gen, pr_ref_test)
    figsize=(8,6).  No title.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update(pr_rcparams)

    fig, ax = plt.subplots(figsize=(8, 6))
    r_mean_ref = None
    radii_last = None

    for pr_data, color, gen_label in (
            (pr_proj, "steelblue",   r"$P_r^{\mathrm{proj}}$"),
            (pr_diff, "forestgreen", r"$P_r^{\mathrm{diff}}$")):
        if not pr_data:
            continue
        radii = next(iter(pr_data.values()))[0]
        radii_last = radii
        G = np.stack([v[1] for v in pr_data.values()], axis=0)
        R = np.stack([v[2] for v in pr_data.values()], axis=0)
        g_mean, g_std = G.mean(0), G.std(0, ddof=1)
        r_mean = R.mean(0)
        r_std = R.std(0, ddof=1)
        ax.plot(radii, g_mean, color=color, lw=1.8, label=gen_label)
        ax.fill_between(radii, g_mean - g_std, g_mean + g_std,
                        color=color, alpha=0.18)
        if r_mean_ref is None:
            # Plot reference once (same for both models by construction)
            r_mean_ref = r_mean
            ax.plot(radii, r_mean, color="#c0392b", lw=1.8,
                    label=r"$P_r^{\mathrm{true}}$ (MNIST test)")
            ax.fill_between(radii, r_mean - r_std, r_mean + r_std,
                            color="darkorange", alpha=0.18)

    # Memorization gap shading for each model
    if r_mean_ref is not None:
        for pr_data, _, gap_label in (
                (pr_proj, "steelblue",   "memo gap (proj)"),
                (pr_diff, "forestgreen", "memo gap (diff)")):
            if not pr_data:
                continue
            radii = next(iter(pr_data.values()))[0]
            G = np.stack([v[1] for v in pr_data.values()], axis=0)
            g_mean = G.mean(0)
            gap = np.maximum(g_mean - r_mean_ref, 0.0)
            if gap.max() > 0:
                ax.fill_between(radii, r_mean_ref, r_mean_ref + gap,
                                where=(gap > 0), color="darkorange", alpha=0.30,
                                label=f"Memorization area")

    ax.set_xlabel(r"radius $r$ (CapsNet 160-d)", fontsize=14)
    ax.set_ylabel(r"$P_r(r)$", fontsize=14)
    if radii_last is not None:
        ax.set_xlim(0.0, float(radii_last[-1]))
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=14, loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved combined test-reference aggregate P_r → {save_path}")
    return fig


def _save_per_seed_pr_2panel(seed: int,
                             pr_proj_h, pr_diff_h,
                             pr_proj_t, pr_diff_t,
                             save_path: Path):
    """2-panel P_r figure for one seed.

    Left panel : per-seed held-out reference.
    Right panel: MNIST test reference.
    Each panel shows P_r^{proj}, P_r^{diff}, P_r^{true} + memo-gap shading.
    figsize=(9.6, 3.9).  No title.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update(pr_rcparams)

    panel_data = [
        (pr_proj_h, pr_diff_h),
        (pr_proj_t, pr_diff_t),
    ]
    if all(p is None and d is None for p, d in panel_data):
        return None

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)

    for ax, (pr_p, pr_d) in zip(axes, panel_data):
        # Collect reference (same for both models by construction)
        radii = pr_ref = None
        if pr_p is not None:
            radii, pr_gen_p, pr_ref = pr_p
        if pr_d is not None:
            radii_d, pr_gen_d, pr_ref_d = pr_d
            if radii is None:
                radii = radii_d
                pr_ref = pr_ref_d

        if radii is None:
            ax.set_visible(False)
            continue

        # Reference curve
        ax.plot(radii, pr_ref, color="darkorange", lw=1.6,
                label=r"$P_r^{\mathrm{true}}$")

        # Projection
        if pr_p is not None:
            _, pr_gen_p, _ = pr_p
            gap_p = np.maximum(pr_gen_p - pr_ref, 0.0)
            ax.plot(radii, pr_gen_p, color="steelblue", lw=1.5,
                    label=r"$P_r^{\mathrm{proj}}$")
            if gap_p.max() > 0:
                ax.fill_between(radii, pr_ref, pr_ref + gap_p,
                                where=(gap_p > 0),
                                color="red", alpha=0.25,
                                label=f"Memorization area")

        # Diffusion
        if pr_d is not None:
            _, pr_gen_d, _ = pr_d
            gap_d = np.maximum(pr_gen_d - pr_ref, 0.0)
            ax.plot(radii, pr_gen_d, color="forestgreen", lw=1.5,
                    label=r"$P_r^{\mathrm{diff}}$")
            if gap_d.max() > 0:
                ax.fill_between(radii, pr_ref, pr_ref + gap_d,
                                where=(gap_d > 0),
                                color="red", alpha=0.22,
                                label=f"Memorization area")

        ax.set_xlabel(r"radius $r$", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel(r"$P_r(r)$", fontsize=9)
        ax.set_xlim(0.0, float(radii[-1]))
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved per-seed 2-panel P_r → {save_path}")
    return fig


def _save_nn_figure(gen: torch.Tensor, train_s: torch.Tensor,
                    save_path: Path,
                    n_rows: int = 8, n_cols_left: int = 5,
                    n_nn: int = 2):
    """Nearest-neighbour visualisation figure for projection models.

    Three panels, left to right:
      1. Training panel : n_rows × n_cols_left grid of training images.
                          Footer label "Training samples".
      2. Generated panel: n_rows × n_cols_left grid of generated images.
                          Footer label "Generated samples".
      3. Comparison     : n_rows × 3 grid (GEN | NN1 | NN2) with column
                          headers GEN / NN1 / NN2 at the bottom.

    All images touch with no visible separators inside each panel; the
    comparison panel keeps a thin vertical gap between its three columns.
    NN computed in pixel L2 space.  figsize=(10.0, 6.0).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(pr_rcparams)

    n_left = n_rows * n_cols_left

    # Generated images for panel 2 (and column 0 of panel 3)
    gen_np = gen[:n_left].cpu().float().numpy().reshape(-1, 28, 28)
    train_np = train_s.cpu().float().numpy().reshape(-1, 28, 28)

    # Training images for panel 1: deterministic first n_left training samples
    train_grid_np = train_np[:n_left]

    # NN in pixel space for the first n_rows generated samples
    sel = gen[:n_rows].float()
    nn_idx = torch.cdist(
        sel, train_s.float()
    ).topk(n_nn, dim=1, largest=False).indices  # (n_rows, n_nn)

    # Figure: roughly 1.5x wider than the previous 2-panel layout so that the
    # two stitched grids on the left preserve their original visual size and
    # the comparison panel keeps its original visual size.
    fig = plt.figure(figsize=(10.0, 6.0))

    # Reserve bottom strip for footer labels (mirrors original header_h).
    header_h = 0.07
    img_top = 1.0 - header_h

    H, W = 28, 28

    # Helper: stitch an n_rows × n_cols grid of (28, 28) images into a single
    # canvas; returns (canvas_h, canvas_w) for header positioning.
    def _stitch(images_np, n_rows_, n_cols_):
        canvas = np.zeros((n_rows_ * H, n_cols_ * W), dtype=np.float32)
        for r in range(n_rows_):
            for c in range(n_cols_):
                k = r * n_cols_ + c
                if k < len(images_np):
                    canvas[r * H:(r + 1) * H,
                           c * W:(c + 1) * W] = images_np[k]
        return canvas

    # Common axes-transform footer style (same as previous header style)
    footer_kwargs = dict(
        ha="center", va="top",
        fontsize=14, fontweight="bold",
    )
    footer_y = -0.03  # axes-fraction y, just below image area

    # ----------------------------
    # PANEL 1: training images (stitched)
    # ----------------------------
    p1_l, p1_r = 0.010, 0.360
    ax_train = fig.add_axes([p1_l, 0.01, p1_r - p1_l, img_top - 0.01])
    train_canvas = _stitch(train_grid_np, n_rows, n_cols_left)
    ax_train.imshow(
        train_canvas, cmap="gray", vmin=0, vmax=1,
        interpolation="nearest", aspect="auto",
    )
    ax_train.set_axis_off()
    ax_train.text(0.5, footer_y, "Training samples",
                  transform=ax_train.transAxes, **footer_kwargs)

    # ----------------------------
    # PANEL 2: generated images (stitched)
    # ----------------------------
    p2_l, p2_r = 0.385, 0.735
    ax_gen = fig.add_axes([p2_l, 0.01, p2_r - p2_l, img_top - 0.01])
    gen_canvas = _stitch(gen_np, n_rows, n_cols_left)
    ax_gen.imshow(
        gen_canvas, cmap="gray", vmin=0, vmax=1,
        interpolation="nearest", aspect="auto",
    )
    ax_gen.set_axis_off()
    ax_gen.text(0.5, footer_y, "Generated samples",
                transform=ax_gen.transAxes, **footer_kwargs)

    # -----------------------------
    # PANEL 3: GEN | NN1 | NN2 comparison
    # -----------------------------
    p3_l, p3_r = 0.760, 0.998
    ax_right = fig.add_axes([p3_l, 0.01, p3_r - p3_l, img_top - 0.01])

    n_right_cols = 1 + n_nn   # 3 columns: GEN, NN1, NN2

    gap_x = 2   # thin separator between columns; set to 1 if you want thinner
    gap_y = 0   # no row gaps

    right_canvas_h = n_rows * H + (n_rows - 1) * gap_y
    right_canvas_w = n_right_cols * W + (n_right_cols - 1) * gap_x

    # use black background; only explicit gaps appear
    right_canvas = np.zeros((right_canvas_h, right_canvas_w), dtype=np.float32)

    for r in range(n_rows):
        y0 = r * (H + gap_y)

        # column 0 = generated
        x0 = 0
        right_canvas[y0:y0 + H, x0:x0 + W] = gen_np[r]

        # columns 1..n_nn = nearest neighbours
        for j in range(n_nn):
            x0 = (j + 1) * (W + gap_x)
            right_canvas[y0:y0 + H, x0:x0 + W] = train_np[nn_idx[r, j].item()]

    ax_right.imshow(
        right_canvas,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="auto",
    )
    ax_right.set_axis_off()

    # Column headers for comparison panel (bottom)
    headers = ["GEN", "NN1", "NN2"]
    for c, hdr in enumerate(headers):
        x_center = (c * (W + gap_x) + W / 2) / right_canvas_w
        ax_right.text(
            x_center, footer_y, hdr,
            transform=ax_right.transAxes,
            **footer_kwargs,
        )

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved NN figure → {save_path}")
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Per-seed evaluation
# ─────────────────────────────────────────────────────────────────────────────


def _evaluate_seed(seed: int, gen: torch.Tensor, train_s: torch.Tensor,
                   remain_s: torch.Tensor,
                   capsnet, emb_test_shared):
    """Compute (metrics_dict, pr_heldout, pr_test) for one seed.

    Held-out reference: first N_REF_HELDOUT rows of remain_s (deterministic,
    per-seed, no cross-seed contamination).

    Returns
    -------
    metrics  : dict
    pr_heldout : (radii, pr_gen, pr_ref_heldout)  or  (None, None, None)
    pr_test    : (radii, pr_gen, pr_ref_test)       or  (None, None, None)
    """
    memo_pix, _, _, _ = memorization_score_ratio(
        gen, train_s, k=1 / 3, batch_size=128)

    out = {"memo_pixel": float(memo_pix)}
    null_panel = (None, None, None)

    if capsnet is None:
        for k in ("fid_heldout", "kid_heldout", "fid_test", "kid_test",
                  "memo_heldout", "memo_test"):
            out[k] = float("nan")
        return out, null_panel, null_panel

    emb_gen = collect_embeddings_from_tensor(capsnet, gen)
    emb_train = collect_embeddings_from_tensor(capsnet, train_s)

    # Per-seed held-out reference: first N_REF_HELDOUT of this seed's remain
    ref_heldout = remain_s[:n_ref_heldout]
    emb_held = collect_embeddings_from_tensor(capsnet, ref_heldout)
    emb_test = emb_test_shared

    mu_g, Sig_g = stats_from_embeddings(emb_gen)
    for ref_name, emb_r in (("heldout", emb_held), ("test", emb_test)):
        mu_r, Sig_r = stats_from_embeddings(emb_r)
        out[f"fid_{ref_name}"] = fid_from_stats(mu_r, Sig_r, mu_g, Sig_g)
        out[f"kid_{ref_name}"] = kid_from_embeddings(emb_r, emb_gen)

    radii_h, prg_h, prr_h, memo_h = compute_pr_curves(
        emb_train.numpy(), emb_gen.numpy(), emb_held.numpy())
    radii_t, prg_t, prr_t, memo_t = compute_pr_curves(
        emb_train.numpy(), emb_gen.numpy(), emb_test.numpy())

    out["memo_heldout"] = float(memo_h)
    out["memo_test"] = float(memo_t)

    return out, (radii_h, prg_h, prr_h), (radii_t, prg_t, prr_t)


def _shared_test_embedding(capsnet) -> torch.Tensor:
    """Embed 10 000 MNIST test images once (seed=999)."""
    ds = MNIST(mnist_data_dir, train=False, download=True,
               transform=transforms.ToTensor())
    g = torch.Generator().manual_seed(999)
    idx = torch.randperm(len(ds), generator=g)[:10_000].tolist()
    dl = DataLoader(Subset(ds, idx), batch_size=256, shuffle=False)
    chunks = []
    for xb, _ in dl:
        chunks.append(capsnet_embed_160(capsnet, xb))
    return torch.cat(chunks, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# PR sidecar (stores per-seed heldout + test arrays for cached re-rendering)
# ─────────────────────────────────────────────────────────────────────────────

def _save_pr_sidecar(path: Path,
                     pr_heldout: dict,
                     pr_test: dict):
    """Save per-seed P_r arrays to an .npz sidecar for cached re-rendering."""
    seeds_h = sorted(pr_heldout.keys())
    radii = pr_heldout[seeds_h[0]][0]
    pr_gen_h = np.stack([pr_heldout[s][1] for s in seeds_h], axis=0)
    pr_ref_h = np.stack([pr_heldout[s][2] for s in seeds_h], axis=0)

    seeds_t = sorted(pr_test.keys())
    pr_gen_t = np.stack([pr_test[s][1] for s in seeds_t], axis=0)\
        if seeds_t else np.empty((0,))
    pr_ref_t = np.stack([pr_test[s][2] for s in seeds_t], axis=0)\
        if seeds_t else np.empty((0,))

    np.savez(path,
             seeds=np.asarray(seeds_h),
             radii=radii,
             pr_gen_heldout=pr_gen_h,
             pr_ref_heldout=pr_ref_h,
             pr_gen_test=pr_gen_t,
             pr_ref_test=pr_ref_t)
    print(f"Saved P_r sidecar → {path}")


def _load_pr_sidecar(path: Path):
    """Load sidecar; returns (pr_heldout, pr_test) dicts or ({}, {})."""
    if not path.exists():
        return {}, {}
    z = np.load(path)
    seeds = z["seeds"].tolist()
    radii = z["radii"]
    pr_h = {int(s): (radii, z["pr_gen_heldout"][i], z["pr_ref_heldout"][i])
            for i, s in enumerate(seeds)}
    pr_t: dict = {}
    if "pr_gen_test" in z and z["pr_gen_test"].ndim == 2:
        pr_t = {int(s): (radii, z["pr_gen_test"][i], z["pr_ref_test"][i])
                for i, s in enumerate(seeds)}
    return pr_h, pr_t


# ─────────────────────────────────────────────────────────────────────────────
# Per-model driver
# ─────────────────────────────────────────────────────────────────────────────

def _hyper_label(kind: str):
    if kind == "projection":
        hyper = (f"Deltas : {[int(d) for d in deltas]}  |  "
                 f"R_inners : {[int(r) for r in r_inners]}  |  "
                 f"channels={list(channels)}")
        label = "All Digits (IPA Projection Model)"
    else:
        hyper = (f"sigma={sigma}  |  epochs={n_diffusion}  "
                 f"|  sample_steps={diffusion_num_steps}  |  snr={diffusion_snr}")
        label = "Diffusion Model (VE-SDE / ScoreNet U-Net)"
    return hyper, label


def run_model(kind: str, capsnet, emb_test_shared, seeds=seeds,
              figures=None, *,
              force_retrain: bool = False,
              force_eval: bool = False):
    """Evaluate one model (projection or diffusion) across all seeds.

    Returns a dict with keys:
      'per_seed'     : {seed: metrics_dict}
      'pr_heldout'   : {seed: (radii, pr_gen, pr_ref_heldout)}
      'pr_test'      : {seed: (radii, pr_gen, pr_ref_test)}
      'gen_per_seed' : {seed: tensor(N,784)}
      'train_per_seed': {seed: tensor(N,784)}
    """
    assert kind in ("projection", "diffusion")
    base = projection_dir if kind == "projection" else diffusion_dir
    figures = figures if figures is not None else []

    print(f"\n{'#'*72}\n# {kind.upper()}  base={base}\n{'#'*72}")
    base.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    report_txt = base / "multiseed_report.txt"
    results_json = base / "multiseed_results.json"
    sidecar_path = base / "multiseed_pr_data.npz"

    use_cache = (not force_retrain) and (not force_eval)\
        and report_txt.exists() and results_json.exists()

    if use_cache:
        print(f"[CACHED] {kind}: loading report from {results_json.name}. "
              f"Pass --force-eval to recompute metrics.")
        raw = json.loads(results_json.read_text())
        per_seed = {int(k): v for k, v in raw.items()}
        wanted = [s for s in seeds if s in per_seed]
        hyper, label = _hyper_label(kind)
        print("\n" + _format_report(label, wanted, n_samples, per_seed, hyper))

        pr_heldout, pr_test = _load_pr_sidecar(sidecar_path)

        # NN figures (projection only): re-run _save_nn_figure using cached
        # generated samples and split files — no model inference, no CapsNet.
        gen_per_seed: dict = {}
        train_per_seed: dict = {}
        if kind == "projection":
            for s in wanted:
                if s != 0:
                    continue
                seed_dir = base / f"seed{s}"
                gen_cache = seed_dir / "proj_generated_samples.pt"
                split_file = seed_dir / "mnist_samples_split.pth"
                if not gen_cache.exists() or not split_file.exists():
                    print(f"  [SKIP NN seed{s}] cached tensors not found; "
                          "run with --force-eval once to populate them.")
                    continue
                gen = torch.load(gen_cache, map_location="cpu",
                                 weights_only=False).float()
                train_s, _ = _load_split(seed_dir, seed=s)
                gen_per_seed[s] = gen
                train_per_seed[s] = train_s
                out_png = figures_dir / f"nn_proj_seed{s}.png"
                f = _save_nn_figure(gen, train_s, out_png)
                if f is not None:
                    figures.append(f)

        return {"per_seed": per_seed, "pr_heldout": pr_heldout,
                "pr_test": pr_test,
                "gen_per_seed": gen_per_seed,
                "train_per_seed": train_per_seed}

    # ── Fresh evaluation path ─────────────────────────────────────────────
    per_seed: dict[int, dict] = {}
    pr_heldout: dict[int, tuple] = {}
    pr_test: dict[int, tuple] = {}
    gen_per_seed: dict[int, torch.Tensor] = {}
    train_per_seed: dict[int, torch.Tensor] = {}

    for s in seeds:
        seed_dir = base / f"seed{s}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- seed {s}  ({seed_dir.name}) ---")
        # Diffusion always uses the projection seed's split file: projection
        # and diffusion share the same 10 k training samples for every seed.
        split_dir = projection_dir / f"seed{s}" if kind == "diffusion" else seed_dir
        train_s, remain_s = _load_split(split_dir, seed=s,
                                        force_regen=force_retrain)
        if kind == "projection":
            gen = _generate_projection(s, train_s, seed_dir,
                                       force_retrain=force_retrain)
        else:
            gen = _load_diffusion_samples(seed_dir, seed=s,
                                          train_flat=train_s,
                                          force_retrain=force_retrain)

        print(f"  gen={tuple(gen.shape)}  train={tuple(train_s.shape)}  "
              f"remain={tuple(remain_s.shape)}")

        metrics, ph, pt = _evaluate_seed(
            s, gen, train_s, remain_s, capsnet, emb_test_shared)
        per_seed[s] = metrics
        gen_per_seed[s] = gen
        train_per_seed[s] = train_s
        if ph[0] is not None:
            pr_heldout[s] = ph
        if pt[0] is not None:
            pr_test[s] = pt

        for k, v in metrics.items():
            print(f"    {k:<16s} = {v:.5f}")

        # Per-seed NN figure (projection only) + cache generated samples
        if kind == "projection":
            torch.save(gen.cpu(), seed_dir / "proj_generated_samples.pt")
            if s == 0:
                try:
                    out_png = figures_dir / f"nn_proj_seed{s}.png"
                    f = _save_nn_figure(gen, train_s, out_png)
                    if f is not None:
                        figures.append(f)
                except Exception as e:
                    print(f"  [WARN] NN figure seed{s}: {e}")

    # Emit text report + JSON
    hyper, label = _hyper_label(kind)
    txt = _format_report(label, seeds, n_samples, per_seed, hyper)
    print("\n" + txt)
    report_txt.write_text(txt + "\n")
    results_json.write_text(
        json.dumps({str(k): v for k, v in per_seed.items()}, indent=2))

    # Save sidecar
    if pr_heldout:
        _save_pr_sidecar(sidecar_path, pr_heldout, pr_test)

    return {"per_seed": per_seed, "pr_heldout": pr_heldout,
            "pr_test": pr_test,
            "gen_per_seed": gen_per_seed,
            "train_per_seed": train_per_seed}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Multi-seed MNIST evaluation for IPA and Diffusion models.")
    ap.add_argument("--only", choices=["projection", "diffusion", "both"],
                    default="both")
    ap.add_argument("--no-show", action="store_true",
                    help="Save figures without calling plt.show().")
    ap.add_argument("--force-eval", action="store_true",
                    help="Recompute metrics/plots from split files + saved "
                    "checkpoints, even when a cached report exists.  "
                    "Does not force retraining; missing artifacts can still "
                    "trigger sampling or training.")
    ap.add_argument("--force-retrain", action="store_true",
                    help="Retrain BOTH projection and diffusion from scratch.")
    ap.add_argument("--force-retrain-projection", action="store_true",
                    help="Retrain only the projection (IPA) cascade.")
    ap.add_argument("--force-retrain-diffusion", action="store_true",
                    help="Retrain only the diffusion (ScoreNet) model.")
    args = ap.parse_args()

    force_proj = args.force_retrain or args.force_retrain_projection
    force_diff = args.force_retrain or args.force_retrain_diffusion
    force_eval = args.force_eval

    print(f"Device: {device}  |  seeds: {seeds}")
    print(f"force_eval={force_eval}  force_proj={force_proj}  "
          f"force_diff={force_diff}")

    figures_dir.mkdir(parents=True, exist_ok=True)

    todo = (["projection", "diffusion"] if args.only == "both"
            else [args.only])

    # Set up CapsNet if any fresh evaluation is needed
    needs_fresh = []
    for k in todo:
        base = projection_dir if k == "projection" else diffusion_dir
        fk = force_proj if k == "projection" else force_diff
        if fk or force_eval or not (
                (base / "multiseed_report.txt").exists()
                and (base / "multiseed_results.json").exists()):
            needs_fresh.append(k)

    if needs_fresh:
        capsnet, ok = _try_setup_capsnet()
        emb_test = _shared_test_embedding(capsnet) if ok else None
        if ok:
            print(f"Shared MNIST test embedding: {tuple(emb_test.shape)}")
    else:
        capsnet, emb_test, ok = None, None, False
        print("[INFO] All cached — CapsNet not loaded. "
              "Pass --force-eval to recompute metrics.")

    figures: list = []
    results: dict = {}

    for k in todo:
        fk = force_proj if k == "projection" else force_diff
        try:
            results[k] = run_model(
                k, capsnet, emb_test,
                seeds=seeds, figures=figures,
                force_retrain=fk,
                force_eval=force_eval)
        except FileNotFoundError as e:
            print(f"[SKIP] {k}: {e}")

    # ── Cross-model combined figures ──────────────────────────────────────
    r_proj = results.get("projection", {})
    r_diff = results.get("diffusion", {})

    # Aggregate test-reference PR: both models in one figure
    pr_test_proj = r_proj.get("pr_test", {})
    pr_test_diff = r_diff.get("pr_test", {})
    if pr_test_proj or pr_test_diff:
        f = _save_aggregate_pr_test_both(
            pr_test_proj, pr_test_diff,
            figures_dir / "aggregate_pr_curve_test_reference.png")
        if f is not None:
            figures.append(f)

    # 2-panel PR for seed 0 only: left=held-out ref, right=MNIST test ref
    pr_heldout_proj = r_proj.get("pr_heldout", {})
    pr_heldout_diff = r_diff.get("pr_heldout", {})
    for s in [0]:
        ph_p = pr_heldout_proj.get(s)
        ph_d = pr_heldout_diff.get(s)
        pt_p = pr_test_proj.get(s)
        pt_d = pr_test_diff.get(s)
        if ph_p is None and ph_d is None and pt_p is None and pt_d is None:
            continue
        out_png = figures_dir / f"combined_pr_seed{s}_2panel.png"
        f = _save_per_seed_pr_2panel(s, ph_p, ph_d, pt_p, pt_d, out_png)
        if f is not None:
            figures.append(f)

    if figures and not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()

    return results


if __name__ == "__main__":
    main()
