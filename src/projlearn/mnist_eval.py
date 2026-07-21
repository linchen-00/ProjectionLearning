"""MNIST evaluation helpers used by ``experiments/06_mnist_multiseed.py``.

Public surface:

* :func:`compose`          — applies the b-snapshot of every stage left-to-right
* :func:`load_digit_models`— loads the 6 cascade snapshots from disk
* :func:`try_setup_capsnet`— locates / extracts / loads the CapsNet feature net
* :func:`capsnet_embed_160`— maps (N,784) images to 160-d capsule embeddings
* :func:`collect_embeddings_from_tensor` — convenience wrapper of the above
* :func:`stats_from_embeddings`, :func:`fid_from_stats`,
  :func:`kid_from_embeddings`,  :func:`compute_pr_curves` — FID / KID / P_r

The IPA cascade architecture (:class:`projlearn.models.ProjectionUNet`) lives
in :mod:`projlearn.models`.
"""

from __future__ import annotations

import glob
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch

from projlearn.evaluation import memorization_score_integral
from projlearn.models import ProjectionUNet

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def compose(models, x, indices=None):
    """Apply models left-to-right; without indices, applies odd-indexed (b) snapshots."""
    if indices is None:
        for i in range(1, len(models), 2):
            x = models[i](x)
    else:
        for i in indices:
            x = models[i](x)
    return x


def load_digit_models(ckpt_dir, stage_specs, channels, device):
    """Load 2*K projection-cascade snapshots from disk.

    Two filename layouts are recognised:

    * **Legacy** (Colab notebook output): ``model{i}{a|b}_epoch{epoch}.pth``
      where ``a`` is the pretrain phase (epoch ``stage_specs[i]["iter1"]``)
      and ``b`` is the projection phase (epoch ``stage_specs[i]["iter2"]``).
    * **Algorithm 2 default**: ``alg2_stage{i}_pretrain_epoch{iter1}.pth``
      and ``alg2_stage{i}_project_epoch{iter2}.pth`` (the names produced by
      :func:`projlearn.training.train_projection_algorithm2` with the
      default ``save_prefix="alg2"``).

    Returns ``[m0a, m0b, m1a, m1b, ...]``.
    """
    ckpt_dir = Path(ckpt_dir)
    models = []
    for i, spec in enumerate(stage_specs):
        for phase, epoch in (("a", spec["iter1"]), ("b", spec["iter2"])):
            legacy = ckpt_dir / f"model{i}{phase}_epoch{epoch}.pth"
            phase_name = "pretrain" if phase == "a" else "project"
            new = ckpt_dir / f"alg2_stage{i}_{phase_name}_epoch{epoch}.pth"
            if legacy.exists():
                ckpt = legacy
            elif new.exists():
                ckpt = new
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found.  Tried:\n  {legacy}\n  {new}\n"
                    "Run training first or supply the missing checkpoint.")
            m = ProjectionUNet(image_size=28, channels=channels)
            m.load_state_dict(
                torch.load(ckpt, map_location=device, weights_only=False)[
                    "model_state_dict"]
            )
            models.append(m.to(device).eval())
    return models


# ─────────────────────────────────────────────────────────────────────────────
# CapsNet setup / embedding
# ─────────────────────────────────────────────────────────────────────────────

def try_setup_capsnet(repo_dir=None, ckpt_dir=None):
    """Load the pre-trained CapsNet feature extractor.

    Both directories are resolved relative to the supplement root if not given.
    Returns ``(capsnet_or_None, ok_bool)``.  Failure is non-fatal: callers
    should fall back to pixel-only metrics.
    """
    root = Path(__file__).resolve().parent.parent
    repo_dir = str(repo_dir or root / "capsule-net-pytorch")
    ckpt_dir = str(ckpt_dir or root / "capsnet_ckpt")
    try:
        if not os.path.exists(repo_dir):
            print("Cloning capsule-net-pytorch …")
            os.system(
                "git clone -q "
                f"https://github.com/cedrickchee/capsule-net-pytorch.git {repo_dir}")
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        from model import Net  # noqa: E402

        os.makedirs(ckpt_dir, exist_ok=True)
        archive_path = os.path.join(ckpt_dir, "model_epoch_50.tar.gz")
        extract_dir = os.path.join(ckpt_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        if not os.path.exists(archive_path) and not os.path.exists(
                os.path.join(extract_dir, ".extracted_ok")):
            import gdown  # noqa: F401
            print("Downloading CapsNet checkpoint …")
            gdown.download(id="1lYtOMSreP4I9hr9un4DsBJZrzodI6l2d",
                           output=archive_path, quiet=False)
        marker = os.path.join(extract_dir, ".extracted_ok")
        if not os.path.exists(marker) and os.path.exists(archive_path):
            with tarfile.open(archive_path, "r:gz") as tar:
                members = [m for m in tar.getmembers()
                           if not os.path.basename(m.name).startswith("._")]
                tar.extractall(extract_dir, members=members, filter="data")
            open(marker, "w").close()
        cands = (glob.glob(extract_dir + "/**/*.pth", recursive=True)
                 + glob.glob(extract_dir + "/**/*.pt",  recursive=True)
                 + glob.glob(extract_dir + "/**/*.tar", recursive=True))
        if not cands:
            raise FileNotFoundError(
                "No CapsNet checkpoint found after extraction.")
        caps_ckpt_path = sorted(
            cands,
            key=lambda p: (("model" not in p.lower())
                           + ("epoch" not in p.lower()), len(p)),
        )[0]
        print("Using CapsNet checkpoint:", caps_ckpt_path)
        cuda_enabled = torch.cuda.is_available()
        caps_device = torch.device(
            "cuda") if cuda_enabled else torch.device("cpu")
        capsnet = Net(
            num_conv_in_channel=1, num_conv_out_channel=256, num_primary_unit=8,
            primary_unit_size=1152, num_classes=10, output_unit_size=16,
            num_routing=3, use_reconstruction_loss=True,
            regularization_scale=0.0005, input_width=28, input_height=28,
            cuda_enabled=cuda_enabled,
        ).to(caps_device).eval()
        capsnet._caps_device = caps_device
        ckpt = torch.load(caps_ckpt_path, map_location="cpu",
                          weights_only=False)
        state = ckpt
        for k in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break
        if any(key.startswith("module.") for key in state):
            state = {key[len("module."):]: v for key, v in state.items()}
        for i in (1, 2, 3):
            for suf in ("weight", "bias"):
                src, dst = f"fc{i}.{suf}", f"decoder.fc{i}.{suf}"
                if src in state and dst not in state:
                    state[dst] = state.pop(src)
        missing, unexpected = capsnet.load_state_dict(state, strict=False)
        print(
            f"CapsNet loaded — missing: {len(missing)}, unexpected: {len(unexpected)}")
        return capsnet, True
    except Exception as e:  # pragma: no cover — best-effort optional dep
        print(f"\n[WARNING] CapsNet setup failed ({type(e).__name__}: {e}).\n")
        return None, False


@torch.no_grad()
def capsnet_embed_160(capsnet, x_flat_or_1chw, batch_size=256):
    """Embed (N,784) flat or (N,1,28,28) images → (N,160) capsule vectors on CPU."""
    caps_device = getattr(capsnet, "_caps_device",
                          next(capsnet.parameters()).device)
    if x_flat_or_1chw.dim() == 2:
        x_flat_or_1chw = x_flat_or_1chw.view(-1, 1, 28, 28)
    chunks = []
    for i in range(0, x_flat_or_1chw.size(0), batch_size):
        xb = ((x_flat_or_1chw[i:i + batch_size].to(caps_device) - MNIST_MEAN)
              / MNIST_STD)
        out = capsnet(xb)
        caps = out[0] if isinstance(out, (tuple, list)) else out
        if caps.dim() == 4 and caps.size(-1) == 1:
            caps = caps.squeeze(-1)
        chunks.append(caps.reshape(caps.size(0), -1).cpu())
    return torch.cat(chunks, dim=0)


def collect_embeddings_from_tensor(capsnet, flat_tensor, batch_size=256):
    """Embed a (N,784) tensor through CapsNet → (N,160) on CPU."""
    return capsnet_embed_160(capsnet, flat_tensor.float().cpu(),
                             batch_size=batch_size)


# ─────────────────────────────────────────────────────────────────────────────
# FID / KID / P_r in embedding space
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def stats_from_embeddings(emb, eps=1e-12):
    emb = emb.detach().to("cpu", dtype=torch.float64)
    mu = emb.mean(dim=0)
    ctr = emb - mu
    Sig = ctr.t() @ ctr / (emb.size(0) - 1)
    Sig = 0.5 * (Sig + Sig.t()) + eps * torch.eye(
        emb.size(1), dtype=torch.float64)
    return mu, Sig


def _sqrtm_psd(A, eps=1e-12):
    A = 0.5 * (A + A.t())
    w, V = torch.linalg.eigh(A)
    return (V * w.clamp_min(eps).sqrt()) @ V.t()


def fid_from_stats(mu_r, Sig_r, mu_g, Sig_g, eps=1e-12):
    diff = mu_r - mu_g
    sqSr = _sqrtm_psd(Sig_r, eps=eps)
    sqA = _sqrtm_psd(sqSr @ Sig_g @ sqSr, eps=eps)
    return max(float((diff.dot(diff) + Sig_r.trace() + Sig_g.trace()
                      - 2 * sqA.trace()).item()), 0.0)


def _poly_kernel(a, b, deg=3):
    return ((a @ b.t()) / a.size(1) + 1.0) ** deg


def kid_from_embeddings(real_emb, fake_emb, subset_size=1000,
                        n_subsets=20, seed=123, degree=3):
    real = real_emb.detach().to("cpu", dtype=torch.float64)
    fake = fake_emb.detach().to("cpu", dtype=torch.float64)
    Nr, Ng = real.size(0), fake.size(0)
    s = min(subset_size, Nr, Ng)
    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(n_subsets):
        X = real[torch.randperm(Nr, generator=g)[:s]]
        Y = fake[torch.randperm(Ng, generator=g)[:s]]
        Kxx = _poly_kernel(X, X, degree)
        Kyy = _poly_kernel(Y, Y, degree)
        Kxy = _poly_kernel(X, Y, degree)
        mmd = ((Kxx.sum() - Kxx.diag().sum()) / (s * (s - 1))
               + (Kyy.sum() - Kyy.diag().sum()) / (s * (s - 1))
               - 2.0 * Kxy.mean())
        vals.append(float(mmd))
    return max(float(np.mean(vals)), 0.0)


def compute_pr_curves(train_emb, fake_emb, ref_emb, n_thresholds=500):
    """Compute ECDF-based P_r curves and the integral memorisation score.

    Returns (radii, pr_gen, pr_ref, memo) on a shared linspace [0, 1] grid.
    """
    from sklearn.metrics import pairwise_distances
    train = np.asarray(train_emb, dtype=np.float32)
    fake = np.asarray(fake_emb, dtype=np.float32)
    ref = np.asarray(ref_emb, dtype=np.float32)

    d_gen = pairwise_distances(fake, train, metric="euclidean",
                               n_jobs=-1).min(axis=1)
    d_ref = pairwise_distances(ref, train, metric="euclidean",
                               n_jobs=-1).min(axis=1)

    upper = 1
    radii = np.linspace(0.0, upper, n_thresholds, dtype=np.float32)
    pr_gen = np.array([(d_gen <= r).mean() for r in radii], dtype=np.float32)
    pr_ref = np.array([(d_ref <= r).mean() for r in radii], dtype=np.float32)
    memo = memorization_score_integral(pr_gen, pr_ref, radii)
    return radii, pr_gen, pr_ref, memo
