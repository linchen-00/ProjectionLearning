"""Evaluation metrics: coverage (Pr), projection error, memorization."""

import torch
import numpy as np
from .utils import distance

# ========================= Coverage metric Pr =========================


def Pr(train_samples, gen_samples, r):
    """Fraction of generated samples within radius r of the training set.

    Args:
        train_samples: (N, d)
        gen_samples: (Ns, d)
        r: array-like of radii

    Returns:
        numpy array of shape (len(r),)
    """
    d = distance(gen_samples, train_samples).view(-1, 1)
    r_t = torch.as_tensor(r, dtype=d.dtype, device=d.device).view(1, -1)
    counts = (d < r_t).sum(dim=0).float()
    return (counts / d.shape[0]).cpu().numpy()


# ========================= Exact Pr for circle =========================
def _lexsort(x):
    """Lexicographic sort on a 2-column tensor."""
    idx = torch.arange(x.size(0), device=x.device)
    idx = idx[torch.argsort(x[idx, 1], stable=True)]
    idx = idx[torch.argsort(x[idx, 0], stable=True)]
    return x[idx]


def Pr_true_circle(theta, r, radius=1):
    """Exact Pr for uniform distribution on a circle.

    Args:
        theta: (N,) or (N,1) angles of training samples
        r: array-like of radii
        radius: circle radius

    Returns:
        numpy array of shape (len(r),)
    """
    theta = theta.view(-1, 1) * radius
    thetas = torch.clamp(theta[:, 0], min=0.0, max=2 * torch.pi * radius)
    r_t = torch.as_tensor(r, dtype=theta.dtype, device=theta.device)
    if r_t.ndim == 0:
        r_t = r_t.unsqueeze(0)
    lr_t = 2.0 * radius * torch.arcsin(r_t / (2.0 * radius))

    probs = []
    for lr in lr_t:
        openings = torch.maximum(thetas - lr, torch.tensor(0.0))
        closings = torch.minimum(
            thetas + lr, torch.tensor(2 * torch.pi * radius))
        left_openings = torch.tensor(2 * torch.pi * radius) + torch.minimum(
            thetas - lr, torch.tensor(0.0)
        )
        left_closings = 2 * torch.pi * radius * torch.ones_like(thetas)
        right_openings = torch.zeros_like(thetas)
        right_closings = torch.maximum(
            thetas + lr - 2 * torch.pi * radius, torch.tensor(0.0)
        )

        total_openings = torch.cat([left_openings, openings, right_openings])
        total_closings = torch.cat([left_closings, closings, right_closings])

        points = torch.cat([
            torch.cat([
                total_openings[:, None],
                -torch.ones_like(total_openings)[:, None],
            ], dim=-1),
            torch.cat([
                total_closings[:, None],
                torch.ones_like(total_closings)[:, None],
            ], dim=-1),
        ], dim=0)
        points = _lexsort(points)
        states = -torch.cumsum(points[:, 1], dim=-1)[:-1]
        increments = points[1:, 0] - points[:-1, 0]
        probs.append(((states > 0) * increments).sum() /
                     (2 * torch.pi * radius))

    return torch.stack(probs).cpu().numpy()


# ========================= Projection error & memorization =========================
def projection_error(gen_samples, exact_proj, train_samples):
    """PErr = E|T(x) - Pi(x)|^2 / trace(Var(training_samples)).

    Var is the unbiased covariance matrix of training_samples.
    """
    num = (torch.linalg.norm(gen_samples - exact_proj, dim=1) ** 2).mean()
    # trace of unbiased covariance
    denom = torch.var(train_samples, dim=0, correction=1).sum()
    return (num / denom).item()


def memorization_score_integral(Pr_gen_values, Pr_true_values, r_vector):
    """Integral $P_r$-gap memorization score.

    .. math::

        \\mathrm{Memo} \\;=\\;
        \\frac{\\int_{0}^{r_{\\max}}
              \\bigl(P_r^{\\mathrm{gen}}(r) - P_r^{\\mathrm{true}}(r)\\bigr)_+\\,dr}
             {\\int_{0}^{r_{\\max}}
              \\bigl(1 - P_r^{\\mathrm{true}}(r)\\bigr)\\,dr},

    with :math:`r_{\\max}` taken as the largest radius in ``r_vector``.

    The numerator is the area of the (positive part of the) gap
    :math:`P_r^{\\mathrm{gen}} - P_r^{\\mathrm{true}}` over the swept radius
    range, and the denominator normalises by the area above the reference
    curve, which is the maximum value the numerator can attain.
    The score is dimensionless and bounded in ``[0, 1]``.

    Args:
        Pr_gen_values:  (n,) array of :math:`P_r^{\\mathrm{gen}}(r)` values.
        Pr_true_values: (n,) array of :math:`P_r^{\\mathrm{true}}(r)` values.
        r_vector:       (n,) array of evaluation radii (monotonically
                         non-decreasing).

    Returns:
        float in ``[0, 1]``.
    """
    Pr_gen_values = np.asarray(Pr_gen_values, dtype=np.float64)
    Pr_true_values = np.asarray(Pr_true_values, dtype=np.float64)
    r_vector = np.asarray(r_vector, dtype=np.float64)
    num = np.trapezoid(np.maximum(Pr_gen_values - Pr_true_values, 0.0),
                       r_vector)
    den = np.trapezoid(1.0 - Pr_true_values, r_vector)
    if den <= 0:
        return 0.0
    return float(num / den)


# ========================= Torus-specific =========================


def dist_torus_embedded(X, torus_dim, ambient_dim, threshold=0.01):
    """Mean L^2 distance of points to the flat torus embedded in R^{ambient_dim}.

    Computes mean over samples of per-sample distance,
    where per-sample distance = sqrt(sum_i (r_i - 1)² + ||extra dims||²).

    Args:
        X: tensor of shape (N, ambient_dim)
        torus_dim: number of S^1 factors
        ambient_dim: ambient dimension
        threshold: pass/fail threshold

    Returns:
        (mean_distance, is_good)
    """
    d_sq = torch.zeros(X.shape[0], device=X.device)
    for i in range(torus_dim):
        r_i = torch.sqrt(X[:, 2 * i] ** 2 + X[:, 2 * i + 1] ** 2)
        d_sq += (r_i - 1) ** 2
    if ambient_dim > 2 * torus_dim:
        d_sq += (X[:, 2 * torus_dim:] ** 2).sum(dim=1)
    val = d_sq.sqrt().mean().item()
    return val, val <= threshold


# ========================= Memorization (ratio-based) =========================


def memorization_score_ratio(gen_samples, train_samples, k=1/3,
                             batch_size=256, eps=1e-12):
    """Memorization score based on nearest/second-nearest distance ratio.

    Score = E[ 1{ d1(x) / d2(x) < k } ]

    Args:
        gen_samples: (N_gen, D) or (N_gen, C, H, W)
        train_samples: (N_train, D) or (N_train, C, H, W)
        k: threshold ratio (default 1/3)
        batch_size: chunk size to avoid OOM
        eps: small constant to avoid division by zero

    Returns:
        (score, ratios, nn1_idx, nn2_idx)
    """
    device = gen_samples.device
    train_samples = train_samples.to(device)

    gen_flat = gen_samples.view(
        gen_samples.shape[0], -1) if gen_samples.dim() > 2 else gen_samples
    train_flat = train_samples.view(
        train_samples.shape[0], -1) if train_samples.dim() > 2 else train_samples

    ratios_all, nn1_all, nn2_all = [], [], []

    for start in range(0, gen_flat.shape[0], batch_size):
        end = min(start + batch_size, gen_flat.shape[0])
        x = gen_flat[start:end]
        dist = torch.cdist(x, train_flat, p=2)
        vals, idx = torch.topk(dist, k=2, dim=1, largest=False)
        ratio = vals[:, 0] / (vals[:, 1] + eps)
        ratios_all.append(ratio.detach().cpu())
        nn1_all.append(idx[:, 0].detach().cpu())
        nn2_all.append(idx[:, 1].detach().cpu())

    ratios = torch.cat(ratios_all)
    nn1_idx = torch.cat(nn1_all)
    nn2_idx = torch.cat(nn2_all)
    score = (ratios < k).float().mean().item()
    return score, ratios, nn1_idx, nn2_idx
