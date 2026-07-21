"""Loss functions for projection learning and score matching."""

import torch
from torch.func import vmap, jvp

from .utils import argmin_xi


def projection_loss(model, samples, smoothed_samples, smooth_output,
                    loss_weights, r_inner=0.4, delta=None,
                    c2_mode="exact", c2_subsample=0.2, c2_fd_eps=1e-3,
                    all_samples=None):
    """Compute the projection loss E(T) with terms C1–C4.

    Args:
        model: projection neural network
        samples: (N, dim) training samples on the manifold
        smoothed_samples: (batch, dim) noisy points x ~ mu_n * eta_delta
        smooth_output: (batch, dim) model outputs T(x)
        loss_weights: dict {"C1": w1, ..., "C4": w4}; missing keys default to 0
        r_inner: inner radius for C3
        delta: tubular-neighborhood radius, required when C4 > 0
        c2_mode: "exact" (vmap jvp), "subsample" (random subset + vmap jvp),
                 "fd" (finite-difference approx), or "stopgrad" (detach the
                 evaluation point: J_T(sg(T(x))) · (T(x) - x))
        c2_subsample: fraction of points for "subsample" mode (0 < s <= 1)
        c2_fd_eps: epsilon for "fd" (finite-difference) mode
        all_samples: optional full training dataset (N, dim) used for argmin in C3;
            when provided, argmin is computed over all_samples instead of the
            mini-batch `samples`.  This is the correct behaviour when training
            with a DataLoader so that C3 finds the true nearest training point
            rather than the nearest point in the current mini-batch.

    Returns:
        (total_loss, C1, C2, C3, C4)
    """
    w = {f"C{i}": loss_weights.get(f"C{i}", 0.0) for i in range(1, 5)}

    vanilla_output = model(samples)
    zero = torch.tensor(0.0, device=samples.device)
    C1 = C2 = C3 = C4 = zero

    # C1: data fidelity — T(x_i) ≈ x_i for training samples
    C1 = w["C1"] * (torch.norm(vanilla_output - samples, dim=-1) ** 2).mean()

    # C2: Jacobian-vector product penalty
    if w["C2"] > 0:
        Tx = smooth_output
        v = smooth_output - smoothed_samples
        eval_pt = Tx

        if c2_mode == "stopgrad":
            eval_pt_sg = eval_pt.detach()

            def Jv_product(x, v):
                _, Jv = jvp(model, (x,), (v,))
                return Jv

            Jv = vmap(Jv_product)(eval_pt_sg, v)
        elif c2_mode == "fd":
            Jv = (model(eval_pt + c2_fd_eps * v) - model(eval_pt)) / c2_fd_eps
        else:
            def Jv_product(x, v):
                _, Jv = jvp(model, (x,), (v,))
                return Jv

            if c2_mode == "subsample":
                k = max(1, int(len(eval_pt) * c2_subsample))
                idx = torch.randperm(len(eval_pt), device=eval_pt.device)[:k]
                Jv = vmap(Jv_product)(eval_pt[idx], v[idx])
            else:  # "exact"
                Jv = vmap(Jv_product)(eval_pt, v)

        C2 = w["C2"] * (Jv.norm(dim=-1) ** 2).mean()

    # C3: nearest-sample penalty — ReLU(|T(x) - argmin_{x_i}|x - x_i|| - r_inner)
    # Use all_samples for proper full-dataset argmin when operating in mini-batch mode.
    if w["C3"] > 0:
        ref = all_samples if all_samples is not None else samples
        nearest = argmin_xi(smoothed_samples, ref)
        C3 = w["C3"] * torch.relu(torch.norm(smooth_output -
                                  nearest, dim=-1) - r_inner).mean()

    # C4: displacement cap — ReLU(|T(x) - x| - delta)
    if w["C4"] > 0:
        if delta is None:
            raise ValueError("delta must be provided when C4 weight > 0")
        C4 = w["C4"] * torch.relu(
            torch.norm(smooth_output - smoothed_samples, dim=-1) - delta
        ).mean()

    return C1 + C2 + C3 + C4, C1, C2, C3, C4


def score_matching_loss(model, x, marginal_prob_std, eps=1e-5):
    """Denoising score matching loss (one t per sample)."""
    random_t = torch.rand(x.shape[0], device=x.device) * (1.0 - eps) + eps
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    if x.dim() == 4:
        std_view = std[:, None, None, None]
    else:
        std_view = std[:, None]
    perturbed_x = x + z * std_view
    score = model(perturbed_x, random_t)
    dims = tuple(range(1, x.dim()))
    return torch.mean(torch.sum((score * std_view + z) ** 2, dim=dims))


def score_matching_loss_multi_t(model, x, marginal_prob_std, M=4, eps=1e-5):
    """Denoising score matching loss with M time samples per data point."""
    N = x.shape[0]
    x_rep = x.repeat_interleave(M, dim=0)
    random_t = torch.rand(N * M, device=x.device) * (1.0 - eps) + eps
    z = torch.randn_like(x_rep)
    std = marginal_prob_std(random_t)
    if x.dim() == 4:
        std_view = std.view(N * M, 1, 1, 1)
    else:
        std_view = std.view(N * M, 1)
    perturbed_x = x_rep + z * std_view
    score = model(perturbed_x, random_t)
    dims = tuple(range(1, x.dim()))
    return torch.mean(torch.sum((score * std_view + z) ** 2, dim=dims))
