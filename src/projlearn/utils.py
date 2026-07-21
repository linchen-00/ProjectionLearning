"""Low-level utilities: distance computations and sampling primitives."""

import torch
import math


def sample_delta_ball(N, delta, ambient_dim):
    """Sample N points from a ball of radius delta in R^ambient_dim."""
    Z = torch.randn(ambient_dim, N)
    W = Z / Z.norm(dim=0, keepdim=True)
    U = torch.rand(1, N)
    r = U ** (1 / ambient_dim)
    x = W * r
    return delta * x.T


def distance(output, samples, block=1024):
    """Min distance from each point in output to the nearest sample.

    Args:
        output: (batch_size, dim)
        samples: (N, dim)

    Returns:
        (batch_size, 1) tensor of nearest-neighbor distances.
    """
    out_min = []
    for x in output.split(block, dim=0):
        m = torch.full((x.size(0),), float("inf"),
                       device=x.device, dtype=x.dtype)
        for y in samples.split(block, dim=0):
            m = torch.minimum(m, torch.cdist(x, y).min(dim=1).values)
        out_min.append(m)
    return torch.cat(out_min).view(-1, 1)


def argmin_xi(x, samples):
    """Nearest sample for each query point.

    Args:
        x: (batch_size, dim)
        samples: (N, dim)

    Returns:
        (batch_size, dim) — the nearest sample for each x.
    """
    dists = torch.cdist(x, samples)
    return samples[dists.argmin(dim=1)]


def second_neighbor_stats(samples):
    """Return mean and max distance to the second nearest neighbor."""
    dists = torch.cdist(samples, samples)
    dists.fill_diagonal_(float("inf"))
    knn_dists, _ = dists.topk(2, largest=False, dim=1)
    return knn_dists[:, 1].mean().item(), knn_dists[:, 1].max().item()
