"""Manifold generators and graph-based image manifolds."""

import torch
import numpy as np


# ========================= 2D curves =========================


def circle_2d(N=100, ambient_dim=2, radius=1):
    """Sample N points from a circle of given radius."""
    theta = 2 * torch.pi * torch.rand(N, 1)
    x = radius * torch.sin(theta)
    y = radius * torch.cos(theta)
    return torch.cat((x, y), dim=1)


# ========================= Higher-dimensional =========================


def torus(N=500, ambient_dim=8):
    """k-torus S^1 x ... x S^1 embedded in R^{2k} where k = ambient_dim // 2."""
    dim = ambient_dim // 2
    alpha = 2 * torch.pi * torch.rand(N, dim)
    samples = torch.zeros(N, 2 * dim)
    for i in range(dim):
        samples[:, 2 * i] = torch.cos(alpha[:, i])
        samples[:, 2 * i + 1] = torch.sin(alpha[:, i])
    return samples


def torus_and_circle_3d(N1=400, N2=100, R1=1.0, R2=0.4, Rc=1.0):
    """Sample a union of a torus surface and a circle curve in R^3.

    Torus:
        x = (R1 + R2*cos(v))*cos(u)
        y = (R1 + R2*cos(v))*sin(u)
        z = R2*sin(v)

    Circle (through origin, centered at (Rc, 0, 0), radius Rc):
        x = Rc + Rc*cos(t)
        y = 0
        z = Rc*sin(t)

    Args:
        N1: number of torus points
        N2: number of circle points
        R1: torus major radius
        R2: torus tube radius
        Rc: circle center x-coordinate and radius

    Returns:
        Tensor of shape (N1 + N2, 3)
    """
    # Torus: area-uniform sampling in v via rejection.
    u = 2 * torch.pi * torch.rand(N1)
    v = torch.empty(N1)
    accepted = 0
    while accepted < N1:
        remaining = N1 - accepted
        n_prop = int(remaining * (R1 + R2) / R1 * 1.3) + 64
        v_prop = 2 * torch.pi * torch.rand(n_prop)
        prob = (R1 + R2 * torch.cos(v_prop)) / (R1 + R2)
        mask = torch.rand(n_prop) < prob
        good = v_prop[mask]
        n_take = min(len(good), remaining)
        v[accepted:accepted + n_take] = good[:n_take]
        accepted += n_take

    torus_pts = torch.zeros(N1, 3)
    torus_pts[:, 0] = (R1 + R2 * torch.cos(v)) * torch.cos(u)
    torus_pts[:, 1] = (R1 + R2 * torch.cos(v)) * torch.sin(u)
    torus_pts[:, 2] = R2 * torch.sin(v)

    t = 2 * torch.pi * torch.rand(N2)
    circle_pts = torch.zeros(N2, 3)
    circle_pts[:, 0] = Rc + Rc * torch.cos(t)
    circle_pts[:, 1] = 0.0
    circle_pts[:, 2] = Rc * torch.sin(t)

    return torch.cat([torus_pts, circle_pts], dim=0)


# ========================= Graph-based (image) manifolds =========================


def _make_space_grid(resolution, lo, hi, device="cpu"):
    """Uniform 2D grid on [lo, hi]^2 at cell midpoints."""
    z = torch.linspace(lo, hi, resolution + 1, device=device)
    midpoints = (z[:-1] + z[1:]) / 2
    X, Y = torch.meshgrid(midpoints, midpoints, indexing="xy")
    return torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)


def graph_visualization(N, ambient_dim, manifold_fn, lo, hi, time_lo, time_hi,
                        device="cpu", intrinsic_dim=1, return_params=False):
    """Generate N image samples by evaluating manifold_fn on a spatial grid.

    Args:
        N: number of samples
        ambient_dim: must be a perfect square (resolution^2)
        manifold_fn: callable(grid, t) or callable(grid, t, s)
        lo, hi: spatial domain bounds
        time_lo, time_hi: parameter domain bounds
        intrinsic_dim: 1 or 2
        return_params: if True, also return the sampled parameter values
    """
    resolution = int(np.sqrt(ambient_dim))
    assert resolution ** 2 == ambient_dim, "ambient_dim must be a perfect square"

    grid = _make_space_grid(resolution, lo, hi, device=device)
    manifold = torch.zeros(N, grid.shape[0], device=device)

    if intrinsic_dim == 1:
        times = time_lo + (time_hi - time_lo) * torch.rand(N, device=device)
        for i, t in enumerate(times):
            manifold[i] = manifold_fn(grid, t)
        return (manifold, times) if return_params else manifold

    if intrinsic_dim == 2:
        t_vals = time_lo + (time_hi - time_lo) * torch.rand(N, device=device)
        s_vals = time_lo + (time_hi - time_lo) * torch.rand(N, device=device)
        params = torch.stack([t_vals, s_vals], dim=1)
        for i in range(N):
            manifold[i] = manifold_fn(grid, params[i, 0], params[i, 1])
        return (manifold, params) if return_params else manifold

    raise ValueError(f"intrinsic_dim must be 1 or 2, got {intrinsic_dim}")


def circular_movement(N, ambient_dim=64, device="cpu"):
    """Linear combination moving on a circle, as images."""
    def fn(x, t): return 0.5 + 0.5 / np.sqrt(2) * (
        x[..., 0] * torch.cos(t) + x[..., 1] * torch.sin(t)
    )
    return graph_visualization(N, ambient_dim, fn, -1, 1, 0, 2 * torch.pi, device)


# ========================= Basis functions =========================


def basis_function(func, ambient_dim=64, lo=-1.0, hi=1.0, device="cpu"):
    """Evaluate a scalar function on the image grid, returning a flat vector."""
    resolution = int(np.sqrt(ambient_dim))
    assert resolution ** 2 == ambient_dim
    grid = _make_space_grid(resolution, lo, hi, device=device)
    return func(grid)


def basis_matrix(basis_functions, ambient_dim=64, lo=-1.0, hi=1.0, device="cpu"):
    """Build (n_basis, ambient_dim) matrix of basis function evaluations."""
    output = torch.zeros(len(basis_functions), ambient_dim, device=device)
    for i, func in enumerate(basis_functions):
        output[i] = basis_function(func, ambient_dim, lo, hi, device)
    return output


# ========================= Torus with density & embedding =========================


def _rejection_sample_cos_density(N, a):
    """Sample N values from density (1 + a*cos(α)) / (2π) on [0, 2π) via rejection."""
    samples = torch.empty(N)
    num_accepted = 0
    while num_accepted < N:
        remaining = N - num_accepted
        n_proposal = int(remaining * (1 + a) * 1.2) + 64
        proposal = 2 * torch.pi * torch.rand(n_proposal)
        accept_prob = (1 + a * torch.cos(proposal)) / (1 + a)
        accepted = proposal[torch.rand(n_proposal) < accept_prob]
        n_take = min(len(accepted), remaining)
        samples[num_accepted:num_accepted + n_take] = accepted[:n_take]
        num_accepted += n_take
    return samples


def torus_embedded(N=500, torus_dim=2, ambient_dim=8, density_params=None):
    """Sample a torus of intrinsic dimension torus_dim embedded in R^ambient_dim.

    The torus lives in the first 2*torus_dim coordinates as
    (cos(α₁), sin(α₁), ..., cos(α_d), sin(α_d)), with remaining coordinates zero.

    Args:
        N: number of samples
        torus_dim: number of S^1 factors
        ambient_dim: must be >= 2 * torus_dim
        density_params: optional list of length torus_dim with values in (0,1].
            If provided, α_i ~ (1 + a_i cos(α)) / (2π). If None, uniform.

    Returns:
        tensor of shape (N, ambient_dim)
    """
    assert ambient_dim >= 2 * torus_dim, \
        f"ambient_dim ({ambient_dim}) must be >= 2 * torus_dim ({2 * torus_dim})"

    if density_params is not None:
        assert len(density_params) == torus_dim
        alpha = torch.stack(
            [_rejection_sample_cos_density(N, a) for a in density_params], dim=1
        )
    else:
        alpha = 2 * torch.pi * torch.rand(N, torus_dim)

    samples = torch.zeros(N, ambient_dim)
    for i in range(torus_dim):
        samples[:, 2 * i] = torch.cos(alpha[:, i])
        samples[:, 2 * i + 1] = torch.sin(alpha[:, i])
    return samples


def dist_torus_embedded(X, torus_dim=2, ambient_dim=8):
    """Empirical L1 distance of a batch X to the embedded torus."""
    d_sq = torch.zeros(X.shape[0], device=X.device)
    for i in range(torus_dim):
        r_i = torch.sqrt(X[:, 2 * i] ** 2 + X[:, 2 * i + 1] ** 2)
        d_sq += (r_i - 1) ** 2
    if ambient_dim > 2 * torus_dim:
        d_sq += (X[:, 2 * torus_dim:] ** 2).sum(dim=1)
    return torch.sqrt(d_sq).mean().item()


def proj_torus_embedded(X, torus_dim=2, ambient_dim=8):
    """Project each point in X onto the flat torus.

    Each (x_{2i}, x_{2i+1}) pair is normalized to the unit circle.
    Extra coordinates beyond 2*torus_dim are set to zero.
    """
    PX = torch.zeros_like(X)
    for i in range(torus_dim):
        r_i = torch.sqrt(X[:, 2 * i] ** 2 + X[:, 2 * i + 1]
                         ** 2).clamp(min=1e-12)
        PX[:, 2 * i] = X[:, 2 * i] / r_i
        PX[:, 2 * i + 1] = X[:, 2 * i + 1] / r_i
    return PX


def angles_torus_embedded(X, torus_dim=2):
    """Recover angles (α₁, ..., α_d) from points on or near the torus.

    Returns:
        alpha: tensor of shape (N, torus_dim) with angles in [0, 2π)
    """
    alpha = torch.stack([
        torch.atan2(X[:, 2 * i + 1], X[:, 2 * i]) for i in range(torus_dim)
    ], dim=1)
    return alpha % (2 * torch.pi)
