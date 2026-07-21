"""Sampling utilities: SDE setup, noise injection, tubular neighborhoods, and diffusion samplers."""

import torch
import numpy as np
import math
import functools

from .utils import sample_delta_ball


# ========================= SDE setup =========================


def marginal_prob_std(t, sigma):
    """Standard deviation of the perturbation kernel p_{0t}(x(t)|x(0))."""
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=torch.float32)
    return torch.sqrt((sigma ** (2 * t) - 1.0) / 2.0 / np.log(sigma))


def diffusion_coeff(t, sigma):
    """Diffusion coefficient g(t) = sigma^t."""
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=torch.float32)
    return sigma ** t


def make_sde_fns(sigma=25):
    """Create (marginal_prob_std_fn, diffusion_coeff_fn) with sigma bound."""
    return (
        functools.partial(marginal_prob_std, sigma=sigma),
        functools.partial(diffusion_coeff, sigma=sigma),
    )


# ========================= Noise injection =========================


def inject_randomness(samples, delta, num_points_per_sample, device=None):
    """For each sample, generate independent perturbations from the delta-ball.
    Returns all noisy samples flattened into one batch.

    Args:
        samples: tensor of shape (B, D) or (B, *spatial_dims)
        delta: radius of the perturbation ball
        num_points_per_sample: number of noisy copies per sample
        device: optional device; defaults to samples.device

    Returns:
        noisy_samples: shape (B * num_points_per_sample, ...) matching samples layout
    """
    if device is None:
        device = samples.device

    batch_size = samples.shape[0]
    ambient_dim = samples.shape[1] if samples.dim() == 2 else samples.shape[1:]

    if isinstance(ambient_dim, int):
        random_delta_vector = sample_delta_ball(
            N=batch_size * num_points_per_sample,
            delta=delta,
            ambient_dim=ambient_dim,
        ).view(batch_size, num_points_per_sample, ambient_dim)
    else:
        flat_dim = math.prod(ambient_dim)
        random_delta_vector = sample_delta_ball(
            N=batch_size * num_points_per_sample,
            delta=delta,
            ambient_dim=flat_dim,
        ).view(batch_size, num_points_per_sample, *ambient_dim)

    random_delta_vector = random_delta_vector.to(device)
    noisy_samples = samples[:, None] + random_delta_vector
    noisy_samples = noisy_samples.flatten(end_dim=1)

    return noisy_samples


def tubular_nbh(samples, delta, ambient_dim, num_chosen_samples=100,
                num_points_per_sample=100, device="cpu", seed=None):
    """Generate the tubular neighborhood M_delta around (a subset of) samples.

    Args:
        seed: if not None, makes sampling deterministic and restores RNG state afterward.
    """
    if seed is not None:
        cpu_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            cuda_states = torch.cuda.get_rng_state_all()
        torch.manual_seed(seed)

    try:
        if num_chosen_samples is None:
            num_chosen_samples = len(samples)
        idx = torch.randperm(len(samples))[:num_chosen_samples]
        samples_ext = samples[idx].to(device)

        if isinstance(ambient_dim, int):
            flat_dim = ambient_dim
        else:
            flat_dim = math.prod(ambient_dim)

        random_vecs = torch.stack([
            sample_delta_ball(num_points_per_sample, delta, flat_dim)
            for _ in range(len(samples_ext))
        ]).to(device)

        M_delta = (samples_ext.unsqueeze(1) + random_vecs).flatten(0, 1)
    finally:
        if seed is not None:
            torch.random.set_rng_state(cpu_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_states)

    return M_delta

# ========================= Diffusion sampler (vectors) =========================


@torch.no_grad()
def vector_sampler_with_traj(score_model, ambient_dim, marginal_prob_std_fn,
                             diffusion_coeff_fn, batch_size=512, num_steps=2000,
                             device="cuda", eps=1e-6, traj_k=20,
                             record_steps=200, ode_last_frac=0.15):
    """SDE sampler for vector data, recording particle trajectories.

    Returns:
        (samples, trajectories) where samples is (batch_size, ambient_dim)
        and trajectories is (record_steps, traj_k, ambient_dim).
    """
    score_model.eval()
    t1 = torch.ones(batch_size, device=device)
    x = torch.randn(batch_size, ambient_dim, device=device) * \
        marginal_prob_std_fn(t1)[:, None]

    time_steps = torch.linspace(1.0, eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    rec_idcs = set(torch.linspace(
        0, num_steps - 1, record_steps).long().tolist())

    traj_k = min(traj_k, batch_size)
    track_idx = torch.randperm(batch_size, device=device)[:traj_k]
    traj = []

    for i, ts in enumerate(time_steps):
        batch_t = torch.ones(batch_size, device=device) * ts
        g = diffusion_coeff_fn(batch_t)
        drift = (g ** 2)[:, None] * score_model(x, batch_t)

        if i >= int((1.0 - ode_last_frac) * num_steps):
            x = x + 0.5 * drift * step_size
        else:
            x = x + drift * step_size + \
                torch.sqrt(step_size) * g[:, None] * torch.randn_like(x)

        if i in rec_idcs:
            traj.append(x[track_idx].cpu())

    return x.cpu(), torch.stack(traj, dim=0)


# ========================= Diffusion samplers (images) =========================


@torch.no_grad()
def pc_sampler(score_model, marginal_prob_std_fn, diffusion_coeff_fn,
               batch_size=64, num_steps=500, snr=0.16, device="cuda",
               eps=1e-3, image_size=28, channels=1, show_progress=False):
    """Predictor-Corrector sampler for image score-based diffusion models.

    Mirrors the ``pc_sampler`` from ``ProjectionLearning.ipynb``.  Returns a
    tensor of shape ``(batch_size, channels, image_size, image_size)``.
    """
    import numpy as _np
    score_model.eval()
    t = torch.ones(batch_size, device=device)
    x = (torch.randn(batch_size, channels, image_size, image_size,
                     device=device)
         * marginal_prob_std_fn(t)[:, None, None, None])
    time_steps = _np.linspace(1.0, eps, num_steps)
    step_size = time_steps[0] - time_steps[1]

    iterator = time_steps
    if show_progress:
        from tqdm import tqdm as _tqdm
        iterator = _tqdm(time_steps, desc="PC sampler", leave=False)

    x_mean = x
    for time_step in iterator:
        batch_time_step = torch.ones(batch_size, device=device) * float(time_step)
        # Corrector: Langevin MCMC step
        grad = score_model(x, batch_time_step)
        grad_norm = torch.norm(
            grad.reshape(grad.shape[0], -1), dim=-1).mean()
        noise_norm = float(_np.sqrt(_np.prod(x.shape[1:])))
        langevin_step_size = 2 * (snr * noise_norm / (grad_norm + 1e-12)) ** 2
        x = (x + langevin_step_size * grad
             + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x))
        # Predictor: Euler-Maruyama step
        g = diffusion_coeff_fn(batch_time_step)
        x_mean = (x + (g ** 2)[:, None, None, None]
                  * score_model(x, batch_time_step) * step_size)
        x = (x_mean
             + torch.sqrt(g ** 2 * step_size)[:, None, None, None]
             * torch.randn_like(x))
    return x_mean
