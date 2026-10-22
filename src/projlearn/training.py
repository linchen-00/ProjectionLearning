"""Training loops for projection models and diffusion models."""

import copy
import time
import torch
import os
from dataclasses import dataclass, field
from tqdm import trange
from torch.utils.data import DataLoader

from .losses import projection_loss, score_matching_loss_multi_t
from .sampling import inject_randomness


def _make_scheduler(optimizer, n_epochs, scheduler):
    """Create an LR scheduler from a string name.

    Args:
        optimizer: torch optimizer
        n_epochs: total training epochs
        scheduler: "constant", "cosine", "step", or "exponential"

    Returns:
        lr_scheduler or None
    """
    if scheduler == "constant":
        return None
    elif scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    elif scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    elif scheduler == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler!r}. "
                         f"Choose from 'constant', 'cosine', 'step', 'exponential'.")


def _refined_anchors(pre_model, batch, prev_delta_, prev_copies, n_pre_legacy,
                     device):
    """Return refined anchors produced by pushing prev-delta noisy points
    through ``pre_model``.

    Two modes are supported:

    1. **Per-sample (paper / MNIST-multiseed style)** — used when
       ``prev_copies`` is not None.  Generates exactly ``prev_copies``
       noisy copies of every sample at radius ``prev_delta_`` and pushes
       them through ``pre_model``.
    2. **Legacy absolute-count** — used when ``prev_copies`` is None and
       ``n_pre_legacy > 0``.  Generates approximately ``n_pre_legacy``
       points total (``n_pre_legacy // len(batch) + 1`` copies per
       sample, then truncated to exactly ``n_pre_legacy``).  Kept for
       backward compatibility with experiments 04 and 05.

    Returns ``None`` when no refined anchors should be added.
    """
    if pre_model is None:
        return None
    if (prev_copies is None) and (n_pre_legacy is None or n_pre_legacy <= 0):
        return None
    pre_model_dev = pre_model.to(device)
    pre_model_dev.eval()
    with torch.no_grad():
        if prev_copies is not None:
            if prev_copies <= 0:
                return None
            pts = inject_randomness(batch, prev_delta_, prev_copies, device)
        else:
            repeat_num = n_pre_legacy // len(batch) + 1
            pts = inject_randomness(
                batch, prev_delta_, repeat_num, device)[:n_pre_legacy]
        return pre_model_dev(pts)


def train_projection(model, samples, delta, n_epochs=400, num_noisy=50,
                     batch_size=64, loss_weights=None, r_inner=0.4,
                     lr=5e-3, print_every=100,
                     use_dataloader=False, show_progress=True, device=None,
                     verbose=True, scheduler="constant",
                     pre_model=None, n_pre=0,
                     prev_delta=None,
                     prev_copies=None,
                     c2_mode="exact", c2_subsample=0.2,
                     c2_fd_eps=1e-3,
                     save_dir=None, save_every_epochs=50, save_prefix="model",
                     epoch_callback=None, metric_every=0,
                     loss_chunk_size=None):
    """Train a projection model.

    Args:
        model: nn.Module projection network (already on device)
        samples: (N, dim) training points
        delta: tubular-neighborhood radius
        n_epochs: number of training iterations
        num_noisy: noisy points per sample for the smoothed distribution at
            the current ``delta``.  Algorithm 2's stage k≥2 typically uses
            ``num_noisy=2`` (current_noisy_copies) here.
        batch_size: batch size when using dataloader mode
        loss_weights: dict like {"C1": 1, "C2": 0.5, "C3": 0.5}; defaults to {"C1": 1, "C3": 1}
        r_inner: inner radius for C3 term
        c2_mode: "exact", "subsample", or "fd". C2 blends the mean squared
            JVP norms A at T(x) and B at x. Each phase starts with s=0.5;
            after each epoch, s=B/(A+B) using detached, sample-weighted
            epoch means for the next epoch. If A=B=0, s stays at 0.5.
        lr: learning rate
        print_every: update tqdm description every this many iterations
        use_dataloader: if True, batch via DataLoader (for large datasets)
        show_progress: show tqdm progress bar
        device: device (inferred from model if None)
        verbose: print final loss summary
        scheduler: LR schedule — "constant", "cosine", or "step"
        pre_model: optional previously-trained projection model.  When set,
            its push-forward of ``prev_delta``-noisy points is concatenated
            with the current ``delta``-noisy points before computing the loss.
        n_pre: legacy absolute-count refined-anchor budget.  Used only when
            ``prev_copies`` is None.  Total of approximately ``n_pre`` points
            (per outer iteration in non-dataloader mode, per minibatch in
            dataloader mode) are drawn at ``prev_delta`` and pushed through
            ``pre_model``.
        prev_delta: noise radius used when generating points for ``pre_model``;
            falls back to ``delta`` when None.
        prev_copies: per-sample count of refined anchors at ``prev_delta``
            (the paper's Algorithm 2 / MNIST-multiseed convention).  When
            provided, supersedes ``n_pre``: every sample gets exactly
            ``prev_copies`` noisy copies at ``prev_delta`` pushed through
            ``pre_model`` and concatenated with ``num_noisy`` current-delta
            copies.
        save_dir: directory to save per-epoch checkpoints; None disables saving
        save_every_epochs: save a checkpoint every this many epochs
        save_prefix: filename prefix for saved checkpoints
        epoch_callback: optional callable invoked every ``metric_every``
            epochs (and once after the final epoch).  Signature::

                epoch_callback(epoch, model, info)

            where ``epoch`` is the 1-based epoch index, ``model`` is the
            currently-training network (still in ``train()`` mode), and
            ``info`` is a dict with keys ``loss`` (last batch loss),
            ``lr`` (current learning rate), ``delta`` and ``r_inner``.
            The callback runs inside the training loop — it should
            ``model.eval()`` / ``no_grad`` itself if needed and is
            responsible for restoring training mode.
        metric_every: frequency (in epochs) at which ``epoch_callback`` is
            invoked.  ``0`` (default) disables periodic invocation but the
            callback is still called once after the final epoch when set.
        loss_chunk_size: optional maximum noisy points per loss computation.
            Accumulate sample-weighted gradients over chunks, then take one
            optimizer step per batch. This reduces peak memory without
            changing the full-batch objective. None uses the entire batch.

    Returns:
        (final_loss, loss_history) — final loss value and per-epoch loss list
    """
    if loss_weights is None:
        loss_weights = {"C1": 1, "C3": 1}
    if device is None:
        device = next(model.parameters()).device
    if loss_chunk_size is not None and (
            not isinstance(loss_chunk_size, int) or loss_chunk_size < 1):
        raise ValueError("loss_chunk_size must be a positive integer or None")

    _prev_delta = prev_delta if prev_delta is not None else delta
    # Full training-set tensor kept on device for correct argmin search in C3.
    samples_dev = samples.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = _make_scheduler(optimizer, n_epochs, scheduler)
    iterator = trange(n_epochs, disable=not show_progress)
    loss_history = []
    c2_balance = 0.5
    t0 = time.time()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    def backward_loss(batch, smoothed):
        """Backpropagate one batch, optionally accumulating noisy-point chunks."""
        def compute_loss(points):
            return projection_loss(
                model, batch, points, model(points), loss_weights, r_inner,
                delta=delta, c2_mode=c2_mode,
                c2_subsample=c2_subsample, c2_fd_eps=c2_fd_eps,
                all_samples=samples_dev, c2_balance=c2_balance,
                return_c2_terms=True,
            )

        if loss_chunk_size is None:
            loss, _, _, _, _, c2_a, c2_b = compute_loss(smoothed)
            loss.backward()
            return loss.detach(), c2_a, c2_b

        totals = smoothed.new_zeros(3)
        for points in smoothed.split(loss_chunk_size):
            loss, _, _, _, _, c2_a, c2_b = compute_loss(points)
            fraction = len(points) / len(smoothed)
            (loss * fraction).backward()
            totals += torch.stack((loss.detach(), c2_a, c2_b)) * fraction
        return totals.unbind()

    if use_dataloader:
        for it in iterator:
            c2_epoch = torch.zeros(2, device=device)
            for batch in DataLoader(samples, batch_size=batch_size, shuffle=True):
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                smoothed = inject_randomness(batch, delta, num_noisy, device)

                refined = _refined_anchors(
                    pre_model, batch, _prev_delta,
                    prev_copies, n_pre, device)
                if refined is not None:
                    smoothed = torch.cat([smoothed, refined], dim=0)

                total_loss, c2_a, c2_b = backward_loss(batch, smoothed)
                c2_epoch += torch.stack((c2_a, c2_b)) * len(smoothed)
                optimizer.step()
            c2_sum = c2_epoch.sum().item()
            c2_balance = (c2_epoch[1].item() / c2_sum) if c2_sum > 0 else 0.5
            loss_history.append(total_loss.item())
            if lr_scheduler:
                lr_scheduler.step()
            if show_progress and (it + 1) % print_every == 0:
                desc = f"loss={total_loss.item():.6f}"
                if lr_scheduler:
                    desc += f" lr={optimizer.param_groups[0]['lr']:.2e}"
                iterator.set_description(desc)
            if save_dir is not None and (it + 1) % save_every_epochs == 0:
                ckpt_path = os.path.join(
                    save_dir, f"{save_prefix}_epoch{it + 1}.pth"
                )
                torch.save(
                    {
                        "epoch": it + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": (
                            lr_scheduler.state_dict()
                            if lr_scheduler is not None else None
                        ),
                        "loss": total_loss.item(),
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint to {ckpt_path}")
            if epoch_callback is not None and metric_every and \
                    (it + 1) % metric_every == 0:
                epoch_callback(it + 1, model, {
                    "loss": total_loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "delta": delta, "r_inner": r_inner,
                })
    else:
        batch = samples_dev
        for it in iterator:
            optimizer.zero_grad(set_to_none=True)
            smoothed = inject_randomness(batch, delta, num_noisy, device)

            refined = _refined_anchors(
                pre_model, batch, _prev_delta,
                prev_copies, n_pre, device)
            if refined is not None:
                smoothed = torch.cat([smoothed, refined], dim=0)

            total_loss, c2_a, c2_b = backward_loss(batch, smoothed)
            optimizer.step()
            c2_sum = (c2_a + c2_b).item()
            c2_balance = (c2_b.item() / c2_sum) if c2_sum > 0 else 0.5
            loss_history.append(total_loss.item())
            if lr_scheduler:
                lr_scheduler.step()
            if show_progress and (it + 1) % print_every == 0:
                desc = f"loss={total_loss.item():.6f}"
                if lr_scheduler:
                    desc += f" lr={optimizer.param_groups[0]['lr']:.2e}"
                iterator.set_description(desc)
            if epoch_callback is not None and metric_every and \
                    (it + 1) % metric_every == 0:
                epoch_callback(it + 1, model, {
                    "loss": total_loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "delta": delta, "r_inner": r_inner,
                })

    elapsed = time.time() - t0
    final_loss = total_loss.item()
    if epoch_callback is not None and (
            metric_every == 0 or n_epochs % metric_every != 0):
        epoch_callback(n_epochs, model, {
            "loss": final_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "delta": delta, "r_inner": r_inner,
            "final": True,
        })
    if verbose:
        print(f"  Final loss: {final_loss:.6f}  |  Time: {elapsed:.1f}s")
    return final_loss, loss_history

def train_diffusion(model, samples, marginal_prob_std_fn, n_epochs=2000,
                    M=250, lr=1e-4, show_progress=True, device=None,
                    use_dataloader=False, batch_size=512,
                    epoch_callback=None, metric_every=0,
                    optimizer_reset_every=None):
    """Train a score-based diffusion model.

    Args:
        model: ScoreNet or ScoreNetVector (already on device)
        samples: training tensor — (N, D) for vectors or (N, 1, H, W) for images
        marginal_prob_std_fn: callable t -> std
        n_epochs: training epochs
        M: number of time samples per data point
        lr: learning rate
        show_progress: show tqdm progress bar
        device: device (inferred from model if None)
        use_dataloader: if True, train with shuffled minibatches
        batch_size: minibatch size when use_dataloader=True
        epoch_callback: optional callable ``(epoch, model, info)`` fired every
            ``metric_every`` epochs (1-based epoch index).  ``info`` is a dict
            containing at least ``{"loss": float}``.
        metric_every: fire ``epoch_callback`` every this many epochs (0 = off).
        optimizer_reset_every: recreate Adam after this many epochs.  This
            reproduces experiments that trained in fixed-length calls while
            retaining one continuous metric callback and epoch axis.
    """
    if device is None:
        device = next(model.parameters()).device
    if optimizer_reset_every is not None and optimizer_reset_every <= 0:
        raise ValueError("optimizer_reset_every must be a positive integer")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    iterator = trange(n_epochs, disable=not show_progress)
    t0 = time.time()

    if use_dataloader:
        ds = samples if torch.is_tensor(samples) else torch.as_tensor(samples)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        for epoch in iterator:
            if (optimizer_reset_every is not None and epoch > 0
                    and epoch % optimizer_reset_every == 0):
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            last_loss = None
            for batch in loader:
                batch = batch.to(device)
                loss = score_matching_loss_multi_t(
                    model, batch, marginal_prob_std_fn, M=M
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                last_loss = loss
            if last_loss is not None:
                iterator.set_description(f"Loss: {last_loss.item():.5f}")
            if (epoch_callback is not None and metric_every > 0
                    and (epoch + 1) % metric_every == 0):
                epoch_callback(epoch + 1, model,
                               {"loss": last_loss.item() if last_loss is not None
                                else float("nan")})
        loss = last_loss
    else:
        x = samples.to(device)
        for epoch in iterator:
            if (optimizer_reset_every is not None and epoch > 0
                    and epoch % optimizer_reset_every == 0):
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss = score_matching_loss_multi_t(
                model, x, marginal_prob_std_fn, M=M)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            iterator.set_description(f"Loss: {loss.item():.5f}")
            if (epoch_callback is not None and metric_every > 0
                    and (epoch + 1) % metric_every == 0):
                epoch_callback(epoch + 1, model, {"loss": loss.item()})

    elapsed = time.time() - t0
    print(f"  Final loss: {loss.item():.6f}  |  Time: {elapsed:.1f}s")

# =============================================================================
# Final-paper algorithms — user-facing wrappers around `train_projection`.
#
# Algorithm 1 (Two-Stage Training of the Projection Model)
# Algorithm 2 (Iterated Projection Algorithm, IPA)
#
# Algorithm 1 is the K=1 special case of Algorithm 2; both share the same
# inner one-phase trainer (`train_projection`).  The two functions below are
# kept distinct for clarity in the paper-facing API.
# =============================================================================


def train_projection_algorithm1(
    model,
    samples,
    delta,
    *,
    n_pretrain,
    n_project,
    lr_pretrain,
    lr_project,
    num_noisy,
    r_inner,
    loss_pretrain=None,
    loss_project=None,
    r_inner_pretrain=0.0,
    batch_size=64,
    use_dataloader=False,
    scheduler_pretrain="constant",
    scheduler_project="constant",
    print_every=100,
    show_progress=True,
    verbose=True,
    device=None,
    pre_model=None,
    prev_delta=None,
    prev_copies=None,
    n_pre=0,
    c2_mode="exact",
    c2_subsample=0.2,
    c2_fd_eps=1e-3,
    save_dir=None,
    save_every_epochs=None,
    save_prefix="alg1",
    epoch_callback=None,
    metric_every=0,
    loss_chunk_size=None,
):
    """Algorithm 1 — Two-Stage Training of the Projection Model.

    Trains ``model`` in place at a single tubular radius ``delta`` with two
    consecutive phases (in order):

    1. **Pretraining** (``loss_pretrain`` ≈ ``{"C1":1,"C3":1}``) for
       ``n_pretrain`` iterations at learning rate ``lr_pretrain``,
       ``r_inner=r_inner_pretrain`` (= 0 by default).
    2. **Projection-training** (``loss_project`` ≈ ``{"C1":1,"C2":1,"C3":1}``)
       for ``n_project`` iterations at learning rate ``lr_project`` with the
       paper's inner radius ``r_inner`` (= underline_delta).

    Both phases use the same training set ``samples`` and the same
    tubular radius ``delta``.

    Args:
        model: ``nn.Module`` projection network (mutated in place).
        samples: ``(N, D)`` tensor of training anchors X.
        delta: tubular-neighborhood radius for both phases.
        n_pretrain, n_project: epochs for each phase.
        lr_pretrain, lr_project: learning rate for each phase.
        num_noisy: noisy points per anchor at the current ``delta``.
        r_inner: inner radius (= underline_delta) used in stage B.
        loss_pretrain, loss_project: loss-weight dicts; defaults shown above.
        r_inner_pretrain: inner radius during stage A (default 0).
        batch_size, use_dataloader: minibatch / dataloader controls forwarded.
        scheduler_pretrain, scheduler_project: per-stage LR schedule names.
        print_every, show_progress, verbose, device: cosmetic controls.
        pre_model: previous-stage projection model used to produce refined
            anchors (Algorithm 2 stages k≥2).  Leave None for plain
            Algorithm 1.
        prev_delta: noise radius for refined-anchor generation; falls back
            to ``delta`` when None.
        prev_copies: per-sample count of refined anchors at ``prev_delta``
            (paper / MNIST-multiseed convention).  When set this overrides
            the legacy ``n_pre`` budget.
        n_pre: legacy absolute-count refined-anchor budget (used only when
            ``prev_copies`` is None).
        c2_mode, c2_subsample, c2_fd_eps: forwarded to ``projection_loss``.
        loss_chunk_size: optional noisy-point chunk size forwarded to
            ``train_projection``; gradients accumulate before each Adam step.
        save_dir, save_every_epochs, save_prefix: optional periodic-checkpoint
            controls; written via ``train_projection`` with prefixes
            ``{save_prefix}_pretrain`` / ``{save_prefix}_project``.
        epoch_callback, metric_every: forwarded to :func:`train_projection`
            *for both phases*.  The callback receives an extra key
            ``"phase"`` (``"pretrain"`` or ``"project"``) so the user can
            distinguish.

    Returns:
        dict with keys ``model``, ``pretrain_snapshot`` (deep-copy after
        stage A), ``pretrain_loss_history``, ``project_loss_history``,
        ``pretrain_final_loss``, ``project_final_loss``, ``delta``,
        ``r_inner``.
    """
    if loss_pretrain is None:
        loss_pretrain = {"C1": 1, "C3": 1}
    if loss_project is None:
        loss_project = {"C1": 1, "C2": 1, "C3": 1}

    def _wrap_cb(phase_name):
        if epoch_callback is None:
            return None

        def _cb(epoch, model_, info):
            info = dict(info)
            info["phase"] = phase_name
            epoch_callback(epoch, model_, info)
        return _cb

    common = dict(
        num_noisy=num_noisy,
        batch_size=batch_size,
        use_dataloader=use_dataloader,
        show_progress=show_progress,
        verbose=verbose,
        device=device,
        pre_model=pre_model,
        n_pre=n_pre,
        prev_delta=prev_delta,
        prev_copies=prev_copies,
        c2_mode=c2_mode,
        c2_subsample=c2_subsample,
        c2_fd_eps=c2_fd_eps,
        metric_every=metric_every,
        loss_chunk_size=loss_chunk_size,
    )

    save_pre = {}
    save_proj = {}
    if save_dir is not None and save_every_epochs is not None:
        save_pre = dict(
            save_dir=save_dir,
            save_every_epochs=save_every_epochs,
            save_prefix=f"{save_prefix}_pretrain",
        )
        save_proj = dict(
            save_dir=save_dir,
            save_every_epochs=save_every_epochs,
            save_prefix=f"{save_prefix}_project",
        )

    pre_loss, pre_history = train_projection(
        model, samples, delta,
        n_epochs=n_pretrain, lr=lr_pretrain,
        loss_weights=loss_pretrain, r_inner=r_inner_pretrain,
        scheduler=scheduler_pretrain, print_every=print_every,
        epoch_callback=_wrap_cb("pretrain"),
        **common, **save_pre,
    )
    pretrain_snapshot = copy.deepcopy(model).eval()

    proj_loss, proj_history = train_projection(
        model, samples, delta,
        n_epochs=n_project, lr=lr_project,
        loss_weights=loss_project, r_inner=r_inner,
        scheduler=scheduler_project, print_every=print_every,
        epoch_callback=_wrap_cb("project"),
        **common, **save_proj,
    )

    return {
        "model": model,
        "pretrain_snapshot": pretrain_snapshot,
        "pretrain_loss_history": pre_history,
        "project_loss_history": proj_history,
        "pretrain_final_loss": pre_loss,
        "project_final_loss": proj_loss,
        "delta": delta,
        "r_inner": r_inner,
    }


def train_projection_algorithm2(
    model,
    samples,
    deltas,
    r_inners,
    *,
    n_pretrain,
    n_project,
    lr_pretrain,
    lr_project,
    num_points_per_samples,
    current_noisy_copies=2,
    loss_pretrain=None,
    loss_project=None,
    r_inner_pretrain=0.0,
    batch_size=64,
    use_dataloader=False,
    scheduler_pretrain="constant",
    scheduler_project="constant",
    print_every=100,
    show_progress=True,
    verbose=True,
    device=None,
    c2_mode="exact",
    c2_subsample=0.2,
    c2_fd_eps=1e-3,
    save_dir=None,
    save_every_epochs=None,
    save_prefix="alg2",
    epoch_callback=None,
    metric_every=0,
):
    """Algorithm 2 — Iterated Projection Algorithm (IPA).

    Trains ``model`` in place across ``K = len(deltas)`` warm-started stages
    with strictly decreasing ``deltas[0] > deltas[1] > ... > deltas[K-1]``
    and non-increasing ``r_inners``.

    **Refined-anchor construction (paper / MNIST-multiseed style).**  At
    every iteration of stage k:

    - **Stage 1** (k=0):  Algorithm 1 at ``(deltas[0], r_inners[0])`` from
      the initial ``model`` weights, with ``num_noisy =
      num_points_per_samples`` noisy copies per sample at ``deltas[0]``
      and no refined anchors.
    - **Stage k ≥ 2**:  Algorithm 1 at ``(deltas[k], r_inners[k])``
      warm-started from the model at the end of stage k−1, with

        * ``current_noisy_copies`` (default 2) noisy copies per sample at
          the *current* ``deltas[k]``;
        * **plus** ``num_points_per_samples`` noisy copies per sample at
          ``prev_delta = deltas[k-1]``, each pushed through the previous
          stage's frozen model (= refined anchors).

      The two sets are concatenated before computing the projection loss.
      No composition of stage models is done during training.

    Args:
        model: ``nn.Module`` projection network (mutated in place).
        samples: ``(N, D)`` tensor of original training anchors X.
        deltas: length-K sequence of strictly decreasing tubular radii.
        r_inners: length-K sequence of non-increasing inner radii.
        n_pretrain, n_project: epochs per stage; scalar or length-K list.
        lr_pretrain, lr_project: per-stage learning rates; scalar or list.
        num_points_per_samples: per-sample copies at ``prev_delta`` for
            refined anchors *and* the per-sample current-delta copies of
            stage 1.
        current_noisy_copies: per-sample current-delta copies for stage k≥2
            (default ``2``, matching the MNIST-multiseed implementation).
        loss_pretrain, loss_project: per-phase loss-weight dicts (defaults
            ``{"C1":1,"C3":1}`` and ``{"C1":1,"C2":1,"C3":1}``).
        r_inner_pretrain: inner radius for the pretrain phase of every
            stage (default 0).
        batch_size, use_dataloader, scheduler_*, print_every, show_progress,
        verbose, device, c2_*: forwarded to Algorithm 1 inside each stage.
        save_dir, save_every_epochs, save_prefix: periodic-checkpoint
            controls; per-stage prefixes are ``{save_prefix}_stage{k}``.
        epoch_callback: optional callable
            ``cb(epoch, model, info)``.  ``info`` carries the keys
            documented for :func:`train_projection`, plus ``"phase"``
            (``"pretrain"`` / ``"project"``) and ``"stage"`` (0-based).
        metric_every: epoch-frequency for ``epoch_callback``.

    Returns:
        dict with keys ``model`` (final, after stage K),
        ``stage_snapshots`` (length-K list of deep-copied eval models, one
        per completed stage), ``pretrain_snapshots`` (length-K list,
        post-pretraining of each stage), ``histories`` (length-K list of
        per-stage info dicts), ``deltas``, ``r_inners``.

    Notes:
        - Algorithm 1 is exactly this function with ``K=1``; for clarity
          callers should use :func:`train_projection_algorithm1` in that case.
        - Composition (T_K ∘ ... ∘ T_1) is intentionally a generation-time
          concern (see :func:`projlearn.mnist_eval.compose`).
    """
    K = len(deltas)
    if len(r_inners) != K:
        raise ValueError(
            f"`deltas` and `r_inners` must have equal length; got {K} vs "
            f"{len(r_inners)}.")
    for k in range(1, K):
        if deltas[k] >= deltas[k - 1]:
            raise ValueError(
                "Algorithm 2 requires strictly decreasing `deltas`; "
                f"got deltas[{k - 1}]={deltas[k - 1]}, deltas[{k}]={deltas[k]}.")
        if r_inners[k] > r_inners[k - 1]:
            raise ValueError(
                "Algorithm 2 requires non-increasing `r_inners`; "
                f"got r_inners[{k - 1}]={r_inners[k - 1]}, "
                f"r_inners[{k}]={r_inners[k]}.")

    def _per_stage(x, name):
        if isinstance(x, (list, tuple)):
            if len(x) != K:
                raise ValueError(
                    f"`{name}` length {len(x)} does not match K={K}")
            return list(x)
        return [x] * K

    n_pre_list = _per_stage(n_pretrain, "n_pretrain")
    n_proj_list = _per_stage(n_project, "n_project")
    lr_pre_list = _per_stage(lr_pretrain, "lr_pretrain")
    lr_proj_list = _per_stage(lr_project, "lr_project")

    stage_snapshots = []
    pretrain_snapshots = []
    histories = []

    for k in range(K):
        prev_model = stage_snapshots[-1] if stage_snapshots else None
        prev_delta = deltas[k - 1] if k > 0 else None

        if prev_model is None:
            # Stage 1: plain Algorithm 1 at deltas[0], no refined anchors.
            num_noisy_k = num_points_per_samples
            prev_copies_k = None
        else:
            # Stage k≥2: current-delta noise (current_noisy_copies copies)
            # PLUS prev-delta refined anchors (num_points_per_samples copies
            # per sample, pushed through prev_model).  Concatenation is
            # handled inside `train_projection` via `prev_copies`.
            num_noisy_k = current_noisy_copies
            prev_copies_k = num_points_per_samples

        def _stage_cb(epoch, model_, info, _stage=k):
            if epoch_callback is None:
                return
            info = dict(info)
            info["stage"] = _stage
            epoch_callback(epoch, model_, info)

        out = train_projection_algorithm1(
            model, samples, deltas[k],
            n_pretrain=n_pre_list[k], n_project=n_proj_list[k],
            lr_pretrain=lr_pre_list[k], lr_project=lr_proj_list[k],
            num_noisy=num_noisy_k, r_inner=r_inners[k],
            loss_pretrain=loss_pretrain, loss_project=loss_project,
            r_inner_pretrain=r_inner_pretrain,
            batch_size=batch_size, use_dataloader=use_dataloader,
            scheduler_pretrain=scheduler_pretrain,
            scheduler_project=scheduler_project,
            print_every=print_every, show_progress=show_progress,
            verbose=verbose, device=device,
            pre_model=prev_model, prev_delta=prev_delta,
            prev_copies=prev_copies_k,
            c2_mode=c2_mode, c2_subsample=c2_subsample, c2_fd_eps=c2_fd_eps,
            save_dir=save_dir, save_every_epochs=save_every_epochs,
            save_prefix=f"{save_prefix}_stage{k}",
            epoch_callback=(_stage_cb if epoch_callback is not None else None),
            metric_every=metric_every,
        )

        stage_snapshots.append(copy.deepcopy(model).eval())
        pretrain_snapshots.append(out["pretrain_snapshot"])
        histories.append({
            "stage": k,
            "delta": deltas[k],
            "r_inner": r_inners[k],
            "num_noisy_current": num_noisy_k,
            "prev_copies": prev_copies_k,
            "pretrain_final_loss": out["pretrain_final_loss"],
            "project_final_loss": out["project_final_loss"],
            "pretrain_loss_history": out["pretrain_loss_history"],
            "project_loss_history": out["project_loss_history"],
        })

    return {
        "model": model,
        "stage_snapshots": stage_snapshots,
        "pretrain_snapshots": pretrain_snapshots,
        "histories": histories,
        "deltas": list(deltas),
        "r_inners": list(r_inners),
    }
