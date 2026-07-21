"""Visualization helpers for projection learning experiments."""

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .utils import sample_delta_ball


def plot_2d(manifold, samples, delta, ambient_dim=2, tubular_nbh=False,
            title=None, save_path=None):
    """Plot 2D generated curve with training samples."""
    manifold = manifold.detach().cpu() if isinstance(
        manifold, torch.Tensor) else manifold
    samples = samples.detach().cpu() if isinstance(
        samples, torch.Tensor) else samples

    plt.figure(figsize=(6, 6))
    plt.scatter(manifold[:, 0], manifold[:, 1],
                label="Generated Samples", s=5, color="green")
    plt.scatter(samples[:, 0], samples[:, 1],
                label="Training Samples", s=30, marker="^", color="red")

    if tubular_nbh:
        delta_ball = (
            samples.unsqueeze(0)
            + sample_delta_ball(200, delta, ambient_dim).unsqueeze(1)
        ).flatten(0, 1)
        plt.scatter(delta_ball[:, 0], delta_ball[:, 1],
                    label=r"Tubular $\mathcal{M}_\delta$", s=0.1, color="gray")

    plt.legend(loc="center", fontsize=14)
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.20, linewidth=0.6)
    ax.tick_params(axis="both", which="both", bottom=False, left=False,
                   top=False, right=False, labelbottom=False, labelleft=False)
    if title is not None:
        plt.title(title)
    if save_path is not None:
        plt.gcf().subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.88)
        plt.savefig(save_path, dpi=600)
    plt.show()


def plot_3d(manifold, samples, view=(20, 120),
            view_coord=(0, 1, 2), title=None, show=True,
            show_legend=True, show_axis_lines=True,
            figsize=(14, 10)):
    """Plot 3D projection of high-dimensional manifold."""
    manifold = manifold.detach().cpu() if isinstance(
        manifold, torch.Tensor) else manifold
    samples = samples.detach().cpu() if isinstance(
        samples, torch.Tensor) else samples

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")

    ax.scatter3D(samples[:, view_coord[0]], samples[:, view_coord[1]],
                 samples[:, view_coord[2]],
                 color="red", s=20, label="Training Samples")
    ax.scatter3D(manifold[:, view_coord[0]], manifold[:, view_coord[1]],
                 manifold[:, view_coord[2]],
                 color="#1b6fab", s=2, alpha=0.5, label="Generated Samples")

    ax.set_xlabel(f"$x_{{{view_coord[0]+1}}}$")
    ax.set_ylabel(f"$x_{{{view_coord[1]+1}}}$")
    ax.set_zlabel(f"$x_{{{view_coord[2]+1}}}$")
    ax.view_init(*view)
    if show_legend:
        ax.legend(fontsize=20, loc="upper left", bbox_to_anchor=(0.7, 0.95))

    if not show_axis_lines:
        ax.set_axis_off()

    ax.xaxis.pane.set_facecolor("white")
    ax.yaxis.pane.set_facecolor("white")
    ax.zaxis.pane.set_facecolor("white")
    plt.tight_layout()
    if title is not None:
        ax.set_title(title)
    if show:
        plt.show()
    return fig


# ========================= 3D scatter with projection lines =========================


def plot_3d_with_lines(manifold, M_delta, samples, delta, ambient_dim,
                       nparticles=100, view=(20, 120), view_coord=(0, 1, 2),
                       compress=False, title=None, show=True):
    """3D scatter with lines connecting M_delta points to their mapped outputs."""
    manifold = manifold.detach().cpu() if isinstance(
        manifold, torch.Tensor) else manifold
    M_delta = M_delta.detach().cpu() if isinstance(
        M_delta, torch.Tensor) else M_delta
    samples = samples.detach().cpu() if isinstance(
        samples, torch.Tensor) else samples

    n = min(nparticles, manifold.shape[0])
    idx = torch.randperm(manifold.shape[0])[:n]
    start = M_delta[idx]
    end = manifold[idx]

    def coords(X):
        if compress:
            return X[:, 0], X[:, 1], torch.norm(X[:, 2:], dim=-1)
        return X[:, view_coord[0]], X[:, view_coord[1]], X[:, view_coord[2]]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")

    sx, sy, sz = coords(start)
    ex, ey, ez = coords(end)
    for i in range(n):
        ax.plot([sx[i], ex[i]], [sy[i], ey[i]], [sz[i], ez[i]],
                color="gray", linestyle="--", linewidth=0.4, alpha=0.5)

    ax.scatter3D(*coords(start), color="orange", s=3,
                 alpha=0.5, label=r"$M_\delta$ points")
    ax.scatter3D(*coords(end), color="#1b6fab", s=3,
                 alpha=0.6, label=r"Mapped $T(x)$")
    ax.scatter3D(*coords(samples), color="red", s=11, label="Training Samples")

    ax.set_xlabel(f"$x_{{{view_coord[0]+1}}}$")
    ax.set_ylabel(f"$x_{{{view_coord[1]+1}}}$")
    ax.set_zlabel(
        r"$\|(x_3, \ldots, x_n)\|$" if compress else f"$x_{{{view_coord[2]+1}}}$")
    ax.view_init(*view)
    ax.legend(fontsize=11)
    ax.xaxis.pane.set_facecolor("white")
    ax.yaxis.pane.set_facecolor("white")
    ax.zaxis.pane.set_facecolor("white")
    plt.tight_layout()
    if title is not None:
        ax.set_title(title)
    if show:
        plt.show()
    return fig
