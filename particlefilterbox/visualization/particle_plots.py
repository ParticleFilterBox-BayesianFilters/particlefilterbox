"""Particle visualization plots for particlefilterbox.

Functions for visualizing particle distributions, trajectories,
and ancestral lineages from particle filter results.

All functions accept an optional `ax` parameter for subplot composition
and return `(fig, ax)` tuples.

References
----------
Doucet, A. & Johansen, A.M. (2011). A tutorial on particle filtering
and smoothing: Fifteen years later.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from particlefilterbox.visualization.themes import get_colors


def _get_fig_ax(
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Get or create figure and axes.

    Parameters
    ----------
    ax : Axes or None
        Existing axes. If None, a new figure is created.
    figsize : tuple or None
        Figure size if creating new figure.

    Returns
    -------
    tuple of (Figure, Axes)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax


def plot_particle_cloud(
    results: Any,
    t: int = -1,
    state_idx: tuple[int, int] = (0, 1),
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot 2D scatter of particle positions at time t.

    Shows the particle cloud colored by weight, giving a visual
    representation of the filtering distribution at a specific time step.

    Parameters
    ----------
    results : FilterResults
        Particle filter results containing `particles` and `weights`.
        Expects `results.particles` shape (T, N, d) and `results.weights` shape (T, N).
    t : int
        Time index to plot. Default is -1 (last time step).
    state_idx : tuple of (int, int)
        Indices of the two state dimensions to plot. Default is (0, 1).
    ax : Axes or None
        Matplotlib axes. If None, creates new figure.
    **kwargs
        Additional keyword arguments passed to `ax.scatter()`.

    Returns
    -------
    tuple of (Figure, Axes)
        The figure and axes objects.

    Examples
    --------
    >>> fig, ax = plot_particle_cloud(results, t=50)
    >>> fig.savefig('cloud.png')
    """
    fig, ax = _get_fig_ax(ax)

    particles = np.asarray(results.particles)
    weights = np.asarray(results.weights)

    if particles.ndim == 3:
        pts = particles[t]
        w = weights[t]
    else:
        pts = particles
        w = weights

    i, j = state_idx

    scatter_kwargs: dict[str, Any] = {
        "c": w,
        "cmap": "viridis",
        "alpha": 0.6,
        "s": 10,
        "edgecolors": "none",
    }
    scatter_kwargs.update(kwargs)

    sc = ax.scatter(pts[:, i], pts[:, j], **scatter_kwargs)
    fig.colorbar(sc, ax=ax, label="Weight")
    ax.set_xlabel(f"State {i}")
    ax.set_ylabel(f"State {j}")
    ax.set_title(f"Particle Cloud at t={t}")

    return fig, ax


def plot_particle_trajectories(
    results: Any,
    n: int = 50,
    state_idx: int = 0,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot trajectories of n randomly selected particles over time.

    Shows how individual particles evolve through the state space,
    giving insight into particle diversity and degeneracy.

    Parameters
    ----------
    results : FilterResults
        Particle filter results. Expects `results.particles` shape (T, N, d).
    n : int
        Number of trajectories to plot. Default is 50.
    state_idx : int
        State dimension index to plot. Default is 0.
    ax : Axes or None
        Matplotlib axes. If None, creates new figure.
    **kwargs
        Additional keyword arguments passed to `ax.plot()`.

    Returns
    -------
    tuple of (Figure, Axes)
        The figure and axes objects.
    """
    fig, ax = _get_fig_ax(ax)
    colors = get_colors()

    particles = np.asarray(results.particles)
    n_times, n_particles, _ = particles.shape

    rng = np.random.default_rng(42)
    idx = rng.choice(n_particles, size=min(n, n_particles), replace=False)

    plot_kwargs: dict[str, Any] = {
        "alpha": 0.3,
        "linewidth": 0.5,
        "color": colors[0],
    }
    plot_kwargs.update(kwargs)

    for i in idx:
        ax.plot(range(n_times), particles[:, i, state_idx], **plot_kwargs)

    ax.set_xlabel("Time")
    ax.set_ylabel(f"State {state_idx}")
    ax.set_title(f"Particle Trajectories (n={len(idx)})")

    return fig, ax


def plot_ancestral_tree(
    results: Any,
    n: int = 20,
    state_idx: int = 0,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot ancestral lineage tree of particles.

    Traces the ancestry of particles backward through time, showing
    how resampling concentrates weight on a few lineages.

    Parameters
    ----------
    results : FilterResults
        Particle filter results. Expects `results.particles` shape (T, N, d)
        and optionally `results.ancestors` shape (T, N).
    n : int
        Number of final-time particles to trace backward. Default is 20.
    state_idx : int
        State dimension index to plot. Default is 0.
    ax : Axes or None
        Matplotlib axes. If None, creates new figure.
    **kwargs
        Additional keyword arguments passed to `ax.plot()`.

    Returns
    -------
    tuple of (Figure, Axes)
        The figure and axes objects.
    """
    fig, ax = _get_fig_ax(ax)
    colors = get_colors()

    particles = np.asarray(results.particles)
    n_times, n_particles, _ = particles.shape

    # Use ancestors if available, otherwise just plot forward trajectories
    ancestors = getattr(results, "ancestors", None)

    rng = np.random.default_rng(42)
    selected = rng.choice(n_particles, size=min(n, n_particles), replace=False)

    plot_kwargs: dict[str, Any] = {
        "alpha": 0.5,
        "linewidth": 0.8,
    }
    plot_kwargs.update(kwargs)

    for k, idx in enumerate(selected):
        color = colors[k % len(colors)]
        trajectory = np.zeros(n_times)
        current_idx = idx

        for t_rev in range(n_times - 1, -1, -1):
            trajectory[t_rev] = particles[t_rev, int(current_idx), state_idx]
            if ancestors is not None and t_rev > 0:
                ancestor_arr = np.asarray(ancestors)
                current_idx = ancestor_arr[t_rev, int(current_idx)]

        ax.plot(range(n_times), trajectory, color=color, **plot_kwargs)

    ax.set_xlabel("Time")
    ax.set_ylabel(f"State {state_idx}")
    ax.set_title(f"Ancestral Tree ({len(selected)} lineages)")

    return fig, ax


def plot_particle_evolution(
    results: Any,
    state_idx: int = 0,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot evolution of particle distribution over time.

    Shows the weighted mean and credible intervals of the particle
    distribution at each time step, providing a summary view of
    how the filtering distribution evolves.

    Parameters
    ----------
    results : FilterResults
        Particle filter results. Expects `results.particles` shape (T, N, d)
        and `results.weights` shape (T, N).
    state_idx : int
        State dimension index to plot. Default is 0.
    ax : Axes or None
        Matplotlib axes. If None, creates new figure.
    **kwargs
        Additional keyword arguments passed to `ax.fill_between()`.

    Returns
    -------
    tuple of (Figure, Axes)
        The figure and axes objects.
    """
    fig, ax = _get_fig_ax(ax)
    colors = get_colors()

    particles = np.asarray(results.particles)
    weights = np.asarray(results.weights)
    n_times, n_particles, _ = particles.shape

    means = np.zeros(n_times)
    q05 = np.zeros(n_times)
    q25 = np.zeros(n_times)
    q75 = np.zeros(n_times)
    q95 = np.zeros(n_times)

    for t in range(n_times):
        pts = particles[t, :, state_idx]
        w = weights[t]
        w = w / w.sum()

        means[t] = np.average(pts, weights=w)

        sorted_idx = np.argsort(pts)
        sorted_pts = pts[sorted_idx]
        cum_w = np.cumsum(w[sorted_idx])

        q05[t] = sorted_pts[np.searchsorted(cum_w, 0.05)]
        q25[t] = sorted_pts[np.searchsorted(cum_w, 0.25)]
        q75[t] = sorted_pts[min(np.searchsorted(cum_w, 0.75), n_particles - 1)]
        q95[t] = sorted_pts[min(np.searchsorted(cum_w, 0.95), n_particles - 1)]

    time = np.arange(n_times)

    ax.fill_between(time, q05, q95, color=colors[0], alpha=0.1, label="90% CI")
    ax.fill_between(time, q25, q75, color=colors[0], alpha=0.25, label="50% CI")
    ax.plot(time, means, color=colors[0], linewidth=1.5, label="Weighted mean")

    ax.set_xlabel("Time")
    ax.set_ylabel(f"State {state_idx}")
    ax.set_title("Particle Distribution Evolution")
    ax.legend(loc="best")

    return fig, ax
