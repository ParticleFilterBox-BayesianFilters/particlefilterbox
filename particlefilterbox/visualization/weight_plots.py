"""Weight diagnostic plots for particlefilterbox.

Functions for visualizing ESS evolution, weight distributions,
entropy, and degeneracy diagnostics.

All functions accept an optional `ax` parameter for subplot composition
and return `(fig, ax)` tuples.
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
    """Get or create figure and axes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax


def plot_ess_timeline(
    results: Any,
    threshold: float | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot Effective Sample Size over time.

    Shows how particle diversity evolves, with optional threshold line
    indicating when resampling is triggered.

    Parameters
    ----------
    results : FilterResults
        Particle filter results. Expects `results.ess` as array-like of shape (T,)
        or `results.weights` of shape (T, N).
    threshold : float or None
        ESS threshold to display as horizontal dashed line.
        If None, uses N/2 where N is number of particles.
    ax : Axes or None
        Matplotlib axes. If None, creates new figure.
    **kwargs
        Additional keyword arguments passed to `ax.plot()`.

    Returns
    -------
    tuple of (Figure, Axes)
        The figure and axes objects.

    Examples
    --------
    >>> fig, ax = plot_ess_timeline(results, threshold=500)
    >>> fig.savefig('ess.png')
    """
    fig, ax = _get_fig_ax(ax)
    colors = get_colors()

    ess = getattr(results, "ess", None)
    if ess is None:
        weights = np.asarray(results.weights)
        if weights.ndim == 2:
            n_times, n_particles = weights.shape
            ess_list: list[float] = []
            for t in range(n_times):
                w = weights[t]
                w_sum = w.sum()
                if w_sum > 0:
                    w_norm = w / w_sum
                    ess_list.append(1.0 / np.sum(w_norm**2))
                else:
                    ess_list.append(0.0)
            ess = np.array(ess_list)
        else:
            ess = np.array([1.0 / np.sum(weights**2)])
            n_particles = len(weights)
    else:
        ess = np.asarray(ess)
        n_particles = getattr(results, "n_particles", len(ess))

    if threshold is None:
        threshold = n_particles / 2.0

    plot_kwargs: dict[str, Any] = {
        "color": colors[0],
        "linewidth": 1.5,
        "label": "ESS",
    }
    plot_kwargs.update(kwargs)

    time = np.arange(len(ess))
    ax.plot(time, ess, **plot_kwargs)
    ax.axhline(
        y=threshold,
        color=colors[2],
        linestyle="--",
        alpha=0.7,
        label=f"Threshold ({threshold:.0f})",
    )
    ax.axhline(y=n_particles, color=colors[1], linestyle=":", alpha=0.5, label=f"N={n_particles}")

    ax.set_xlabel("Time")
    ax.set_ylabel("ESS")
    ax.set_title("Effective Sample Size")
    ax.legend(loc="best")
    ax.set_ylim(bottom=0)

    return fig, ax


def plot_weight_histogram(
    results: Any,
    t: int = -1,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot histogram of particle weights at time t.

    Visualizes the distribution of normalized importance weights,
    revealing degeneracy (most weight on few particles) or
    uniformity (healthy particle set).

    Parameters
    ----------
    results : FilterResults
        Particle filter results. Expects `results.weights` shape (T, N) or (N,).
    t : int
        Time index to plot. Default is -1 (last time step).
    ax : Axes or None
        Matplotlib axes. If None, creates new figure.
    **kwargs
        Additional keyword arguments passed to `ax.hist()`.

    Returns
    -------
    tuple of (Figure, Axes)
        The figure and axes objects.
    """
    fig, ax = _get_fig_ax(ax)
    colors = get_colors()

    weights = np.asarray(results.weights)
    w = weights[t] if weights.ndim == 2 else weights
    w = w / w.sum()

    hist_kwargs: dict[str, Any] = {
        "bins": 50,
        "color": colors[0],
        "alpha": 0.7,
        "edgecolor": "white",
        "linewidth": 0.5,
    }
    hist_kwargs.update(kwargs)

    ax.hist(w, **hist_kwargs)
    ax.axvline(
        x=1.0 / len(w),
        color=colors[2],
        linestyle="--",
        alpha=0.7,
        label=f"Uniform (1/N={1.0 / len(w):.4f})",
    )

    ax.set_xlabel("Weight")
    ax.set_ylabel("Count")
    ax.set_title(f"Weight Distribution at t={t}")
    ax.legend(loc="best")

    return fig, ax


def plot_weight_entropy(
    results: Any,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot entropy of particle weights over time.

    Weight entropy measures the uniformity of the weight distribution.
    Maximum entropy = log(N) corresponds to uniform weights.
    Low entropy indicates weight degeneracy.

    Parameters
    ----------
    results : FilterResults
        Particle filter results. Expects `results.weights` shape (T, N).
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

    weights = np.asarray(results.weights)
    if weights.ndim == 1:
        weights = weights[np.newaxis, :]

    n_times, n_particles = weights.shape
    max_entropy = np.log(n_particles)

    entropies = np.zeros(n_times)
    for t in range(n_times):
        w = weights[t]
        w = w / w.sum()
        w_pos = w[w > 0]
        entropies[t] = -np.sum(w_pos * np.log(w_pos))

    plot_kwargs: dict[str, Any] = {
        "color": colors[0],
        "linewidth": 1.5,
        "label": "Weight entropy",
    }
    plot_kwargs.update(kwargs)

    time = np.arange(n_times)
    ax.plot(time, entropies, **plot_kwargs)
    ax.axhline(
        y=max_entropy,
        color=colors[1],
        linestyle="--",
        alpha=0.7,
        label=f"Max entropy (log N = {max_entropy:.2f})",
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Entropy")
    ax.set_title("Weight Entropy Over Time")
    ax.legend(loc="best")
    ax.set_ylim(bottom=0)

    return fig, ax


def plot_weight_max(
    results: Any,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot maximum particle weight over time.

    Tracks the maximum normalized weight at each time step.
    High max weight indicates particle degeneracy.

    Parameters
    ----------
    results : FilterResults
        Particle filter results. Expects `results.weights` shape (T, N).
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

    weights = np.asarray(results.weights)
    if weights.ndim == 1:
        weights = weights[np.newaxis, :]

    n_times, n_particles = weights.shape

    max_weights = np.zeros(n_times)
    for t in range(n_times):
        w = weights[t]
        w = w / w.sum()
        max_weights[t] = np.max(w)

    plot_kwargs: dict[str, Any] = {
        "color": colors[2],
        "linewidth": 1.5,
        "label": "Max weight",
    }
    plot_kwargs.update(kwargs)

    time = np.arange(n_times)
    ax.plot(time, max_weights, **plot_kwargs)
    ax.axhline(
        y=1.0 / n_particles,
        color=colors[0],
        linestyle="--",
        alpha=0.7,
        label=f"Uniform (1/N={1.0 / n_particles:.4f})",
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Max Weight")
    ax.set_title("Maximum Particle Weight Over Time")
    ax.legend(loc="best")
    ax.set_ylim(0, 1)

    return fig, ax
