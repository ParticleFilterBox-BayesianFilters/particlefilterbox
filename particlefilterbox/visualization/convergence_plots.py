"""Convergence diagnostic plots for particlefilterbox.

Functions for visualizing particle filter convergence rate,
log-likelihood distributions across particle counts, and
QQ-plots of weight distributions.

References
----------
Chopin, N. (2004). Central limit theorem for sequential Monte Carlo
methods and its application to Bayesian inference. Annals of Statistics.
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


def plot_convergence_rate(
    conv_results: dict[int, list[float]] | Any,
    true_value: float | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot convergence rate: error vs number of particles.

    Shows how the estimation error decreases with increasing particle count,
    typically following a 1/sqrt(N) rate for well-behaved particle filters.

    Parameters
    ----------
    conv_results : dict[int, list[float]] or similar
        Mapping from N (particle count) to list of estimates (from repeated runs).
        Can also be an object with `n_particles_list` and `estimates` attributes.
    true_value : float or None
        True value for computing error. If None, uses the mean of the
        largest-N estimates as reference.
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
    >>> conv = {100: [0.9, 1.1], 500: [0.99, 1.01], 1000: [0.999, 1.001]}
    >>> fig, ax = plot_convergence_rate(conv, true_value=1.0)
    """
    fig, ax = _get_fig_ax(ax)
    colors = get_colors()

    if isinstance(conv_results, dict):
        ns = sorted(conv_results.keys())
        estimates_by_n = conv_results
    else:
        ns = list(getattr(conv_results, "n_particles_list", []))
        estimates_by_n = getattr(conv_results, "estimates", {})

    if true_value is None:
        largest_n = max(ns)
        true_value = float(np.mean(estimates_by_n[largest_n]))

    rmses = []
    for n in ns:
        vals = np.asarray(estimates_by_n[n])
        rmses.append(np.sqrt(np.mean((vals - true_value) ** 2)))

    ns_arr = np.array(ns, dtype=float)
    rmses_arr = np.array(rmses)

    plot_kwargs: dict[str, Any] = {
        "color": colors[0],
        "marker": "o",
        "linewidth": 1.5,
        "markersize": 6,
        "label": "RMSE",
    }
    plot_kwargs.update(kwargs)

    ax.loglog(ns_arr, rmses_arr, **plot_kwargs)

    # Reference 1/sqrt(N) line
    c = rmses_arr[0] * np.sqrt(ns_arr[0])
    ref_line = c / np.sqrt(ns_arr)
    ax.loglog(
        ns_arr,
        ref_line,
        color=colors[1],
        linestyle="--",
        alpha=0.7,
        label=r"$O(1/\sqrt{N})$",
    )

    ax.set_xlabel("Number of Particles (N)")
    ax.set_ylabel("RMSE")
    ax.set_title("Convergence Rate")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_loglike_distribution(
    results_by_n: dict[int, list[float]] | Any,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot distribution of log-likelihood estimates for different N.

    Shows how the variability of the log-likelihood estimate decreases
    with more particles, useful for choosing particle count.

    Parameters
    ----------
    results_by_n : dict[int, list[float]] or similar
        Mapping from N to list of log-likelihood estimates from repeated runs.
    ax : Axes or None
        Matplotlib axes. If None, creates new figure.
    **kwargs
        Additional keyword arguments passed to `ax.boxplot()`.

    Returns
    -------
    tuple of (Figure, Axes)
        The figure and axes objects.
    """
    fig, ax = _get_fig_ax(ax)
    colors = get_colors()

    if isinstance(results_by_n, dict):
        ns = sorted(results_by_n.keys())
        data = [results_by_n[n] for n in ns]
    else:
        ns = list(getattr(results_by_n, "n_particles_list", []))
        estimates = getattr(results_by_n, "estimates", {})
        data = [estimates[n] for n in ns]

    bp_kwargs: dict[str, Any] = {
        "patch_artist": True,
        "notch": True,
    }
    bp_kwargs.update(kwargs)

    bp = ax.boxplot(data, tick_labels=[str(n) for n in ns], **bp_kwargs)

    # Color the boxes
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(colors[i % len(colors)])
        box.set_alpha(0.7)

    ax.set_xlabel("Number of Particles (N)")
    ax.set_ylabel("Log-Likelihood")
    ax.set_title("Log-Likelihood Distribution by N")

    return fig, ax


def plot_qq_weights(
    results: Any,
    t: int = -1,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot QQ-plot of particle weights against uniform distribution.

    Compares the empirical weight distribution with a uniform distribution.
    Deviations from the diagonal indicate weight degeneracy.

    Parameters
    ----------
    results : FilterResults
        Particle filter results. Expects `results.weights` shape (T, N) or (N,).
    t : int
        Time index. Default is -1 (last time step).
    ax : Axes or None
        Matplotlib axes. If None, creates new figure.
    **kwargs
        Additional keyword arguments passed to `ax.scatter()`.

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
    n = len(w)

    # Sort weights
    sorted_weights = np.sort(w)
    # Expected uniform quantiles
    uniform_quantiles = np.arange(1, n + 1) / (n + 1) / n

    scatter_kwargs: dict[str, Any] = {
        "color": colors[0],
        "s": 10,
        "alpha": 0.6,
    }
    scatter_kwargs.update(kwargs)

    ax.scatter(uniform_quantiles, sorted_weights, **scatter_kwargs)

    # Reference line
    max_val = max(np.max(uniform_quantiles), np.max(sorted_weights))
    ax.plot(
        [0, max_val],
        [0, max_val],
        color=colors[2],
        linestyle="--",
        alpha=0.7,
        label="Uniform reference",
    )

    ax.set_xlabel("Uniform Quantiles")
    ax.set_ylabel("Weight Quantiles")
    ax.set_title(f"QQ-Plot of Weights at t={t}")
    ax.legend(loc="best")

    return fig, ax
