"""Model-specific visualization plots for particlefilterbox.

Functions for visualizing stochastic volatility, jump indicators,
regime probabilities, and impulse response functions.

All functions accept an optional `ax` parameter for subplot composition
and return `(fig, ax)` tuples.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

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


def plot_volatility(
    results: Any,
    true_vol: NDArray[np.floating[Any]] | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot estimated stochastic volatility over time.

    Shows the filtered log-volatility estimate with credible intervals,
    optionally compared to the true volatility path.

    Parameters
    ----------
    results : FilterResults
        Filter results. Expects `results.filtered_mean` or `results.particles`
        with volatility in state_idx=0.
    true_vol : NDArray or None
        True log-volatility path shape (T,) for comparison.
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
    >>> fig, ax = plot_volatility(results, true_vol=true_h)
    >>> fig.savefig('volatility.png')
    """
    fig, ax = _get_fig_ax(ax)
    colors = get_colors()

    # Extract volatility estimate
    filtered_mean = getattr(results, "filtered_mean", None)
    if filtered_mean is not None:
        vol = np.asarray(filtered_mean)
        if vol.ndim == 2:
            vol = vol[:, 0]
    else:
        particles = np.asarray(results.particles)
        weights = np.asarray(results.weights)
        t_len = particles.shape[0]
        vol = np.zeros(t_len)
        for t in range(t_len):
            vol[t] = np.average(particles[t, :, 0], weights=weights[t])

    t_len = len(vol)
    time = np.arange(t_len)

    plot_kwargs: dict[str, Any] = {
        "color": colors[0],
        "linewidth": 1.5,
        "label": "Estimated volatility",
    }
    plot_kwargs.update(kwargs)

    ax.plot(time, vol, **plot_kwargs)

    if true_vol is not None:
        true_arr = np.asarray(true_vol)
        ax.plot(
            time[: len(true_arr)],
            true_arr[:t_len],
            color=colors[2],
            linewidth=1.0,
            linestyle="--",
            alpha=0.8,
            label="True volatility",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Log-Volatility")
    ax.set_title("Stochastic Volatility")
    ax.legend(loc="best")

    return fig, ax


def plot_jump_indicators(
    results: Any,
    threshold: float = 0.5,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot jump probability indicators over time.

    Shows the filtered probability of a jump occurring at each time step,
    useful for jump-diffusion and regime-switching models.

    Parameters
    ----------
    results : FilterResults
        Filter results. Expects `results.jump_probs` shape (T,) or
        jump indicators in the state vector.
    threshold : float
        Probability threshold for highlighting jumps. Default is 0.5.
    ax : Axes or None
        Matplotlib axes. If None, creates new figure.
    **kwargs
        Additional keyword arguments passed to `ax.bar()`.

    Returns
    -------
    tuple of (Figure, Axes)
        The figure and axes objects.
    """
    fig, ax = _get_fig_ax(ax)
    colors = get_colors()

    jump_probs = getattr(results, "jump_probs", None)
    if jump_probs is None:
        jump_probs = getattr(results, "jump_indicators", np.array([]))

    probs = np.asarray(jump_probs)
    t_len = len(probs)
    time = np.arange(t_len)

    bar_kwargs: dict[str, Any] = {
        "alpha": 0.7,
        "width": 1.0,
    }
    bar_kwargs.update(kwargs)

    # Color bars above/below threshold differently
    above = probs >= threshold
    below = ~above

    if np.any(above):
        ax.bar(time[above], probs[above], color=colors[3], label="Jump", **bar_kwargs)
    if np.any(below):
        ax.bar(time[below], probs[below], color=colors[0], label="No jump", **bar_kwargs)

    ax.axhline(
        y=threshold,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Threshold={threshold}",
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Jump Probability")
    ax.set_title("Jump Indicators")
    ax.legend(loc="best")
    ax.set_ylim(0, 1)

    return fig, ax


def plot_regime_probabilities(
    results: Any,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot filtered regime probabilities over time.

    Shows the probability of being in each regime at each time step,
    useful for Markov-switching and regime-switching models.

    Parameters
    ----------
    results : FilterResults
        Filter results. Expects `results.regime_probs` shape (T, K)
        where K is the number of regimes.
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

    regime_probs = getattr(results, "regime_probs", None)
    if regime_probs is None:
        msg = "Results must contain 'regime_probs' attribute"
        raise AttributeError(msg)

    probs = np.asarray(regime_probs)
    t_len, k = probs.shape
    time = np.arange(t_len)

    # Stacked area plot
    bottom = np.zeros(t_len)
    for i in range(k):
        color = colors[i % len(colors)]
        ax.fill_between(
            time,
            bottom,
            bottom + probs[:, i],
            color=color,
            alpha=0.7,
            label=f"Regime {i + 1}",
            **kwargs,
        )
        bottom += probs[:, i]

    ax.set_xlabel("Time")
    ax.set_ylabel("Probability")
    ax.set_title("Regime Probabilities")
    ax.legend(loc="best")
    ax.set_ylim(0, 1)

    return fig, ax


def plot_irf(
    results: Any,
    shock_idx: int = 0,
    response_idx: int = 0,
    n_periods: int = 20,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot impulse response function.

    Shows the dynamic response of a variable to a one-standard-deviation
    shock, with credible intervals from posterior uncertainty.

    Parameters
    ----------
    results : Any
        Results containing `irf` attribute of shape (n_draws, n_periods, n_vars, n_shocks)
        or (n_periods, n_vars) for a single IRF.
    shock_idx : int
        Index of the shock. Default is 0.
    response_idx : int
        Index of the response variable. Default is 0.
    n_periods : int
        Number of periods to plot. Default is 20.
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

    irf_data = getattr(results, "irf", None)
    if irf_data is None:
        msg = "Results must contain 'irf' attribute"
        raise AttributeError(msg)

    irf = np.asarray(irf_data)

    if irf.ndim == 4:
        # (n_draws, n_periods, n_vars, n_shocks)
        irf_selected = irf[:, :n_periods, response_idx, shock_idx]
        median = np.median(irf_selected, axis=0)
        q05 = np.percentile(irf_selected, 5, axis=0)
        q95 = np.percentile(irf_selected, 95, axis=0)

        periods = np.arange(len(median))
        ax.fill_between(periods, q05, q95, color=colors[0], alpha=0.2, label="90% CI")
        ax.plot(periods, median, color=colors[0], linewidth=1.5, label="Median", **kwargs)
    elif irf.ndim == 2:
        # (n_periods, n_vars)
        response = irf[:n_periods, response_idx]
        periods = np.arange(len(response))
        ax.plot(periods, response, color=colors[0], linewidth=1.5, **kwargs)
    else:
        response = irf[:n_periods]
        periods = np.arange(len(response))
        ax.plot(periods, response, color=colors[0], linewidth=1.5, **kwargs)

    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.set_xlabel("Period")
    ax.set_ylabel("Response")
    ax.set_title(f"Impulse Response (shock={shock_idx}, response={response_idx})")
    ax.legend(loc="best")

    return fig, ax
