"""State estimation plots for particlefilterbox.

Functions for visualizing filtered states, smoothed states,
observation fit, and forecasts from particle filter results.

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


def plot_filtered_state(
    results: Any,
    state_idx: int = 0,
    true_state: NDArray[np.floating[Any]] | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot filtered state estimate with credible intervals.

    Shows the weighted mean of the filtering distribution along with
    50% and 90% credible intervals. Optionally overlays the true state.

    Parameters
    ----------
    results : FilterResults
        Particle filter results. Expects `results.filtered_mean` shape (T, d)
        or `results.particles` shape (T, N, d) and `results.weights` shape (T, N).
    state_idx : int
        State dimension index to plot. Default is 0.
    true_state : NDArray or None
        True state values shape (T,) for comparison. Default is None.
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
    >>> fig, ax = plot_filtered_state(results, state_idx=0, true_state=x_true)
    >>> fig.savefig('filtered_state.png')
    """
    fig, ax = _get_fig_ax(ax)
    colors = get_colors()

    # Try to get pre-computed filtered mean
    filtered_mean = getattr(results, "filtered_mean", None)
    filtered_ci_lower = getattr(results, "filtered_ci_lower", None)
    filtered_ci_upper = getattr(results, "filtered_ci_upper", None)

    ci50_lower: NDArray[np.floating[Any]] | None = None
    ci50_upper: NDArray[np.floating[Any]] | None = None

    if filtered_mean is not None:
        mean_vals = np.asarray(filtered_mean)
        if mean_vals.ndim == 2:
            mean_vals = mean_vals[:, state_idx]
        n_times = len(mean_vals)
    else:
        # Compute from particles and weights
        particles = np.asarray(results.particles)
        weights = np.asarray(results.weights)
        n_times, n_particles, _ = particles.shape

        mean_vals = np.zeros(n_times)
        ci_lower = np.zeros(n_times)
        ci_upper = np.zeros(n_times)
        ci50_lo = np.zeros(n_times)
        ci50_hi = np.zeros(n_times)

        for t in range(n_times):
            pts = particles[t, :, state_idx]
            w = weights[t]
            w = w / w.sum()
            mean_vals[t] = np.average(pts, weights=w)

            sorted_idx = np.argsort(pts)
            sorted_pts = pts[sorted_idx]
            cum_w = np.cumsum(w[sorted_idx])

            ci_lower[t] = sorted_pts[np.searchsorted(cum_w, 0.05)]
            ci_upper[t] = sorted_pts[min(np.searchsorted(cum_w, 0.95), n_particles - 1)]
            ci50_lo[t] = sorted_pts[np.searchsorted(cum_w, 0.25)]
            ci50_hi[t] = sorted_pts[min(np.searchsorted(cum_w, 0.75), n_particles - 1)]

        filtered_ci_lower = ci_lower
        filtered_ci_upper = ci_upper
        ci50_lower = ci50_lo
        ci50_upper = ci50_hi

    time = np.arange(n_times)

    # Plot credible intervals
    if filtered_ci_lower is not None and filtered_ci_upper is not None:
        ci_lo = np.asarray(filtered_ci_lower)
        ci_hi = np.asarray(filtered_ci_upper)
        if ci_lo.ndim == 2:
            ci_lo = ci_lo[:, state_idx]
            ci_hi = ci_hi[:, state_idx]
        ax.fill_between(time, ci_lo, ci_hi, color=colors[0], alpha=0.15, label="90% CI")

    # Plot 50% CI if computed
    if ci50_lower is not None and ci50_upper is not None:
        ax.fill_between(time, ci50_lower, ci50_upper, color=colors[0], alpha=0.3, label="50% CI")

    # Plot filtered mean
    plot_kwargs: dict[str, Any] = {
        "color": colors[0],
        "linewidth": 1.5,
        "label": "Filtered mean",
    }
    plot_kwargs.update(kwargs)
    ax.plot(time, mean_vals, **plot_kwargs)

    # Plot true state if provided
    if true_state is not None:
        true_arr = np.asarray(true_state)
        ax.plot(
            time[: len(true_arr)],
            true_arr[:n_times],
            color=colors[2],
            linewidth=1.0,
            linestyle="--",
            alpha=0.8,
            label="True state",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel(f"State {state_idx}")
    ax.set_title("Filtered State Estimate")
    ax.legend(loc="best")

    return fig, ax


def plot_smoothed_state(
    results: Any,
    state_idx: int = 0,
    true_state: NDArray[np.floating[Any]] | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot smoothed state estimate with credible intervals.

    Similar to `plot_filtered_state` but uses smoothed estimates,
    which incorporate information from the entire observation sequence.

    Parameters
    ----------
    results : SmootherResults or FilterResults
        Results containing `smoothed_mean` or `smoothed_particles`.
    state_idx : int
        State dimension index to plot. Default is 0.
    true_state : NDArray or None
        True state values shape (T,) for comparison.
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

    smoothed_mean = getattr(results, "smoothed_mean", None)
    if smoothed_mean is not None:
        mean_vals = np.asarray(smoothed_mean)
        if mean_vals.ndim == 2:
            mean_vals = mean_vals[:, state_idx]
    else:
        # Fallback: use filtered mean
        filtered_mean = getattr(results, "filtered_mean", None)
        if filtered_mean is not None:
            mean_vals = np.asarray(filtered_mean)
            if mean_vals.ndim == 2:
                mean_vals = mean_vals[:, state_idx]
        else:
            msg = "Results must contain 'smoothed_mean' or 'filtered_mean'"
            raise AttributeError(msg)

    n_times = len(mean_vals)
    time = np.arange(n_times)

    plot_kwargs: dict[str, Any] = {
        "color": colors[1],
        "linewidth": 1.5,
        "label": "Smoothed mean",
    }
    plot_kwargs.update(kwargs)
    ax.plot(time, mean_vals, **plot_kwargs)

    if true_state is not None:
        true_arr = np.asarray(true_state)
        ax.plot(
            time[: len(true_arr)],
            true_arr[:n_times],
            color=colors[2],
            linewidth=1.0,
            linestyle="--",
            alpha=0.8,
            label="True state",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel(f"State {state_idx}")
    ax.set_title("Smoothed State Estimate")
    ax.legend(loc="best")

    return fig, ax


def plot_filtered_vs_smoothed(
    results: Any,
    state_idx: int = 0,
    true_state: NDArray[np.floating[Any]] | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot filtered and smoothed state estimates side by side.

    Compares the filtering and smoothing distributions, showing
    how the smoother reduces uncertainty using future observations.

    Parameters
    ----------
    results : FilterResults or SmootherResults
        Results containing both filtered and smoothed estimates.
    state_idx : int
        State dimension index to plot. Default is 0.
    true_state : NDArray or None
        True state values shape (T,) for comparison.
    ax : Axes or None
        Matplotlib axes. If None, creates new figure.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    tuple of (Figure, Axes)
        The figure and axes objects.
    """
    fig, ax = _get_fig_ax(ax)
    colors = get_colors()

    n_times = 0

    # Get filtered mean
    filtered_mean = getattr(results, "filtered_mean", None)
    smoothed_mean = getattr(results, "smoothed_mean", None)

    if filtered_mean is not None:
        filt = np.asarray(filtered_mean)
        if filt.ndim == 2:
            filt = filt[:, state_idx]
        n_times = len(filt)
        time = np.arange(n_times)
        ax.plot(time, filt, color=colors[0], linewidth=1.2, label="Filtered", alpha=0.8)

    if smoothed_mean is not None:
        smooth = np.asarray(smoothed_mean)
        if smooth.ndim == 2:
            smooth = smooth[:, state_idx]
        n_times = len(smooth)
        time = np.arange(n_times)
        ax.plot(time, smooth, color=colors[1], linewidth=1.5, label="Smoothed")

    if true_state is not None:
        true_arr = np.asarray(true_state)
        t_plot = min(len(true_arr), n_times)
        ax.plot(
            np.arange(t_plot),
            true_arr[:t_plot],
            color=colors[2],
            linewidth=1.0,
            linestyle="--",
            alpha=0.8,
            label="True state",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel(f"State {state_idx}")
    ax.set_title("Filtered vs Smoothed State")
    ax.legend(loc="best")

    return fig, ax


def plot_observation_fit(
    results: Any,
    observations: NDArray[np.floating[Any]] | None = None,
    obs_idx: int = 0,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot observation fit: predicted vs actual observations.

    Shows how well the model's predicted observations match the
    actual data, using the filtered state estimates.

    Parameters
    ----------
    results : FilterResults
        Particle filter results. May contain `predicted_obs` or `observations`.
    observations : NDArray or None
        Observed data shape (T,) or (T, m). If None, tries `results.observations`.
    obs_idx : int
        Observation dimension index. Default is 0.
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

    if observations is None:
        observations = getattr(results, "observations", None)
    if observations is None:
        msg = "Must provide observations or results.observations"
        raise ValueError(msg)

    obs = np.asarray(observations)
    if obs.ndim == 2:
        obs = obs[:, obs_idx]

    n_obs = len(obs)
    time = np.arange(n_obs)

    # Plot observations
    ax.plot(time, obs, color=colors[3], alpha=0.5, linewidth=0.8, label="Observed")

    # Plot predicted observations if available
    predicted = getattr(results, "predicted_obs", None)
    if predicted is not None:
        pred = np.asarray(predicted)
        if pred.ndim == 2:
            pred = pred[:, obs_idx]
        ax.plot(time, pred[:n_obs], color=colors[0], linewidth=1.5, label="Predicted")

    ax.set_xlabel("Time")
    ax.set_ylabel(f"Observation {obs_idx}")
    ax.set_title("Observation Fit")
    ax.legend(loc="best")

    return fig, ax


def plot_forecast(
    results: Any,
    n_ahead: int = 10,
    state_idx: int = 0,
    observations: NDArray[np.floating[Any]] | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot state forecast with prediction intervals.

    Extends the filtered state estimate into the future, showing
    how uncertainty grows with the forecast horizon.

    Parameters
    ----------
    results : FilterResults
        Particle filter results. May contain `forecast_mean` and `forecast_ci`.
    n_ahead : int
        Number of steps to forecast ahead. Default is 10.
    state_idx : int
        State dimension index. Default is 0.
    observations : NDArray or None
        Historical observations for context.
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

    n_times = 0
    filt = None

    # Get filtered history
    filtered_mean = getattr(results, "filtered_mean", None)
    if filtered_mean is not None:
        filt = np.asarray(filtered_mean)
        if filt.ndim == 2:
            filt = filt[:, state_idx]
        n_times = len(filt)
        time_hist = np.arange(n_times)
        ax.plot(time_hist, filt, color=colors[0], linewidth=1.5, label="Filtered")

    # Get forecast if available
    forecast_mean = getattr(results, "forecast_mean", None)
    forecast_ci_lower = getattr(results, "forecast_ci_lower", None)
    forecast_ci_upper = getattr(results, "forecast_ci_upper", None)

    if forecast_mean is not None:
        fc = np.asarray(forecast_mean)
        if fc.ndim == 2:
            fc = fc[:, state_idx]
        t_hist = len(filt) if filt is not None else 0
        time_fc = np.arange(t_hist, t_hist + len(fc))

        ax.plot(
            time_fc,
            fc,
            color=colors[1],
            linewidth=1.5,
            linestyle="--",
            label="Forecast",
        )

        if forecast_ci_lower is not None and forecast_ci_upper is not None:
            fc_lo = np.asarray(forecast_ci_lower)
            fc_hi = np.asarray(forecast_ci_upper)
            if fc_lo.ndim == 2:
                fc_lo = fc_lo[:, state_idx]
                fc_hi = fc_hi[:, state_idx]
            ax.fill_between(
                time_fc,
                fc_lo,
                fc_hi,
                color=colors[1],
                alpha=0.2,
                label="Forecast 90% CI",
            )

    # Vertical line at forecast start
    if filtered_mean is not None:
        ax.axvline(x=n_times - 1, color="gray", linestyle=":", alpha=0.5)

    ax.set_xlabel("Time")
    ax.set_ylabel(f"State {state_idx}")
    ax.set_title("State Forecast")
    ax.legend(loc="best")

    return fig, ax
