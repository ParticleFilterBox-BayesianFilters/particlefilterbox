"""PMCMC diagnostic plots for particlefilterbox.

Functions for visualizing MCMC chain diagnostics including trace plots,
posterior distributions, autocorrelation, and pairwise parameter plots.

All functions accept an optional `ax` parameter for subplot composition
and return `(fig, ax)` or `(fig, axes)` tuples.

References
----------
Andrieu, C., Doucet, A. & Holenstein, R. (2010). Particle Markov chain
Monte Carlo methods. JRSS-B, 72(3), 269-342.
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


def plot_trace(
    results: Any,
    param: str | int = 0,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot MCMC trace (chain history) for a parameter.

    Shows the sampled values of a parameter across MCMC iterations,
    useful for detecting non-stationarity, mixing problems, and burn-in.

    Parameters
    ----------
    results : PMCMCResults or similar
        MCMC results. Expects `results.chain` shape (n_iter, k_params)
        or `results.chains` dict mapping param names to arrays.
    param : str or int
        Parameter name or index to plot. Default is 0.
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
    >>> fig, ax = plot_trace(results, param='mu')
    >>> fig.savefig('trace_mu.png')
    """
    fig, ax = _get_fig_ax(ax)
    colors = get_colors()

    chain = _extract_chain(results, param)
    param_name = _get_param_name(results, param)

    plot_kwargs: dict[str, Any] = {
        "color": colors[0],
        "linewidth": 0.5,
        "alpha": 0.7,
    }
    plot_kwargs.update(kwargs)

    ax.plot(range(len(chain)), chain, **plot_kwargs)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(param_name)
    ax.set_title(f"Trace: {param_name}")

    # Add running mean
    if len(chain) > 100:
        window = max(len(chain) // 50, 10)
        running_mean = np.convolve(chain, np.ones(window) / window, mode="valid")
        offset = window // 2
        ax.plot(
            range(offset, offset + len(running_mean)),
            running_mean,
            color=colors[2],
            linewidth=1.5,
            label="Running mean",
        )
        ax.legend(loc="best")

    return fig, ax


def plot_posterior(
    results: Any,
    param: str | int = 0,
    prior: Any | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot posterior distribution for a parameter.

    Shows a histogram/KDE of the posterior samples, optionally overlaid
    with the prior distribution for comparison.

    Parameters
    ----------
    results : PMCMCResults or similar
        MCMC results containing chain samples.
    param : str or int
        Parameter name or index. Default is 0.
    prior : callable or None
        Prior PDF function: prior(x) -> density. If provided, overlaid on plot.
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

    chain = _extract_chain(results, param)
    param_name = _get_param_name(results, param)

    # Discard burn-in (first 25%)
    burn_in = len(chain) // 4
    samples = chain[burn_in:]

    hist_kwargs: dict[str, Any] = {
        "bins": 50,
        "density": True,
        "color": colors[0],
        "alpha": 0.7,
        "edgecolor": "white",
        "linewidth": 0.5,
        "label": "Posterior",
    }
    hist_kwargs.update(kwargs)

    ax.hist(samples, **hist_kwargs)

    # Overlay prior if provided
    if prior is not None:
        x_range = np.linspace(samples.min(), samples.max(), 200)
        try:
            prior_vals = np.array([prior(x) for x in x_range])
            ax.plot(
                x_range,
                prior_vals,
                color=colors[1],
                linewidth=2.0,
                linestyle="--",
                label="Prior",
            )
        except (TypeError, ValueError):
            pass

    # Add posterior mean line
    post_mean = np.mean(samples)
    ax.axvline(
        x=post_mean,
        color=colors[2],
        linestyle=":",
        linewidth=1.5,
        label=f"Mean = {post_mean:.3f}",
    )

    ax.set_xlabel(param_name)
    ax.set_ylabel("Density")
    ax.set_title(f"Posterior: {param_name}")
    ax.legend(loc="best")

    return fig, ax


def plot_acf(
    results: Any,
    param: str | int = 0,
    max_lag: int = 50,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot autocorrelation function for a parameter chain.

    Measures serial correlation in the MCMC chain, useful for
    assessing mixing efficiency. Slow-decaying ACF indicates poor mixing.

    Parameters
    ----------
    results : PMCMCResults or similar
        MCMC results containing chain samples.
    param : str or int
        Parameter name or index. Default is 0.
    max_lag : int
        Maximum lag to compute. Default is 50.
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

    chain = _extract_chain(results, param)
    param_name = _get_param_name(results, param)

    # Compute ACF
    n = len(chain)
    mean = np.mean(chain)
    var = np.var(chain)
    if var < 1e-12:
        acf = np.ones(max_lag + 1)
    else:
        centered = chain - mean
        acf = np.zeros(max_lag + 1)
        for lag in range(max_lag + 1):
            if lag >= n:
                break
            acf[lag] = np.mean(centered[: n - lag] * centered[lag:]) / var

    lags = np.arange(len(acf))

    bar_kwargs: dict[str, Any] = {
        "color": colors[0],
        "alpha": 0.7,
        "width": 0.8,
    }
    bar_kwargs.update(kwargs)

    ax.bar(lags, acf, **bar_kwargs)

    # Significance bounds (approximate 95% CI)
    sig_bound = 1.96 / np.sqrt(n)
    ax.axhline(y=sig_bound, color=colors[2], linestyle="--", alpha=0.5)
    ax.axhline(y=-sig_bound, color=colors[2], linestyle="--", alpha=0.5)
    ax.axhline(y=0, color="black", linewidth=0.5)

    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.set_title(f"Autocorrelation: {param_name}")

    return fig, ax


def plot_pairplot(
    results: Any,
    params: list[str | int] | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Any]:
    """Plot pairwise scatter matrix of parameter posteriors.

    Shows bivariate scatter plots for all pairs of parameters,
    with histograms on the diagonal. Useful for detecting correlations.

    Parameters
    ----------
    results : PMCMCResults or similar
        MCMC results containing chain samples.
    params : list of (str or int) or None
        Parameters to include. If None, uses all (up to 6).
    ax : None
        Ignored for pairplot (creates its own grid).
    **kwargs
        Additional keyword arguments passed to scatter plots.

    Returns
    -------
    tuple of (Figure, ndarray of Axes)
        The figure and axes grid.
    """
    colors = get_colors()

    # Get parameter chains
    chain = _get_full_chain(results)
    param_names = _get_all_param_names(results, chain.shape[1])

    if params is not None:
        indices = []
        names = []
        for p in params:
            idx = param_names.index(p) if isinstance(p, str) else p
            indices.append(idx)
            names.append(param_names[idx])
        chain = chain[:, indices]
        param_names = names
    else:
        # Limit to 6 params
        if chain.shape[1] > 6:
            chain = chain[:, :6]
            param_names = param_names[:6]

    # Burn-in
    burn_in = chain.shape[0] // 4
    chain = chain[burn_in:]

    n_params = chain.shape[1]
    fig, axes = plt.subplots(n_params, n_params, figsize=(3 * n_params, 3 * n_params))

    if n_params == 1:
        axes = np.array([[axes]])

    scatter_kwargs: dict[str, Any] = {
        "alpha": 0.2,
        "s": 3,
        "color": colors[0],
    }
    scatter_kwargs.update(kwargs)

    for i in range(n_params):
        for j in range(n_params):
            ax_ij = axes[i, j]
            if i == j:
                ax_ij.hist(
                    chain[:, i],
                    bins=30,
                    color=colors[0],
                    alpha=0.7,
                    edgecolor="white",
                )
                ax_ij.set_ylabel("Count" if j == 0 else "")
            elif i > j:
                ax_ij.scatter(chain[:, j], chain[:, i], **scatter_kwargs)
            else:
                ax_ij.set_visible(False)

            if i == n_params - 1:
                ax_ij.set_xlabel(param_names[j])
            else:
                ax_ij.set_xticklabels([])

            if j == 0 and i != j:
                ax_ij.set_ylabel(param_names[i])
            elif j != 0:
                ax_ij.set_yticklabels([])

    fig.suptitle("Parameter Pairplot", y=1.02)
    fig.tight_layout()

    return fig, axes


def plot_running_mean(
    results: Any,
    param: str | int = 0,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot running mean of a parameter chain.

    Shows the cumulative mean of the chain, which should stabilize
    as the chain converges. Useful for assessing convergence.

    Parameters
    ----------
    results : PMCMCResults or similar
        MCMC results containing chain samples.
    param : str or int
        Parameter name or index. Default is 0.
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

    chain = _extract_chain(results, param)
    param_name = _get_param_name(results, param)

    running_mean = np.cumsum(chain) / np.arange(1, len(chain) + 1)

    plot_kwargs: dict[str, Any] = {
        "color": colors[0],
        "linewidth": 1.5,
    }
    plot_kwargs.update(kwargs)

    ax.plot(range(len(running_mean)), running_mean, **plot_kwargs)

    # Final mean
    final_mean = running_mean[-1]
    ax.axhline(
        y=final_mean,
        color=colors[2],
        linestyle="--",
        alpha=0.7,
        label=f"Final mean = {final_mean:.4f}",
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(f"Running Mean of {param_name}")
    ax.set_title(f"Running Mean: {param_name}")
    ax.legend(loc="best")

    return fig, ax


def plot_posterior_predictive(
    results: Any,
    observations: NDArray[np.floating[Any]] | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot posterior predictive distribution vs observed data.

    Compares the model's posterior predictive distribution with
    the observed data, useful for model checking.

    Parameters
    ----------
    results : PMCMCResults or similar
        MCMC results. May contain `posterior_predictive` attribute.
    observations : NDArray or None
        Observed data. If None, tries `results.observations`.
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

    if observations is None:
        observations = getattr(results, "observations", None)

    pp = getattr(results, "posterior_predictive", None)

    if pp is not None:
        pp_arr = np.asarray(pp)
        pp_flat = pp_arr.flatten() if pp_arr.ndim > 1 else pp_arr

        hist_kwargs: dict[str, Any] = {
            "bins": 50,
            "density": True,
            "alpha": 0.5,
            "edgecolor": "white",
        }
        hist_kwargs.update(kwargs)

        ax.hist(
            pp_flat,
            color=colors[0],
            label="Posterior predictive",
            **hist_kwargs,
        )

    if observations is not None:
        obs = np.asarray(observations).flatten()
        ax.hist(
            obs,
            bins=50,
            density=True,
            color=colors[2],
            alpha=0.5,
            edgecolor="white",
            label="Observed",
        )

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Posterior Predictive Check")
    ax.legend(loc="best")

    return fig, ax


# --- Helper functions ---


def _extract_chain(results: Any, param: str | int) -> NDArray[np.floating[Any]]:
    """Extract chain for a specific parameter."""
    if isinstance(param, str):
        chains = getattr(results, "chains", None)
        if chains is not None and param in chains:
            return np.asarray(chains[param])
        chain = getattr(results, "chain", None)
        if chain is not None:
            param_names = getattr(results, "param_names", None)
            if param_names is not None:
                idx = list(param_names).index(param)
                return np.asarray(chain)[:, idx]
        msg = f"Cannot find parameter '{param}' in results"
        raise KeyError(msg)
    else:
        chain = getattr(results, "chain", None)
        if chain is not None:
            return np.asarray(chain)[:, param]
        chains = getattr(results, "chains", None)
        if chains is not None:
            keys = list(chains.keys())
            return np.asarray(chains[keys[param]])
        msg = f"Cannot extract parameter at index {param}"
        raise KeyError(msg)


def _get_param_name(results: Any, param: str | int) -> str:
    """Get the display name for a parameter."""
    if isinstance(param, str):
        return param
    param_names = getattr(results, "param_names", None)
    if param_names is not None:
        return str(param_names[param])
    return f"param_{param}"


def _get_full_chain(results: Any) -> NDArray[np.floating[Any]]:
    """Extract the full chain matrix (n_iter, k_params)."""
    chain = getattr(results, "chain", None)
    if chain is not None:
        return np.asarray(chain)
    chains = getattr(results, "chains", None)
    if chains is not None:
        arrays = [np.asarray(v) for v in chains.values()]
        return np.column_stack(arrays)
    msg = "Cannot find chain data in results"
    raise AttributeError(msg)


def _get_all_param_names(results: Any, n_params: int) -> list[str]:
    """Get all parameter names."""
    param_names = getattr(results, "param_names", None)
    if param_names is not None:
        return list(param_names)
    chains = getattr(results, "chains", None)
    if chains is not None:
        return list(chains.keys())
    return [f"param_{i}" for i in range(n_params)]
