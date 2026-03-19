"""Visualization module for particlefilterbox.

Provides plotting functions for particle filters, PMCMC diagnostics,
state estimation, and model-specific visualizations.

All plot functions follow matplotlib conventions:
- Accept optional `ax` parameter for subplot composition
- Accept `**kwargs` passed to underlying matplotlib calls
- Return `(fig, ax)` tuple

Themes:
    Use `set_theme('nodesecon')` to set the institutional color palette.

Examples
--------
>>> from particlefilterbox.visualization import plot_filtered_state, set_theme
>>> set_theme('nodesecon')
>>> fig, ax = plot_filtered_state(results, state_idx=0)
>>> fig.savefig('filtered.png')
"""

from __future__ import annotations

from particlefilterbox.visualization.convergence_plots import (
    plot_convergence_rate,
    plot_loglike_distribution,
    plot_qq_weights,
)
from particlefilterbox.visualization.export import save_figure
from particlefilterbox.visualization.model_plots import (
    plot_irf,
    plot_jump_indicators,
    plot_regime_probabilities,
    plot_volatility,
)
from particlefilterbox.visualization.particle_plots import (
    plot_ancestral_tree,
    plot_particle_cloud,
    plot_particle_evolution,
    plot_particle_trajectories,
)
from particlefilterbox.visualization.pmcmc_plots import (
    plot_acf,
    plot_pairplot,
    plot_posterior,
    plot_posterior_predictive,
    plot_running_mean,
    plot_trace,
)
from particlefilterbox.visualization.state_plots import (
    plot_filtered_state,
    plot_filtered_vs_smoothed,
    plot_forecast,
    plot_observation_fit,
    plot_smoothed_state,
)
from particlefilterbox.visualization.themes import get_theme, set_theme
from particlefilterbox.visualization.weight_plots import (
    plot_ess_timeline,
    plot_weight_entropy,
    plot_weight_histogram,
    plot_weight_max,
)

__all__ = [
    "get_theme",
    "set_theme",
    "save_figure",
    "plot_particle_cloud",
    "plot_particle_trajectories",
    "plot_ancestral_tree",
    "plot_particle_evolution",
    "plot_ess_timeline",
    "plot_weight_histogram",
    "plot_weight_entropy",
    "plot_weight_max",
    "plot_filtered_state",
    "plot_smoothed_state",
    "plot_filtered_vs_smoothed",
    "plot_observation_fit",
    "plot_forecast",
    "plot_trace",
    "plot_posterior",
    "plot_acf",
    "plot_pairplot",
    "plot_running_mean",
    "plot_posterior_predictive",
    "plot_volatility",
    "plot_jump_indicators",
    "plot_regime_probabilities",
    "plot_irf",
    "plot_convergence_rate",
    "plot_loglike_distribution",
    "plot_qq_weights",
]
