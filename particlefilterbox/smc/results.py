"""SMCResults: container for Sequential Monte Carlo output.

Provides methods for posterior summaries, credible intervals,
and conversion to pandas DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from particlefilterbox._logging import get_logger

logger = get_logger("smc.results")


@dataclass
class SMCResults:
    """Container for SMC algorithm output.

    Stores the final particle population, weights, log-evidence estimate,
    and diagnostic information. Provides methods for posterior analysis.

    Attributes
    ----------
    particles : NDArray, shape (N, k_params)
        Final particle positions in parameter space.
    weights : NDArray, shape (N,)
        Normalized importance weights (sum to 1).
    log_evidence : float
        Estimate of log marginal likelihood log p(y).
    param_names : list[str] or None
        Names for each parameter dimension. If None, uses 'param_0', etc.
    schedule : list[float]
        Tempering schedule (beta values) used during the run.
    ess_history : list[float]
        ESS at each step of the algorithm.
    acceptance_rates : list[float]
        MCMC acceptance rates at each rejuvenation step.
    n_steps : int
        Total number of SMC steps (tempering stages).

    Examples
    --------
    >>> import numpy as np
    >>> particles = np.random.randn(500, 2)
    >>> weights = np.ones(500) / 500
    >>> results = SMCResults(
    ...     particles=particles,
    ...     weights=weights,
    ...     log_evidence=-10.5,
    ...     n_steps=15,
    ... )
    >>> print(results.summary())
    """

    particles: NDArray[np.floating[Any]]
    weights: NDArray[np.floating[Any]]
    log_evidence: float
    param_names: list[str] | None = None
    schedule: list[float] = field(default_factory=lambda: list[float]())
    ess_history: list[float] = field(default_factory=lambda: list[float]())
    acceptance_rates: list[float] = field(default_factory=lambda: list[float]())
    n_steps: int = 0

    def __post_init__(self) -> None:
        """Validate inputs and set defaults."""
        self.particles = np.asarray(self.particles, dtype=np.float64)
        self.weights = np.asarray(self.weights, dtype=np.float64)

        if self.particles.ndim == 1:
            self.particles = self.particles[:, np.newaxis]

        n_particles, k_params = self.particles.shape

        if len(self.weights) != n_particles:
            msg = f"weights length ({len(self.weights)}) != n_particles ({n_particles})"
            raise ValueError(msg)

        # Normalize weights
        w_sum = np.sum(self.weights)
        if w_sum > 0:
            self.weights = self.weights / w_sum

        if self.param_names is None:
            self.param_names = [f"param_{i}" for i in range(k_params)]
        elif len(self.param_names) != k_params:
            msg = f"param_names length ({len(self.param_names)}) != k_params ({k_params})"
            raise ValueError(msg)

    @property
    def n_particles(self) -> int:
        """Number of particles."""
        return self.particles.shape[0]

    @property
    def k_params(self) -> int:
        """Number of parameter dimensions."""
        return self.particles.shape[1]

    @property
    def param_mean(self) -> dict[str, float]:
        """Weighted posterior mean for each parameter.

        Returns
        -------
        dict[str, float]
            Map from parameter name to weighted mean.
        """
        means = self.posterior_mean()
        assert self.param_names is not None
        return {name: float(means[i]) for i, name in enumerate(self.param_names)}

    @property
    def param_cov(self) -> NDArray[np.floating[Any]]:
        """Weighted posterior covariance matrix.

        Returns
        -------
        NDArray, shape (k, k)
            Weighted covariance matrix.
        """
        mean = self.posterior_mean()
        diff = self.particles - mean[np.newaxis, :]
        weighted_diff = diff * self.weights[:, np.newaxis]
        cov: NDArray[np.floating[Any]] = weighted_diff.T @ diff
        return cov

    @property
    def param_quantiles(self) -> dict[str, dict[str, float]]:
        """Weighted posterior quantiles (5%, 25%, 50%, 75%, 95%).

        Returns
        -------
        dict[str, dict[str, float]]
            Nested dict: param_name -> quantile_label -> value.
        """
        quantile_levels = [0.05, 0.25, 0.50, 0.75, 0.95]
        quantile_labels = ["5%", "25%", "50%", "75%", "95%"]
        result: dict[str, dict[str, float]] = {}

        assert self.param_names is not None
        for j, name in enumerate(self.param_names):
            values = self.particles[:, j]
            sorted_idx = np.argsort(values)
            sorted_vals = values[sorted_idx]
            cum_weights = np.cumsum(self.weights[sorted_idx])

            quantiles: dict[str, float] = {}
            for level, label in zip(quantile_levels, quantile_labels, strict=False):
                idx = np.searchsorted(cum_weights, level)
                idx = min(idx, len(sorted_vals) - 1)
                quantiles[label] = float(sorted_vals[idx])
            result[name] = quantiles

        return result

    def posterior_mean(self) -> NDArray[np.floating[Any]]:
        """Weighted posterior mean.

        Returns
        -------
        NDArray, shape (k,)
            Weighted mean of particles.
        """
        return np.average(self.particles, weights=self.weights, axis=0)

    def posterior_std(self) -> NDArray[np.floating[Any]]:
        """Weighted posterior standard deviation.

        Returns
        -------
        NDArray, shape (k,)
            Weighted standard deviation for each parameter.
        """
        return np.sqrt(np.diag(self.param_cov))

    def credible_interval(
        self,
        level: float = 0.95,
    ) -> NDArray[np.floating[Any]]:
        """Weighted credible interval for each parameter.

        Parameters
        ----------
        level : float
            Credible level (e.g., 0.95 for 95% CI). Default is 0.95.

        Returns
        -------
        NDArray, shape (k, 2)
            Lower and upper bounds for each parameter.
        """
        alpha = (1.0 - level) / 2.0
        intervals = np.zeros((self.k_params, 2))

        for j in range(self.k_params):
            values = self.particles[:, j]
            sorted_idx = np.argsort(values)
            sorted_vals = values[sorted_idx]
            cum_weights = np.cumsum(self.weights[sorted_idx])

            lo_idx = np.searchsorted(cum_weights, alpha)
            hi_idx = np.searchsorted(cum_weights, 1.0 - alpha)

            lo_idx = min(lo_idx, len(sorted_vals) - 1)
            hi_idx = min(hi_idx, len(sorted_vals) - 1)

            intervals[j, 0] = sorted_vals[lo_idx]
            intervals[j, 1] = sorted_vals[hi_idx]

        return intervals

    def summary(self) -> str:
        """Human-readable summary of SMC results.

        Returns
        -------
        str
            Formatted summary string.
        """
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("SMC Results Summary")
        lines.append("=" * 60)
        lines.append(f"  N particles:   {self.n_particles}")
        lines.append(f"  K parameters:  {self.k_params}")
        lines.append(f"  N steps:       {self.n_steps}")
        lines.append(f"  Log-evidence:  {self.log_evidence:.4f}")

        if self.ess_history:
            lines.append(f"  Final ESS:     {self.ess_history[-1]:.1f}")
        if self.acceptance_rates:
            mean_acc = float(np.mean(self.acceptance_rates))
            lines.append(f"  Mean acc rate: {mean_acc:.3f}")

        lines.append("")
        lines.append("  Parameter summaries:")
        lines.append(f"  {'Name':<15} {'Mean':>10} {'Std':>10} {'CI_lo':>10} {'CI_hi':>10}")
        lines.append("  " + "-" * 55)

        means = self.posterior_mean()
        stds = self.posterior_std()
        ci = self.credible_interval(level=0.95)

        assert self.param_names is not None
        for j, name in enumerate(self.param_names):
            lines.append(
                f"  {name:<15} {means[j]:>10.4f} {stds[j]:>10.4f} "
                f"{ci[j, 0]:>10.4f} {ci[j, 1]:>10.4f}"
            )

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        """Convert particles and weights to a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns for each parameter and 'weight'.
        """
        import pandas as pd

        assert self.param_names is not None
        df = pd.DataFrame(self.particles, columns=self.param_names)
        df["weight"] = self.weights
        return df
