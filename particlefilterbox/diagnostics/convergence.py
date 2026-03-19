"""Convergence study for particle filters.

Verifies the theoretical sqrt(N) convergence rate by running the filter
with increasing numbers of particles and fitting a power law regression.

Reference:
    Chopin, N. (2004). Central limit theorem for sequential Monte Carlo
    methods and its application to Bayesian inference. Annals of Statistics,
    32(6), 2385-2411.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


class FilterModel(Protocol):
    """Protocol for a model that can be filtered."""

    def simulate(
        self,
        n_obs: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Simulate states and observations."""
        ...


class FilterFactory(Protocol):
    """Protocol for creating a particle filter."""

    def create(self, model: Any, n_particles: int) -> Any:
        """Create a filter with given N."""
        ...


@dataclass
class ConvergenceResult:
    """Results of a convergence study.

    Attributes:
        n_values: Array of N values tested.
        rmse_values: Array of mean RMSE for each N.
        rmse_std: Array of RMSE standard deviations.
        rate: Estimated convergence rate (beta in RMSE ~ N^(-beta)).
        intercept: Intercept of log-log regression.
        r_squared: R-squared of the log-log fit.
    """

    n_values: NDArray[np.int64]
    rmse_values: NDArray[np.float64]
    rmse_std: NDArray[np.float64]
    rate: float
    intercept: float
    r_squared: float


class ConvergenceStudy:
    """Study convergence rate of a particle filter.

    Runs the filter with different particle counts and fits
    log(RMSE) ~ a - beta * log(N) to estimate the convergence rate.
    Theoretical rate is beta ~ 0.5 (sqrt(N) convergence).

    Parameters:
        model: Model with simulate() method.
        filter_factory: Factory that creates filters with create(model, n_particles).
        n_values: List of particle counts to test.
        n_repeats: Number of repetitions per N value.
        n_obs: Number of observations to simulate.
        seed: Random seed for reproducibility.

    Examples:
        >>> cs = ConvergenceStudy(model, factory, n_values=[100, 500, 1000, 5000])
        >>> result = cs.run()
        >>> print(f"Convergence rate: {result.rate:.3f}")  # ~0.5
    """

    def __init__(
        self,
        model: Any,
        filter_factory: Any,
        n_values: list[int] | None = None,
        n_repeats: int = 50,
        n_obs: int = 100,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.filter_factory = filter_factory
        self.n_values = n_values or [100, 500, 1000, 5000]
        self.n_repeats = n_repeats
        self.n_obs = n_obs
        self.seed = seed

        self._result: ConvergenceResult | None = None

    @property
    def rate(self) -> float:
        """Estimated convergence rate beta."""
        if self._result is None:
            raise RuntimeError("Must call run() first.")
        return self._result.rate

    @property
    def rmse_by_n(self) -> dict[int, float]:
        """Dictionary mapping N -> mean RMSE."""
        if self._result is None:
            raise RuntimeError("Must call run() first.")
        return dict(
            zip(
                self._result.n_values.tolist(),
                self._result.rmse_values.tolist(),
                strict=True,
            )
        )

    def run(self) -> ConvergenceResult:
        """Run the convergence study.

        Returns:
            ConvergenceResult with estimated rate and RMSE values.
        """
        rng = np.random.default_rng(self.seed)

        # Simulate true states and observations once
        states_true, observations = self.model.simulate(self.n_obs, rng=rng)

        n_arr = np.array(self.n_values, dtype=np.int64)
        rmse_means = np.zeros(len(self.n_values), dtype=np.float64)
        rmse_stds = np.zeros(len(self.n_values), dtype=np.float64)

        for i, n_particles in enumerate(self.n_values):
            rmse_list: list[float] = []

            for _rep in range(self.n_repeats):
                rng.integers(0, 2**31)  # advance rng state for reproducibility

                # Create filter and run
                pf = self.filter_factory.create(self.model, n_particles)
                result = pf.filter(observations)

                # Compute RMSE of filtered mean vs true states
                filtered_mean: NDArray[np.float64] = np.asarray(
                    result.filtered_means,
                    dtype=np.float64,
                )
                if filtered_mean.ndim > 1:
                    filtered_mean = filtered_mean[:, 0]
                true_states: NDArray[np.float64] = states_true
                if true_states.ndim > 1:
                    true_states = true_states[:, 0]

                n_compare = min(len(filtered_mean), len(true_states))
                diff = filtered_mean[:n_compare] - true_states[:n_compare]
                rmse = float(np.sqrt(np.mean(diff**2)))
                rmse_list.append(rmse)

            rmse_means[i] = float(np.mean(rmse_list))
            rmse_stds[i] = float(np.std(rmse_list))

        # Fit log-log regression: log(RMSE) = a - beta * log(N)
        log_n = np.log(n_arr.astype(np.float64))
        log_rmse = np.log(rmse_means)

        # OLS fit
        design = np.column_stack([np.ones_like(log_n), log_n])
        lstsq_result = np.linalg.lstsq(design, log_rmse, rcond=None)
        coeffs = lstsq_result[0]
        intercept = float(coeffs[0])
        slope = float(coeffs[1])
        rate = -slope  # beta = -slope since RMSE ~ N^(-beta)

        # R-squared
        ss_res = float(np.sum((log_rmse - design @ coeffs) ** 2))
        ss_tot = float(np.sum((log_rmse - np.mean(log_rmse)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        self._result = ConvergenceResult(
            n_values=n_arr,
            rmse_values=rmse_means,
            rmse_std=rmse_stds,
            rate=rate,
            intercept=intercept,
            r_squared=r_squared,
        )
        return self._result

    def summary(self) -> dict[str, Any]:
        """Generate summary of convergence study.

        Returns:
            Dictionary with rate, R-squared, and RMSE by N.
        """
        if self._result is None:
            raise RuntimeError("Must call run() first.")

        r = self._result
        return {
            "rate": r.rate,
            "expected_rate": 0.5,
            "rate_in_range": 0.3 <= r.rate <= 0.7,
            "r_squared": r.r_squared,
            "rmse_by_n": self.rmse_by_n,
            "n_values": r.n_values.tolist(),
        }
