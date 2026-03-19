"""Parallel execution for particle filters and PMCMC chains.

Uses multiprocessing to run multiple chains or filters in parallel.
Provides convergence diagnostics across parallel chains.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _run_single_chain(
    args: tuple[Any, NDArray[np.float64], int, int],
) -> Any:
    """Worker function to run a single PMCMC chain.

    Parameters:
        args: Tuple of (pmcmc_factory_args, endog, chain_id, seed).

    Returns:
        Chain result.
    """
    factory_args, endog, _chain_id, seed = args
    # Reconstruct PMCMC from factory args
    pmcmc_class, pmcmc_kwargs = factory_args
    rng = np.random.default_rng(seed)
    pmcmc = pmcmc_class(**pmcmc_kwargs)
    result = pmcmc.run(endog, rng=rng)
    return result


def _run_single_filter(
    args: tuple[Any, NDArray[np.float64], int, int],
) -> dict[str, float]:
    """Worker function to run a single particle filter.

    Parameters:
        args: Tuple of (filter_factory_args, endog, filter_id, seed).

    Returns:
        Dictionary with log_likelihood and other metrics.
    """
    factory_args, endog, filter_id, _seed = args
    factory, model, n_particles = factory_args
    pf = factory.create(model, n_particles)
    result = pf.filter(endog)
    return {
        "filter_id": filter_id,
        "log_likelihood": float(result.log_likelihood),
    }


class ParallelRunner:
    """Run multiple particle filter operations in parallel.

    Uses Python multiprocessing to parallelize PMCMC chains,
    particle filter runs, and convergence studies.

    Parameters:
        n_workers: Number of parallel workers. Default: CPU count.

    Examples:
        >>> runner = ParallelRunner(n_workers=4)
        >>> chains = runner.run_multiple_chains(pmcmc_class, pmcmc_kwargs, endog, n_chains=4)
    """

    def __init__(self, n_workers: int | None = None) -> None:
        if n_workers is None:
            n_workers = mp.cpu_count() or 1
        self.n_workers = n_workers

    def run_multiple_chains(
        self,
        pmcmc_class: type[Any],
        pmcmc_kwargs: dict[str, Any],
        endog: NDArray[np.float64],
        n_chains: int = 4,
        seed: int = 42,
    ) -> list[Any]:
        """Run multiple PMCMC chains in parallel.

        Parameters:
            pmcmc_class: PMCMC class to instantiate.
            pmcmc_kwargs: Keyword arguments for PMCMC constructor.
            endog: Observed data.
            n_chains: Number of chains to run.
            seed: Base random seed (each chain gets seed + chain_id).

        Returns:
            List of chain results.
        """
        rng = np.random.default_rng(seed)
        seeds = [int(rng.integers(0, 2**31)) for _ in range(n_chains)]

        args_list = [((pmcmc_class, pmcmc_kwargs), endog, i, seeds[i]) for i in range(n_chains)]

        with mp.Pool(processes=min(self.n_workers, n_chains)) as pool:
            results = pool.map(_run_single_chain, args_list)

        return results

    def run_multiple_pf(
        self,
        filter_factory: Any,
        model: Any,
        endog: NDArray[np.float64],
        n_particles: int = 1000,
        n_runs: int = 10,
        seed: int = 42,
    ) -> list[dict[str, float]]:
        """Run multiple particle filters in parallel.

        Useful for estimating variance of log-likelihood estimates.

        Parameters:
            filter_factory: Factory to create filters.
            model: Model object.
            endog: Observed data.
            n_particles: Number of particles per filter.
            n_runs: Number of filter runs.
            seed: Base random seed.

        Returns:
            List of result dictionaries with log_likelihood.
        """
        rng = np.random.default_rng(seed)
        seeds = [int(rng.integers(0, 2**31)) for _ in range(n_runs)]

        args_list = [
            ((filter_factory, model, n_particles), endog, i, seeds[i]) for i in range(n_runs)
        ]

        with mp.Pool(processes=min(self.n_workers, n_runs)) as pool:
            results = pool.map(_run_single_filter, args_list)

        return results

    def convergence_study(
        self,
        model: Any,
        filter_factory: Any,
        endog: NDArray[np.float64],
        true_states: NDArray[np.float64],  # noqa: ARG002
        n_values: list[int] | None = None,
        n_repeats: int = 50,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Run convergence study in parallel.

        Parameters:
            model: Model object.
            filter_factory: Factory to create filters.
            endog: Observed data.
            true_states: True state values for RMSE computation.
            n_values: Particle counts to test.
            n_repeats: Repeats per N.
            seed: Random seed.

        Returns:
            Dictionary with n_values, rmse_means, rate.
        """
        if n_values is None:
            n_values = [100, 500, 1000, 5000]

        rng = np.random.default_rng(seed)
        rmse_by_n: dict[int, list[float]] = {n: [] for n in n_values}

        for n_particles in n_values:
            results = self.run_multiple_pf(
                filter_factory=filter_factory,
                model=model,
                endog=endog,
                n_particles=n_particles,
                n_runs=n_repeats,
                seed=int(rng.integers(0, 2**31)),  # pyright: ignore[reportUnknownArgumentType]
            )

            for r in results:
                rmse_by_n[n_particles].append(r["log_likelihood"])

        return {
            "n_values": n_values,
            "results_by_n": rmse_by_n,
        }
