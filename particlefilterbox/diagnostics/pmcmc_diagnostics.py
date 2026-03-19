"""PMCMC diagnostics: trace, ACF, ESS, R-hat, Geweke.

Provides comprehensive diagnostics for evaluating convergence and
mixing of MCMC chains produced by Particle MCMC methods.

References:
    Gelman, A. & Rubin, D.B. (1992). Inference from iterative simulation
    using multiple sequences. Statistical Science, 7(4), 457-472.

    Geyer, C.J. (1992). Practical Markov chain Monte Carlo. Statistical
    Science, 7(4), 473-483.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class PMCMCDiagnostics:
    """Diagnostics for PMCMC chains.

    Analyzes one or more MCMC chains for convergence, mixing, and
    effective sample size.

    Parameters:
        chains: Array of shape (M, L, D) for M chains of length L with D parameters,
            or (L, D) for a single chain, or (L,) for a single parameter.

    Examples:
        >>> chains = np.random.randn(4, 1000, 3)  # 4 chains, 1000 iters, 3 params
        >>> diag = PMCMCDiagnostics(chains)
        >>> print(f"R-hat: {diag.r_hat()}")
        >>> print(f"ESS: {diag.ess()}")
        >>> assert diag.is_converged()
    """

    def __init__(self, chains: NDArray[np.float64]) -> None:
        arr = np.asarray(chains, dtype=np.float64)

        if arr.ndim == 1:
            # Single parameter, single chain: (L,) -> (1, L, 1)
            arr = arr[np.newaxis, :, np.newaxis]
        elif arr.ndim == 2:
            # Single chain: (L, D) -> (1, L, D)
            arr = arr[np.newaxis, :, :]
        elif arr.ndim == 3:
            pass  # (M, L, D) - already correct
        else:
            raise ValueError(f"chains must be 1D, 2D, or 3D, got {arr.ndim}D")

        self._chains = arr  # (M, L, D)
        self._n_chains, self._chain_length, self._n_params = arr.shape

    @property
    def n_chains(self) -> int:
        """Number of chains."""
        return self._n_chains

    @property
    def chain_length(self) -> int:
        """Length of each chain."""
        return self._chain_length

    @property
    def n_params(self) -> int:
        """Number of parameters."""
        return self._n_params

    def trace(self, param: int = 0, chain: int | None = None) -> NDArray[np.float64]:
        """Get trace of a parameter.

        Parameters:
            param: Parameter index.
            chain: Chain index. If None, returns all chains stacked.

        Returns:
            Array of parameter values along the chain(s).
        """
        if chain is not None:
            return self._chains[chain, :, param].copy()
        return self._chains[:, :, param].copy()  # (M, L)

    def acf(
        self,
        param: int = 0,
        chain: int = 0,
        max_lag: int = 50,
    ) -> NDArray[np.float64]:
        """Compute autocorrelation function.

        Parameters:
            param: Parameter index.
            chain: Chain index.
            max_lag: Maximum lag to compute.

        Returns:
            Array of autocorrelation values from lag 0 to max_lag.
        """
        x = self._chains[chain, :, param]
        n = len(x)
        max_lag = min(max_lag, n - 1)
        x_centered = x - np.mean(x)
        var = np.var(x)

        if var == 0:
            return np.ones(max_lag + 1, dtype=np.float64)

        acf_values = np.zeros(max_lag + 1, dtype=np.float64)
        for k in range(max_lag + 1):
            acf_values[k] = np.mean(x_centered[: n - k] * x_centered[k:]) / var

        return acf_values

    def ess(self, param: int | None = None) -> float | NDArray[np.float64]:
        """Compute effective sample size using autocorrelation.

        ESS = L / (1 + 2 * sum_{k=1}^K rho_k)

        The sum is truncated at the first negative autocorrelation.

        Parameters:
            param: Parameter index. If None, returns ESS for all parameters.

        Returns:
            ESS value(s).
        """
        if param is None:
            return np.array([self.ess(p) for p in range(self._n_params)], dtype=np.float64)

        ess_per_chain: list[float] = []
        for m in range(self._n_chains):
            acf_vals = self.acf(param=param, chain=m, max_lag=min(500, self._chain_length // 2))

            # Sum until first negative autocorrelation
            total = 0.0
            for k in range(1, len(acf_vals)):
                if acf_vals[k] < 0:
                    break
                total += acf_vals[k]

            chain_ess = self._chain_length / (1.0 + 2.0 * total)
            ess_per_chain.append(max(1.0, chain_ess))

        return float(np.sum(ess_per_chain))

    def r_hat(self, param: int | None = None) -> float | NDArray[np.float64]:
        """Compute Gelman-Rubin R-hat statistic.

        Requires multiple chains (M >= 2). R-hat < 1.1 indicates convergence.

        Parameters:
            param: Parameter index. If None, returns R-hat for all parameters.

        Returns:
            R-hat value(s).

        Raises:
            ValueError: If only one chain is available.
        """
        if param is None:
            return np.array([self.r_hat(p) for p in range(self._n_params)], dtype=np.float64)

        if self._n_chains < 2:
            raise ValueError("R-hat requires at least 2 chains.")

        n_l = self._chain_length

        # Chain means
        chain_means = np.mean(self._chains[:, :, param], axis=1)

        # Between-chain variance
        b_var = n_l * np.var(chain_means, ddof=1)

        # Within-chain variance
        chain_vars = np.var(self._chains[:, :, param], axis=1, ddof=1)
        w_var: float = float(np.mean(np.asarray(chain_vars, dtype=np.float64)))

        if w_var == 0:
            return 1.0

        # Pooled variance estimate
        var_hat: float = ((n_l - 1) / n_l) * w_var + (1.0 / n_l) * float(b_var)

        r_hat_val = float(np.sqrt(var_hat / w_var))
        return r_hat_val

    def geweke(
        self,
        param: int = 0,
        chain: int = 0,
        first: float = 0.1,
        last: float = 0.5,
    ) -> dict[str, float]:
        """Geweke convergence diagnostic.

        Compares the mean of the first portion of the chain with the mean
        of the last portion using a z-test. |z| < 2 suggests convergence.

        Parameters:
            param: Parameter index.
            chain: Chain index.
            first: Fraction of chain for the first segment (default 0.1).
            last: Fraction of chain for the last segment (default 0.5).

        Returns:
            Dictionary with z_score, p_value, and converged flag.
        """
        from scipy import stats  # type: ignore[reportMissingTypeStubs]

        x = self._chains[chain, :, param]
        n = len(x)

        n_first = max(1, int(first * n))
        n_last = max(1, int(last * n))

        x_first = x[:n_first]
        x_last = x[n - n_last :]

        mean_first = np.mean(x_first)
        mean_last = np.mean(x_last)
        var_first = np.var(x_first, ddof=1) / n_first
        var_last = np.var(x_last, ddof=1) / n_last

        se = np.sqrt(var_first + var_last)
        z = 0.0 if se == 0 else float((mean_first - mean_last) / se)

        # Two-tailed p-value from standard normal
        cdf_val: float = float(stats.norm.cdf(abs(z)))  # type: ignore[reportUnknownMemberType]
        p_value = float(2.0 * (1.0 - cdf_val))

        return {
            "z_score": z,
            "p_value": p_value,
            "converged": abs(z) < 2.0,
        }

    def acceptance_rate(self, chain: int = 0) -> float:
        """Estimate acceptance rate from a chain.

        Counts fraction of consecutive values that differ.

        Parameters:
            chain: Chain index.

        Returns:
            Estimated acceptance rate.
        """
        n_accepted = int(np.sum(np.any(np.diff(self._chains[chain], axis=0) != 0, axis=1)))
        return float(n_accepted / (self._chain_length - 1))

    def is_converged(
        self,
        r_hat_threshold: float = 1.1,
        ess_threshold: float = 100.0,
    ) -> bool:
        """Check if chains have converged.

        Parameters:
            r_hat_threshold: Maximum acceptable R-hat.
            ess_threshold: Minimum acceptable ESS per parameter.

        Returns:
            True if all convergence criteria are met.
        """
        # Check ESS
        ess_vals = self.ess()
        if isinstance(ess_vals, np.ndarray):
            if np.any(ess_vals < ess_threshold):
                return False
        elif ess_vals < ess_threshold:
            return False

        # Check R-hat if multiple chains
        if self._n_chains >= 2:
            r_hat_vals = self.r_hat()
            if isinstance(r_hat_vals, np.ndarray):
                if np.any(r_hat_vals > r_hat_threshold):
                    return False
            elif r_hat_vals > r_hat_threshold:
                return False

        return True

    def summary(self) -> dict[str, Any]:
        """Generate comprehensive diagnostics summary.

        Returns:
            Dictionary with all diagnostic statistics.
        """
        result: dict[str, Any] = {
            "n_chains": self._n_chains,
            "chain_length": self._chain_length,
            "n_params": self._n_params,
        }

        # ESS per parameter
        ess_vals = self.ess()
        if isinstance(ess_vals, np.ndarray):
            result["ess"] = ess_vals.tolist()
            result["ess_min"] = float(np.min(ess_vals))
        else:
            result["ess"] = [ess_vals]
            result["ess_min"] = ess_vals

        # R-hat if multiple chains
        if self._n_chains >= 2:
            r_hat_vals = self.r_hat()
            if isinstance(r_hat_vals, np.ndarray):
                result["r_hat"] = r_hat_vals.tolist()
                result["r_hat_max"] = float(np.max(r_hat_vals))
            else:
                result["r_hat"] = [r_hat_vals]
                result["r_hat_max"] = r_hat_vals

        # Acceptance rate per chain
        acc_rates = [self.acceptance_rate(m) for m in range(self._n_chains)]
        result["acceptance_rates"] = acc_rates
        result["acceptance_rate_mean"] = float(np.mean(acc_rates))

        # Geweke for each param/chain
        geweke_results: list[dict[str, float]] = []
        for p in range(self._n_params):
            for m in range(self._n_chains):
                g = self.geweke(param=p, chain=m)
                geweke_results.append(g)
        result["geweke_all_converged"] = all(g["converged"] for g in geweke_results)

        result["is_converged"] = self.is_converged()

        return result
