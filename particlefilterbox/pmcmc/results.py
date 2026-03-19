"""Results container for Particle MCMC methods.

Provides storage and diagnostics for MCMC chains produced by PMCMC samplers,
including posterior summaries, convergence diagnostics (ESS, R-hat, Geweke),
and autocorrelation analysis.

References:
    Gelman, A. & Rubin, D. B. (1992). Inference from iterative simulation
    using multiple sequences. Statistical Science, 7(4), 457-472.
    Geweke, J. (1992). Evaluating the accuracy of sampling-based approaches
    to calculating posterior moments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["PMCMCResults"]


@dataclass
class PMCMCResults:
    """Container for PMCMC inference results with diagnostics.

    Stores the full MCMC chain and provides methods for posterior summaries,
    convergence diagnostics, and chain analysis.

    Parameters
    ----------
    chains : NDArray[np.float64]
        MCMC chain of shape ``(n_iterations, k_params)``.
    param_names : list[str] | None
        Names of parameters. If None, uses ``['param_0', 'param_1', ...]``.
    log_likelihood_chain : NDArray[np.float64]
        Log-likelihood values at each iteration, shape ``(n_iterations,)``.
    acceptance_history : NDArray[np.bool_]
        Boolean array of acceptance decisions, shape ``(n_iterations,)``.
    burnin : int
        Number of burn-in iterations to discard.
    thin : int
        Thinning factor.
    """

    chains: NDArray[np.float64]
    param_names: list[str] | None = None
    log_likelihood_chain: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    acceptance_history: NDArray[np.bool_] = field(default_factory=lambda: np.array([], dtype=bool))
    burnin: int = 0
    thin: int = 1

    def __post_init__(self) -> None:
        """Validate and set defaults."""
        if self.param_names is None:
            k = self.chains.shape[1] if self.chains.ndim > 1 else 1
            self.param_names = [f"param_{i}" for i in range(k)]

    @property
    def posterior_samples(self) -> NDArray[np.float64]:
        """Return post-burnin, thinned samples.

        Returns
        -------
        NDArray[np.float64]
            Posterior samples of shape ``(n_effective, k_params)``.
        """
        return self.chains[self.burnin :: self.thin]

    @property
    def n_params(self) -> int:
        """Number of parameters."""
        return self.chains.shape[1] if self.chains.ndim > 1 else 1

    @property
    def n_iterations(self) -> int:
        """Total number of iterations."""
        return self.chains.shape[0]

    @property
    def n_effective_samples(self) -> int:
        """Number of post-burnin, thinned samples."""
        return len(self.posterior_samples)

    def summary(self) -> str:
        """Generate formatted summary of posterior inference.

        Returns
        -------
        str
            Multi-line summary string with posterior statistics.
        """
        samples = self.posterior_samples
        lines: list[str] = []
        lines.append("=" * 70)
        lines.append("PMCMC Posterior Summary")
        lines.append("=" * 70)
        lines.append(
            f"{'Parameter':<15} {'Mean':>10} {'Std':>10} "
            f"{'2.5%':>10} {'50%':>10} {'97.5%':>10} {'ESS':>8}"
        )
        lines.append("-" * 70)

        assert self.param_names is not None
        for j, name in enumerate(self.param_names):
            col = samples[:, j] if samples.ndim > 1 else samples
            mean = np.mean(col)
            std = np.std(col, ddof=1)
            q025 = np.percentile(col, 2.5)
            q50 = np.percentile(col, 50.0)
            q975 = np.percentile(col, 97.5)
            ess = self.effective_sample_size(j)
            lines.append(
                f"{name:<15} {mean:>10.4f} {std:>10.4f} "
                f"{q025:>10.4f} {q50:>10.4f} {q975:>10.4f} {ess:>8.1f}"
            )

        lines.append("-" * 70)
        lines.append(f"Total iterations: {self.n_iterations}")
        lines.append(f"Burn-in: {self.burnin}")
        lines.append(f"Thinning: {self.thin}")
        lines.append(f"Effective samples: {self.n_effective_samples}")
        lines.append(f"Acceptance rate: {self.acceptance_rate():.4f}")
        lines.append("=" * 70)

        return "\n".join(lines)

    def posterior_mean(self) -> NDArray[np.float64]:
        """Compute posterior mean for each parameter.

        Returns
        -------
        NDArray[np.float64]
            Posterior means of shape ``(k_params,)``.
        """
        return np.mean(self.posterior_samples, axis=0)

    def posterior_std(self) -> NDArray[np.float64]:
        """Compute posterior standard deviation for each parameter.

        Returns
        -------
        NDArray[np.float64]
            Posterior standard deviations of shape ``(k_params,)``.
        """
        return np.std(self.posterior_samples, axis=0, ddof=1)

    def credible_interval(
        self, alpha: float = 0.05
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute equal-tailed credible interval.

        Parameters
        ----------
        alpha : float
            Significance level. Default 0.05 gives 95% interval.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]]
            Lower and upper bounds of shape ``(k_params,)``.
        """
        samples = self.posterior_samples
        lower = np.percentile(samples, 100 * alpha / 2, axis=0)
        upper = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)
        return lower, upper

    def acceptance_rate(self) -> float:
        """Compute overall acceptance rate.

        Returns
        -------
        float
            Fraction of proposals that were accepted.
        """
        if len(self.acceptance_history) == 0:
            return 0.0
        return float(np.mean(self.acceptance_history))

    def effective_sample_size(self, param_idx: int = 0) -> float:
        """Compute effective sample size using autocorrelation.

        Uses the formula ESS = L / (1 + 2 * sum(rho_k)) where rho_k is the
        autocorrelation at lag k, summed until the first negative value
        (initial monotone sequence estimator).

        Parameters
        ----------
        param_idx : int
            Index of the parameter. Default 0.

        Returns
        -------
        float
            Effective sample size.
        """
        samples = self.posterior_samples
        x = samples[:, param_idx] if samples.ndim > 1 else samples

        n = len(x)
        if n < 4:
            return float(n)

        # Compute autocorrelations using FFT
        x_centered = x - np.mean(x)
        var = np.var(x, ddof=0)
        if var < 1e-15:
            return float(n)

        # FFT-based autocorrelation
        fft_x = np.fft.fft(x_centered, n=2 * n)
        acf_full = np.real(np.fft.ifft(fft_x * np.conj(fft_x)))[:n]
        acf_full = acf_full / acf_full[0]

        # Sum autocorrelations until first negative (Geyer's initial positive
        # sequence estimator)
        tau = 0.0
        for k in range(1, n):
            if acf_full[k] < 0:
                break
            tau += acf_full[k]

        ess = n / (1.0 + 2.0 * tau)
        return max(1.0, ess)

    def trace_plot_data(
        self, param_idx: int = 0
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return data for trace plot.

        Parameters
        ----------
        param_idx : int
            Index of the parameter.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]]
            Iteration indices and parameter values.
        """
        values = self.chains[:, param_idx] if self.chains.ndim > 1 else self.chains
        iterations = np.arange(len(values), dtype=np.float64)
        return iterations, values

    def acf(self, param_idx: int = 0, max_lag: int = 50) -> NDArray[np.float64]:
        """Compute autocorrelation function of the chain.

        Parameters
        ----------
        param_idx : int
            Index of the parameter.
        max_lag : int
            Maximum lag to compute. Default 50.

        Returns
        -------
        NDArray[np.float64]
            Autocorrelation values of shape ``(max_lag + 1,)``.
        """
        samples = self.posterior_samples
        x = samples[:, param_idx] if samples.ndim > 1 else samples

        n = len(x)
        max_lag = min(max_lag, n - 1)

        x_centered = x - np.mean(x)
        var = np.var(x, ddof=0)
        if var < 1e-15:
            result = np.zeros(max_lag + 1)
            result[0] = 1.0
            return result

        # FFT-based autocorrelation
        fft_x = np.fft.fft(x_centered, n=2 * n)
        acf_full = np.real(np.fft.ifft(fft_x * np.conj(fft_x)))[:n]
        acf_full = acf_full / acf_full[0]

        return acf_full[: max_lag + 1]

    def r_hat(
        self,
        other_chains: list[PMCMCResults] | None = None,
        param_idx: int = 0,
    ) -> float:
        """Compute Gelman-Rubin R-hat convergence diagnostic.

        Requires multiple chains. If ``other_chains`` is provided, computes
        R-hat across all chains. Otherwise, splits the current chain in half.

        Parameters
        ----------
        other_chains : list[PMCMCResults] | None
            Additional chain results for multi-chain R-hat.
        param_idx : int
            Index of the parameter.

        Returns
        -------
        float
            R-hat statistic. Values close to 1.0 indicate convergence.
        """
        if other_chains is not None and len(other_chains) > 0:
            # Multi-chain R-hat
            all_samples = [self.posterior_samples]
            for ch in other_chains:
                all_samples.append(ch.posterior_samples)

            chains_list: list[NDArray[np.float64]] = []
            for s in all_samples:
                if s.ndim > 1:
                    chains_list.append(s[:, param_idx])
                else:
                    chains_list.append(s)
        else:
            # Split chain in half
            samples = self.posterior_samples
            x = samples[:, param_idx] if samples.ndim > 1 else samples
            mid = len(x) // 2
            chains_list = [x[:mid], x[mid:]]

        m = len(chains_list)
        n = min(len(c) for c in chains_list)

        if n < 2 or m < 2:
            return float("nan")

        # Trim chains to same length
        chains_list = [c[:n] for c in chains_list]

        # Between-chain variance
        chain_means = np.array([np.mean(c) for c in chains_list])
        overall_mean = np.mean(chain_means)
        b_var = n / (m - 1) * np.sum((chain_means - overall_mean) ** 2)

        # Within-chain variance
        chain_vars = np.array([np.var(c, ddof=1) for c in chains_list])
        w_var = np.mean(chain_vars)

        if w_var < 1e-15:
            return 1.0

        # Pooled variance estimate
        var_hat = (1 - 1 / n) * w_var + (1 / n) * b_var

        r_hat = np.sqrt(var_hat / w_var)
        return float(r_hat)

    def geweke_test(
        self,
        param_idx: int = 0,
        first_frac: float = 0.1,
        last_frac: float = 0.5,
    ) -> tuple[float, float]:
        """Geweke convergence diagnostic.

        Compares the mean of the first portion of the chain to the mean of
        the last portion using a z-test.

        Parameters
        ----------
        param_idx : int
            Index of the parameter.
        first_frac : float
            Fraction of chain for first segment. Default 0.1.
        last_frac : float
            Fraction of chain for last segment. Default 0.5.

        Returns
        -------
        tuple[float, float]
            (z-score, p-value). Large p-values indicate convergence.
        """
        from scipy import stats  # pyright: ignore[reportMissingTypeStubs]

        samples = self.posterior_samples
        x = samples[:, param_idx] if samples.ndim > 1 else samples

        n = len(x)
        n_first = max(int(n * first_frac), 1)
        n_last = max(int(n * last_frac), 1)

        first = x[:n_first]
        last = x[n - n_last :]

        mean_first = np.mean(first)
        mean_last = np.mean(last)

        # Spectral density at frequency 0 using batch means to account
        # for autocorrelation in MCMC chains
        var_first = self._spectral_variance_of_mean(first)
        var_last = self._spectral_variance_of_mean(last)

        denom = np.sqrt(var_first + var_last)
        if denom < 1e-15:
            return 0.0, 1.0

        z = (mean_first - mean_last) / denom
        p_value = 2 * (1 - float(stats.norm.cdf(abs(z))))  # pyright: ignore[reportUnknownMemberType]

        return float(z), float(p_value)

    @staticmethod
    def _spectral_variance_of_mean(x: NDArray[np.float64]) -> float:
        """Estimate variance of the sample mean using batch means.

        Accounts for autocorrelation in MCMC chains by dividing the chain
        into batches and computing the variance of batch means.

        Parameters
        ----------
        x : NDArray[np.float64]
            Chain segment.

        Returns
        -------
        float
            Estimated variance of the sample mean.
        """
        n = len(x)
        if n < 2:
            return float(np.var(x, ddof=0))

        # Use autocorrelation-adjusted variance: Var(mean) = var/n * (1 + 2*sum(rho_k))
        x_centered = x - np.mean(x)
        var = float(np.var(x, ddof=1))
        if var < 1e-15:
            return 0.0

        # FFT-based autocorrelation
        fft_x = np.fft.fft(x_centered, n=2 * n)
        acf_full = np.real(np.fft.ifft(fft_x * np.conj(fft_x)))[:n]
        acf_full = acf_full / acf_full[0]

        # Sum autocorrelations until first negative (Geyer's IMSE)
        tau = 0.0
        for k in range(1, n):
            if acf_full[k] < 0:
                break
            tau += acf_full[k]

        return var / n * (1.0 + 2.0 * tau)

    def to_dataframe(self) -> Any:
        """Convert posterior samples to a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns for each parameter.
        """
        import pandas as pd

        samples = self.posterior_samples
        assert self.param_names is not None

        if samples.ndim == 1:
            return pd.DataFrame({self.param_names[0]: samples})

        return pd.DataFrame(samples, columns=self.param_names)
