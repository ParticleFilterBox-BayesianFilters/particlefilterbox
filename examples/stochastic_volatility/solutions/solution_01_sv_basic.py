"""Solution script for Basic Stochastic Volatility on simulated S&P 500 returns.

Pipeline:
    1. Load sp500_returns.csv.
    2. Filter latent log-volatility with a Bootstrap Particle Filter.
    3. Estimate posterior of (mu, phi, sigma_h) via PMMH.
    4. Forecast log-volatility H steps ahead.
    5. Save filtered volatility + parameters to results_sv_basic.csv.

Run:
    python solution_01_sv_basic.py
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from particlefilterbox.models.stochastic_volatility import StochasticVolatility
from particlefilterbox.pmcmc import PMMH

SEED = 42
T_USE = 500
N_PF = 1000
N_PMMH_ITER = 2000
N_PMMH_PARTICLES = 200
PMMH_BURNIN = 500
FORECAST_H = 30


def bootstrap_pf_sv(
    y: np.ndarray,
    mu: float,
    phi: float,
    sigma: float,
    n_particles: int = 1000,
    seed: int = 42,
) -> dict:
    """Run bootstrap PF on basic SV model. Returns filtered means, bands, log-lik."""
    rng = np.random.default_rng(seed)
    T = len(y)

    var_stat = sigma**2 / (1.0 - phi**2)
    particles = rng.normal(mu, np.sqrt(max(var_stat, 1e-10)), size=n_particles)

    filtered_means = np.zeros(T)
    filtered_stds = np.zeros(T)
    filtered_q05 = np.zeros(T)
    filtered_q95 = np.zeros(T)
    ess_history = np.zeros(T)
    log_lik = 0.0

    for t in range(T):
        vol = np.exp(particles / 2.0)
        log_w = -0.5 * np.log(2 * np.pi) - np.log(vol) - 0.5 * (y[t] / vol) ** 2
        max_lw = float(np.max(log_w))
        w = np.exp(log_w - max_lw)
        sw = float(np.sum(w))
        log_lik += max_lw + np.log(sw) - np.log(n_particles)
        w /= sw

        filtered_means[t] = float(np.sum(w * particles))
        filtered_stds[t] = float(np.sqrt(np.sum(w * (particles - filtered_means[t]) ** 2)))
        idx_sort = np.argsort(particles)
        cumw = np.cumsum(w[idx_sort])
        filtered_q05[t] = float(particles[idx_sort][np.searchsorted(cumw, 0.05)])
        q95_idx = int(np.clip(np.searchsorted(cumw, 0.95, side="right"), 0, n_particles - 1))
        filtered_q95[t] = float(particles[idx_sort][q95_idx])
        ess_history[t] = 1.0 / float(np.sum(w**2))

        idx = rng.choice(n_particles, size=n_particles, p=w)
        particles = particles[idx]
        particles = mu + phi * (particles - mu) + sigma * rng.standard_normal(n_particles)

    return {
        "means": filtered_means,
        "stds": filtered_stds,
        "q05": filtered_q05,
        "q95": filtered_q95,
        "ess": ess_history,
        "log_likelihood": log_lik,
    }


@dataclass
class FilterResult:
    log_likelihood: float
    filtered_means: np.ndarray | None = None


class SVBasicWrapper:
    """Model wrapper for PMMH on basic SV."""

    def __init__(self) -> None:
        self.sv = StochasticVolatility(variant="basic")
        self.param_names = list(self.sv.param_names)
        self._rng_internal = np.random.default_rng(0)

    def set_params(self, theta: np.ndarray) -> None:
        for i, name in enumerate(self.param_names):
            self.sv.params[name] = float(theta[i])

    def get_params(self) -> np.ndarray:
        return np.array([self.sv.params[n] for n in self.param_names])

    def filter(self, endog: np.ndarray, n_particles: int = 200, rng=None) -> FilterResult:
        if rng is None:
            rng = self._rng_internal
        mu = self.sv.params["mu"]
        phi = self.sv.params["phi"]
        sigma = self.sv.params["sigma"]
        if abs(phi) >= 1.0 or sigma <= 0:
            return FilterResult(log_likelihood=-np.inf)
        var_stat = sigma**2 / (1.0 - phi**2)
        particles = rng.normal(mu, np.sqrt(max(var_stat, 1e-10)), size=n_particles)
        log_lik = 0.0
        for t in range(len(endog)):
            vol = np.exp(particles / 2.0)
            log_w = -0.5 * np.log(2 * np.pi) - np.log(vol) - 0.5 * (endog[t] / vol) ** 2
            max_lw = float(np.max(log_w))
            w = np.exp(log_w - max_lw)
            sw = float(np.sum(w))
            if sw < 1e-300:
                return FilterResult(log_likelihood=-np.inf)
            log_lik += max_lw + np.log(sw) - np.log(n_particles)
            w /= sw
            idx = rng.choice(n_particles, size=n_particles, p=w)
            particles = particles[idx]
            particles = mu + phi * (particles - mu) + sigma * rng.standard_normal(n_particles)
        return FilterResult(log_likelihood=log_lik)


class SVBasicPrior:
    """Prior for (mu, phi, sigma)."""

    def logpdf(self, theta: np.ndarray) -> float:
        mu_val, phi_val, sig_val = float(theta[0]), float(theta[1]), float(theta[2])
        if not (0.0 < phi_val < 1.0) or sig_val <= 0.0:
            return -np.inf
        lp = float(stats.norm.logpdf(mu_val, loc=-1.0, scale=1.0))
        lp += float(stats.beta.logpdf(phi_val, 20.0, 1.5))
        lp += float(stats.invgamma.logpdf(sig_val, a=2.5, scale=0.025))
        return lp

    def sample(self, rng) -> np.ndarray:
        mu_s = float(rng.normal(-1.0, 1.0))
        phi_s = float(stats.beta.rvs(20.0, 1.5, random_state=rng))
        sig_s = float(stats.invgamma.rvs(2.5, scale=0.025, random_state=rng))
        return np.array([mu_s, phi_s, sig_s])

    @property
    def cov(self) -> np.ndarray:
        return np.array([1.0, 20 * 1.5 / ((21.5) ** 2 * 22.5), 0.025**2 / (1.5**2 * 0.5)])


def forecast_volatility(
    y: np.ndarray,
    mu: float,
    phi: float,
    sigma: float,
    H: int,
    n_particles: int = 2000,
    seed: int = 123,
) -> dict:
    """Re-run filter, then propagate particles H steps without observations."""
    rng = np.random.default_rng(seed)
    var_stat = sigma**2 / (1.0 - phi**2)
    particles = rng.normal(mu, np.sqrt(max(var_stat, 1e-10)), size=n_particles)
    for t in range(len(y)):
        vol = np.exp(particles / 2.0)
        log_w = -0.5 * np.log(2 * np.pi) - np.log(vol) - 0.5 * (y[t] / vol) ** 2
        max_lw = float(np.max(log_w))
        w = np.exp(log_w - max_lw)
        w /= float(np.sum(w))
        idx = rng.choice(n_particles, size=n_particles, p=w)
        particles = particles[idx]
        particles = mu + phi * (particles - mu) + sigma * rng.standard_normal(n_particles)

    forecast_mean = np.zeros(H)
    forecast_q05 = np.zeros(H)
    forecast_q95 = np.zeros(H)
    ahead = particles.copy()
    for h in range(H):
        ahead = mu + phi * (ahead - mu) + sigma * rng.standard_normal(n_particles)
        forecast_mean[h] = float(np.mean(ahead))
        forecast_q05[h] = float(np.percentile(ahead, 5))
        forecast_q95[h] = float(np.percentile(ahead, 95))

    return {"mean": forecast_mean, "q05": forecast_q05, "q95": forecast_q95}


def main() -> None:
    print("=" * 70)
    print("SOLUTION 01: Basic Stochastic Volatility on S&P 500")
    print("=" * 70)

    data_path = os.path.join(SCRIPT_DIR, "..", "data", "sp500_returns.csv")
    df = pd.read_csv(data_path)
    returns = np.asarray(df["returns"].to_numpy(dtype=float))[:T_USE]
    dates = np.asarray(df["date"].to_numpy())[:T_USE]
    print(f"Loaded {len(returns)} returns from {dates[0]} to {dates[-1]}")
    print(f"  mean={returns.mean():.6f}, std={returns.std():.6f}, kurt={stats.kurtosis(returns):.2f}")

    print(f"\n[1/3] Bootstrap PF (N={N_PF}, seed={SEED}) ...")
    t0 = time.time()
    pf_result = bootstrap_pf_sv(returns, mu=-1.0, phi=0.97, sigma=0.15, n_particles=N_PF, seed=SEED)
    print(f"  done in {time.time() - t0:.2f}s, log-lik={pf_result['log_likelihood']:.2f}, mean ESS={pf_result['ess'].mean():.0f}")

    print(f"\n[2/3] PMMH (iters={N_PMMH_ITER}, N={N_PMMH_PARTICLES}, burnin={PMMH_BURNIN}) ...")
    model_wrapper = SVBasicWrapper()
    prior = SVBasicPrior()
    y_pmmh = returns[:200]
    pmmh = PMMH(
        model=model_wrapper,
        prior=prior,
        n_particles=N_PMMH_PARTICLES,
        n_iterations=N_PMMH_ITER,
        proposal_cov="adaptive",
        target_acceptance=0.234,
        burnin=PMMH_BURNIN,
        thin=1,
        seed=SEED,
    )
    t0 = time.time()
    pmmh_results = pmmh.run(endog=y_pmmh, theta_init=np.array([-1.0, 0.97, 0.15]), verbose=500)
    print(f"  done in {time.time() - t0:.1f}s, acceptance={pmmh_results.acceptance_rate():.3f}")

    post_mean = pmmh_results.posterior_mean()
    post_std = pmmh_results.posterior_std()
    ci_low, ci_high = pmmh_results.credible_interval(alpha=0.05)
    mu_hat, phi_hat, sig_hat = float(post_mean[0]), float(post_mean[1]), float(post_mean[2])
    print(f"  posterior means: mu={mu_hat:.4f}, phi={phi_hat:.4f}, sigma={sig_hat:.4f}")

    print(f"\n[3/3] Forecasting {FORECAST_H} steps ahead ...")
    fc = forecast_volatility(returns, mu_hat, phi_hat, sig_hat, H=FORECAST_H, n_particles=2000, seed=SEED + 1)
    print(f"  forecast mean h: {fc['mean'][0]:.4f} -> {fc['mean'][-1]:.4f}")

    print("\nWriting results_sv_basic.csv ...")
    h_mean = pf_result["means"]
    h_q05 = pf_result["q05"]
    h_q95 = pf_result["q95"]
    filtered_rows = pd.DataFrame({
        "section": "filtered",
        "date": dates,
        "t": np.arange(T_USE),
        "return": returns,
        "h_mean": h_mean,
        "h_q05": h_q05,
        "h_q95": h_q95,
        "vol_mean": np.exp(h_mean / 2.0),
        "vol_q05": np.exp(h_q05 / 2.0),
        "vol_q95": np.exp(h_q95 / 2.0),
        "ess": pf_result["ess"],
    })
    forecast_rows = pd.DataFrame({
        "section": "forecast",
        "date": [""] * FORECAST_H,
        "t": np.arange(T_USE, T_USE + FORECAST_H),
        "return": [np.nan] * FORECAST_H,
        "h_mean": fc["mean"],
        "h_q05": fc["q05"],
        "h_q95": fc["q95"],
        "vol_mean": np.exp(fc["mean"] / 2.0),
        "vol_q05": np.exp(fc["q05"] / 2.0),
        "vol_q95": np.exp(fc["q95"] / 2.0),
        "ess": [np.nan] * FORECAST_H,
    })
    params_rows = pd.DataFrame({
        "section": "parameters",
        "date": "",
        "t": np.nan,
        "return": np.nan,
        "h_mean": np.nan,
        "h_q05": np.nan,
        "h_q95": np.nan,
        "vol_mean": np.nan,
        "vol_q05": np.nan,
        "vol_q95": np.nan,
        "ess": np.nan,
        "parameter": ["mu", "phi", "sigma"],
        "post_mean": post_mean,
        "post_std": post_std,
        "ci_low": ci_low,
        "ci_high": ci_high,
    })
    combined = pd.concat([filtered_rows, forecast_rows, params_rows], ignore_index=True, sort=False)
    out_path = os.path.join(SCRIPT_DIR, "results_sv_basic.csv")
    combined.to_csv(out_path, index=False)
    print(f"  saved {len(combined)} rows to {out_path}")

    print("\n--- Parameter Summary ---")
    summary = pd.DataFrame({
        "parameter": ["mu", "phi", "sigma"],
        "post_mean": post_mean,
        "post_std": post_std,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ess_param": [pmmh_results.effective_sample_size(i) for i in range(3)],
    })
    print(summary.to_string(index=False))
    print(f"\nAcceptance rate: {pmmh_results.acceptance_rate():.3f}")
    print(f"Final log-likelihood at posterior mean: estimated during PMMH run")
    print("Done.")


if __name__ == "__main__":
    main()
