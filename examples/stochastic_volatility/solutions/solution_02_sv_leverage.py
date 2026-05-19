"""Solution script for Stochastic Volatility with Leverage.

Pipeline:
    1. Load simulated_sv_leverage.csv (true rho = -0.5).
    2. Filter with a Bootstrap PF accounting for leverage.
    3. Estimate posterior of (mu, phi, sigma, rho) via PMMH.
    4. Save filtered volatility + parameters to results_sv_leverage.csv.

Run:
    python solution_02_sv_leverage.py
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
T_USE = 200
N_PF = 1000
N_PMMH_ITER = 4000
N_PMMH_PARTICLES = 300
PMMH_BURNIN = 1000

TRUE_PARAMS = {"mu": -1.0, "phi": 0.97, "sigma": 0.15, "rho": -0.5}


def bootstrap_pf_sv_leverage(
    y: np.ndarray,
    mu: float,
    phi: float,
    sigma: float,
    rho: float = 0.0,
    n_particles: int = 1000,
    seed: int = 42,
) -> dict:
    """Bootstrap PF for SV with optional leverage correlation rho."""
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
    var_cond = max(1.0 - rho**2, 1e-10)
    etas = np.zeros(n_particles)

    for t in range(T):
        vol = np.exp(particles / 2.0)
        eps = y[t] / vol
        if t == 0 or abs(rho) < 1e-10:
            log_w = -0.5 * np.log(2 * np.pi) - np.log(vol) - 0.5 * eps**2
        else:
            mean_cond = rho * etas
            log_w = (
                -0.5 * np.log(2 * np.pi * var_cond)
                - 0.5 * (eps - mean_cond) ** 2 / var_cond
                - np.log(vol)
            )
        max_lw = float(np.max(log_w))
        w = np.exp(log_w - max_lw)
        sw = float(np.sum(w))
        if sw < 1e-300:
            return {
                "means": filtered_means,
                "stds": filtered_stds,
                "q05": filtered_q05,
                "q95": filtered_q95,
                "ess": ess_history,
                "log_likelihood": -np.inf,
            }
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
        etas = rng.standard_normal(n_particles)
        particles = mu + phi * (particles - mu) + sigma * etas

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


class SVLeverageWrapper:
    """Wrapper for PGAS with full state (h, eta_prev)."""

    def __init__(self) -> None:
        self.sv = StochasticVolatility(variant="leverage")
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
        rho = self.sv.params.get("rho", 0.0)
        if abs(phi) >= 1.0 or sigma <= 0:
            return FilterResult(log_likelihood=-np.inf)
        var_stat = sigma**2 / (1.0 - phi**2)
        particles = rng.normal(mu, np.sqrt(max(var_stat, 1e-10)), size=n_particles)
        log_lik = 0.0
        var_cond = max(1.0 - rho**2, 1e-10)
        etas = np.zeros(n_particles)
        filtered_h = np.zeros(len(endog))
        filtered_eta = np.zeros(len(endog))
        for t in range(len(endog)):
            vol = np.exp(particles / 2.0)
            eps = endog[t] / vol
            if t == 0:
                log_w = -0.5 * np.log(2 * np.pi) - np.log(vol) - 0.5 * eps**2
            else:
                mean_cond = rho * etas
                log_w = (
                    -0.5 * np.log(2 * np.pi * var_cond)
                    - 0.5 * (eps - mean_cond) ** 2 / var_cond
                    - np.log(vol)
                )
            max_lw = float(np.max(log_w))
            w = np.exp(log_w - max_lw)
            sw = float(np.sum(w))
            if sw < 1e-300:
                return FilterResult(log_likelihood=-np.inf)
            log_lik += max_lw + np.log(sw) - np.log(n_particles)
            w /= sw
            filtered_h[t] = float(np.sum(w * particles))
            filtered_eta[t] = float(np.sum(w * etas))
            idx = rng.choice(n_particles, size=n_particles, p=w)
            particles = particles[idx]
            etas = rng.standard_normal(n_particles)
            particles = mu + phi * (particles - mu) + sigma * etas
        return FilterResult(
            log_likelihood=log_lik,
            filtered_means=np.column_stack([filtered_h, filtered_eta]),
        )

    def initial_sample(self, n: int, rng) -> np.ndarray:
        mu = self.sv.params["mu"]
        phi = self.sv.params["phi"]
        sigma = self.sv.params["sigma"]
        if abs(phi) >= 1.0 or sigma <= 0:
            return np.column_stack([rng.standard_normal(n), np.zeros(n)])
        var_stat = sigma**2 / (1.0 - phi**2)
        h_init = rng.normal(mu, np.sqrt(max(var_stat, 1e-10)), size=n)
        return np.column_stack([h_init, np.zeros(n)])

    def transition_sample(self, x_prev, rng) -> np.ndarray:
        mu = self.sv.params["mu"]
        phi = self.sv.params["phi"]
        sigma = self.sv.params["sigma"]
        h_prev = float(x_prev[0]) if hasattr(x_prev, "__getitem__") else float(x_prev)
        eta = float(rng.standard_normal())
        h_new = mu + phi * (h_prev - mu) + sigma * eta
        return np.array([h_new, eta])

    def transition_logpdf(self, x_new, x_old) -> float:
        mu = self.sv.params["mu"]
        phi = self.sv.params["phi"]
        sigma = self.sv.params["sigma"]
        if sigma <= 0:
            return -np.inf
        h_new = float(x_new[0]) if hasattr(x_new, "__getitem__") else float(x_new)
        h_old = float(x_old[0]) if hasattr(x_old, "__getitem__") else float(x_old)
        return float(stats.norm.logpdf(h_new, loc=mu + phi * (h_old - mu), scale=sigma))

    def observation_logpdf(self, y, x) -> float:
        h = float(x[0]) if hasattr(x, "__getitem__") else float(x)
        eta = float(x[1]) if (hasattr(x, "__getitem__") and len(x) > 1) else 0.0
        rho = self.sv.params.get("rho", 0.0)
        vol = np.exp(h / 2.0)
        if vol < 1e-300:
            return -np.inf
        eps = float(y) / vol
        var_cond = max(1.0 - rho**2, 1e-10)
        mean_cond = rho * eta
        return float(
            -0.5 * np.log(2 * np.pi * var_cond)
            - 0.5 * (eps - mean_cond) ** 2 / var_cond
            - np.log(vol)
        )


class SVLeveragePrior:
    """Prior for (mu, phi, sigma, rho)."""

    def logpdf(self, theta: np.ndarray) -> float:
        mu_val = float(theta[0])
        phi_val = float(theta[1])
        sig_val = float(theta[2])
        rho_val = float(theta[3])
        if not (0.0 < phi_val < 1.0) or sig_val <= 0.0 or not (-1.0 < rho_val < 1.0):
            return -np.inf
        lp = float(stats.norm.logpdf(mu_val, loc=-1.0, scale=1.0))
        lp += float(stats.beta.logpdf(phi_val, 20.0, 1.5))
        lp += float(stats.invgamma.logpdf(sig_val, a=2.5, scale=0.025))
        lp += float(stats.uniform.logpdf(rho_val, loc=-1.0, scale=2.0))
        return lp

    def sample(self, rng) -> np.ndarray:
        mu_s = float(rng.normal(-1.0, 1.0))
        phi_s = float(stats.beta.rvs(20.0, 1.5, random_state=rng))
        sig_s = float(stats.invgamma.rvs(2.5, scale=0.025, random_state=rng))
        rho_s = float(rng.uniform(-1.0, 1.0))
        return np.array([mu_s, phi_s, sig_s, rho_s])

    @property
    def cov(self) -> np.ndarray:
        return np.array([1.0, 20 * 1.5 / ((21.5) ** 2 * 22.5), 0.025**2 / (1.5**2 * 0.5), 1.0 / 3.0])


def _cond_loglik(endog: np.ndarray, states: np.ndarray, theta: np.ndarray) -> float:
    mu = float(theta[0])
    phi = float(theta[1])
    sigma = float(theta[2])
    rho = float(theta[3])
    if abs(phi) >= 1.0 or sigma <= 0 or abs(rho) >= 1.0:
        return -np.inf
    T = len(endog)
    loglik = 0.0
    var_cond = max(1.0 - rho**2, 1e-10)
    if states.ndim == 2:
        h_vals = states[:, 0]
        eta_vals = states[:, 1]
    else:
        h_vals = states
        eta_vals = np.zeros(T)
    var_stat = sigma**2 / (1.0 - phi**2)
    loglik += float(stats.norm.logpdf(h_vals[0], loc=mu, scale=np.sqrt(max(var_stat, 1e-10))))
    for t in range(T):
        h_t = float(h_vals[t])
        vol = np.exp(h_t / 2.0)
        if vol < 1e-300:
            return -np.inf
        eps = endog[t] / vol
        if t == 0:
            loglik += -0.5 * np.log(2 * np.pi) - np.log(vol) - 0.5 * eps**2
        else:
            mean_cond = rho * eta_vals[t]
            loglik += (
                -0.5 * np.log(2 * np.pi * var_cond)
                - 0.5 * (eps - mean_cond) ** 2 / var_cond
                - np.log(vol)
            )
        if t < T - 1:
            loglik += float(
                stats.norm.logpdf(h_vals[t + 1], loc=mu + phi * (h_t - mu), scale=sigma)
            )
    return loglik


def sv_leverage_param_sampler(
    model, states: np.ndarray, endog: np.ndarray, theta_current: np.ndarray, rng
) -> np.ndarray:
    """Metropolis-within-Gibbs sampler for (mu, phi, sigma, rho)."""
    prior = SVLeveragePrior()
    step_sizes = np.array([0.05, 0.01, 0.01, 0.05])
    theta = theta_current.copy()
    for _ in range(3):
        for j in range(len(theta)):
            theta_prop = theta.copy()
            theta_prop[j] += step_sizes[j] * rng.standard_normal()
            lp_prop = prior.logpdf(theta_prop)
            if not np.isfinite(lp_prop):
                continue
            lp_curr = prior.logpdf(theta)
            ll_curr = _cond_loglik(endog, states, theta)
            ll_prop = _cond_loglik(endog, states, theta_prop)
            log_alpha = (lp_prop + ll_prop) - (lp_curr + ll_curr)
            if np.log(rng.random()) < log_alpha:
                theta = theta_prop
    return theta


def main() -> None:
    print("=" * 70)
    print("SOLUTION 02: Stochastic Volatility with Leverage")
    print("=" * 70)

    data_path = os.path.join(SCRIPT_DIR, "..", "data", "simulated_sv_leverage.csv")
    df = pd.read_csv(data_path)
    y_obs = np.asarray(df["y_obs"].to_numpy(dtype=float))[:T_USE]
    h_true = np.asarray(df["h_true"].to_numpy(dtype=float))[:T_USE]
    t_idx = np.arange(T_USE)
    print(f"Loaded {len(y_obs)} observations (T_USE={T_USE}).")
    print(f"True parameters: {TRUE_PARAMS}")

    print(f"\n[1/3] Bootstrap PF with leverage (N={N_PF}, seed={SEED}) ...")
    t0 = time.time()
    pf_basic = bootstrap_pf_sv_leverage(
        y_obs,
        mu=TRUE_PARAMS["mu"],
        phi=TRUE_PARAMS["phi"],
        sigma=TRUE_PARAMS["sigma"],
        rho=0.0,
        n_particles=N_PF,
        seed=SEED,
    )
    pf_lev = bootstrap_pf_sv_leverage(
        y_obs,
        mu=TRUE_PARAMS["mu"],
        phi=TRUE_PARAMS["phi"],
        sigma=TRUE_PARAMS["sigma"],
        rho=TRUE_PARAMS["rho"],
        n_particles=N_PF,
        seed=SEED,
    )
    print(f"  done in {time.time() - t0:.2f}s")
    print(f"  log-lik SV basic (rho=0):     {pf_basic['log_likelihood']:.2f}")
    print(f"  log-lik SV-L  (rho={TRUE_PARAMS['rho']:+.2f}): {pf_lev['log_likelihood']:.2f}")
    print(f"  delta:                        {pf_lev['log_likelihood'] - pf_basic['log_likelihood']:+.2f}")

    print(f"\n[2/3] PMMH (iters={N_PMMH_ITER}, N={N_PMMH_PARTICLES}, burnin={PMMH_BURNIN}) ...")
    model_lev = SVLeverageWrapper()
    prior_lev = SVLeveragePrior()
    theta_init = np.array([
        TRUE_PARAMS["mu"],
        TRUE_PARAMS["phi"],
        TRUE_PARAMS["sigma"],
        TRUE_PARAMS["rho"],
    ])
    pmmh = PMMH(
        model=model_lev,
        prior=prior_lev,
        n_particles=N_PMMH_PARTICLES,
        n_iterations=N_PMMH_ITER,
        proposal_cov="adaptive",
        target_acceptance=0.234,
        burnin=PMMH_BURNIN,
        thin=1,
        seed=SEED,
    )
    t0 = time.time()
    pmmh_results = pmmh.run(endog=y_obs, theta_init=theta_init, verbose=500)
    print(f"  done in {time.time() - t0:.1f}s, acceptance={pmmh_results.acceptance_rate():.3f}")

    post_mean = pmmh_results.posterior_mean()
    post_std = pmmh_results.posterior_std()
    ci_low, ci_high = pmmh_results.credible_interval(alpha=0.05)
    mu_hat, phi_hat, sig_hat, rho_hat = (float(v) for v in post_mean)
    print(f"  posterior means: mu={mu_hat:.4f}, phi={phi_hat:.4f}, sigma={sig_hat:.4f}, rho={rho_hat:.4f}")

    samples = pmmh_results.posterior_samples
    rho_samples = samples[:, 3]
    p_rho_neg = float((rho_samples < 0).mean())
    print(f"  P(rho < 0 | data) = {p_rho_neg:.3f}")
    if rho_hat >= 0:
        print("  WARNING: posterior mean of rho is non-negative (expected negative for leverage).")
    else:
        print("  posterior mean of rho is negative — leverage effect detected.")

    print(f"\n[3/3] Re-filter with posterior mean params ...")
    pf_post = bootstrap_pf_sv_leverage(
        y_obs,
        mu=mu_hat,
        phi=phi_hat,
        sigma=sig_hat,
        rho=rho_hat,
        n_particles=N_PF,
        seed=SEED + 7,
    )
    rmse_h = float(np.sqrt(np.mean((pf_post["means"] - h_true) ** 2)))
    print(f"  log-lik at posterior mean: {pf_post['log_likelihood']:.2f}")
    print(f"  RMSE of filtered h_t vs true: {rmse_h:.4f}")

    print("\nWriting results_sv_leverage.csv ...")
    h_mean = pf_post["means"]
    h_q05 = pf_post["q05"]
    h_q95 = pf_post["q95"]
    filtered_rows = pd.DataFrame({
        "section": "filtered",
        "t": t_idx,
        "y_obs": y_obs,
        "h_true": h_true,
        "h_mean": h_mean,
        "h_q05": h_q05,
        "h_q95": h_q95,
        "vol_mean": np.exp(h_mean / 2.0),
        "vol_q05": np.exp(h_q05 / 2.0),
        "vol_q95": np.exp(h_q95 / 2.0),
        "ess": pf_post["ess"],
    })

    param_names = ["mu", "phi", "sigma", "rho"]
    params_rows = pd.DataFrame({
        "section": "parameters",
        "t": np.nan,
        "y_obs": np.nan,
        "h_true": np.nan,
        "h_mean": np.nan,
        "h_q05": np.nan,
        "h_q95": np.nan,
        "vol_mean": np.nan,
        "vol_q05": np.nan,
        "vol_q95": np.nan,
        "ess": np.nan,
        "parameter": param_names,
        "true_value": [TRUE_PARAMS[n] for n in param_names],
        "post_mean": post_mean,
        "post_std": post_std,
        "ci_low": ci_low,
        "ci_high": ci_high,
    })

    diag_rows = pd.DataFrame({
        "section": "diagnostics",
        "t": np.nan,
        "y_obs": np.nan,
        "h_true": np.nan,
        "h_mean": np.nan,
        "h_q05": np.nan,
        "h_q95": np.nan,
        "vol_mean": np.nan,
        "vol_q05": np.nan,
        "vol_q95": np.nan,
        "ess": np.nan,
        "parameter": ["ll_basic", "ll_leverage", "ll_posterior", "rmse_h", "p_rho_neg"],
        "value": [
            pf_basic["log_likelihood"],
            pf_lev["log_likelihood"],
            pf_post["log_likelihood"],
            rmse_h,
            p_rho_neg,
        ],
    })

    combined = pd.concat([filtered_rows, params_rows, diag_rows], ignore_index=True, sort=False)
    out_path = os.path.join(SCRIPT_DIR, "results_sv_leverage.csv")
    combined.to_csv(out_path, index=False)
    print(f"  saved {len(combined)} rows to {out_path}")

    print("\n--- Parameter Summary ---")
    summary = pd.DataFrame({
        "parameter": param_names,
        "true": [TRUE_PARAMS[n] for n in param_names],
        "post_mean": post_mean,
        "post_std": post_std,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "covers_true": [ci_low[i] <= TRUE_PARAMS[n] <= ci_high[i] for i, n in enumerate(param_names)],
    })
    print(summary.to_string(index=False))
    print("Done.")


if __name__ == "__main__":
    main()
