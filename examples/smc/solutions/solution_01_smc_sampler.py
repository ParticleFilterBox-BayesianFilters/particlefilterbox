#!/usr/bin/env python
"""Solution 01: SMC Sampler for Linear-Gaussian Parameter Estimation.

Runs SMC sampler with adaptive tempering on a linear-Gaussian state-space model,
estimates parameters (phi, sigma_x, sigma_y) and log marginal likelihood.
Saves results to results_smc_sampler.csv.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

# Ensure particlefilterbox is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from particlefilterbox.smc import SMCSampler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
N_PARTICLES = 2000
N_MCMC_MOVES = 5
ESS_TARGET_RATIO = 0.5

TRUE_PHI = 0.95
TRUE_SIGMA_X = 0.5
TRUE_SIGMA_Y = 1.0

PARAM_NAMES = ["phi", "sigma_x", "sigma_y"]
TRUE_VALUES = [TRUE_PHI, TRUE_SIGMA_X, TRUE_SIGMA_Y]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data() -> NDArray[np.float64]:
    data_path = Path(__file__).resolve().parent.parent / "data" / "simulated_linear_gaussian.csv"
    df = pd.read_csv(data_path)
    y_obs: NDArray[np.float64] = df["y_obs"].values
    print(f"Loaded {len(y_obs)} observations from {data_path.name}")
    return y_obs


# ---------------------------------------------------------------------------
# Model: log-likelihood via Kalman filter
# ---------------------------------------------------------------------------
def log_likelihood_lg(theta: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Log-likelihood via Kalman filter for linear-Gaussian model.

    theta = [phi, sigma_x, sigma_y]
    """
    phi, sigma_x, sigma_y = theta[0], theta[1], theta[2]

    if sigma_x <= 0 or sigma_y <= 0 or abs(phi) >= 1:
        return -np.inf

    Q = sigma_x**2
    R = sigma_y**2
    T = len(y)

    x_pred = 0.0
    P_pred = Q / (1.0 - phi**2)

    log_lik = 0.0
    for t in range(T):
        v = y[t] - x_pred
        F = P_pred + R
        log_lik += -0.5 * (np.log(2 * np.pi * F) + v**2 / F)

        K = P_pred / F
        x_filt = x_pred + K * v
        P_filt = (1.0 - K) * P_pred

        x_pred = phi * x_filt
        P_pred = phi**2 * P_filt + Q

    return float(log_lik)


def log_prior(theta: NDArray[np.float64]) -> float:
    """Log prior: phi ~ U(0,1), sigma_x ~ HalfNormal(1), sigma_y ~ HalfNormal(2)."""
    phi, sigma_x, sigma_y = theta[0], theta[1], theta[2]

    if phi <= 0 or phi >= 1 or sigma_x <= 0 or sigma_y <= 0:
        return -np.inf

    lp = 0.0  # Uniform on (0,1) for phi
    lp += stats.halfnorm.logpdf(sigma_x, scale=1.0)
    lp += stats.halfnorm.logpdf(sigma_y, scale=2.0)
    return float(lp)


# Global y_obs for closures
_y_obs: NDArray[np.float64] = np.array([])


def log_target(theta: NDArray[np.float64]) -> float:
    """Log posterior (unnormalized) = log prior + log likelihood."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_lg(theta, _y_obs)


def sample_prior(rng: np.random.Generator) -> NDArray[np.float64]:
    """Draw from the prior."""
    phi = rng.uniform(0.01, 0.99)
    sigma_x = abs(rng.standard_normal()) * 1.0
    sigma_y = abs(rng.standard_normal()) * 2.0
    if sigma_x < 0.01:
        sigma_x = 0.01
    if sigma_y < 0.01:
        sigma_y = 0.01
    return np.array([phi, sigma_x, sigma_y])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    global _y_obs
    _y_obs = load_data()

    # Verify model at true parameters
    theta_true = np.array(TRUE_VALUES)
    ll_true = log_likelihood_lg(theta_true, _y_obs)
    print(f"Log-likelihood at true params: {ll_true:.2f}")
    print(f"Log-prior at true params: {log_prior(theta_true):.4f}")

    # Run SMC Sampler
    print(f"\nRunning SMC Sampler (N={N_PARTICLES}, MCMC moves={N_MCMC_MOVES})...")
    sampler = SMCSampler(
        target_logpdf=log_target,
        prior_logpdf=log_prior,
        prior_sample=sample_prior,
        n_particles=N_PARTICLES,
        n_mcmc_moves=N_MCMC_MOVES,
        ess_target_ratio=ESS_TARGET_RATIO,
        seed=SEED,
    )

    t0 = time.time()
    results = sampler.run()
    elapsed = time.time() - t0

    print(f"SMC Sampler completed in {elapsed:.1f}s")
    print(f"Number of tempering steps: {results.n_steps}")
    print(f"Log marginal likelihood: {results.log_evidence:.4f}")

    # Posterior summaries
    means = results.posterior_mean()
    stds = results.posterior_std()
    ci = results.credible_interval(0.95)

    print("\nParameter Estimates vs True Values")
    print("=" * 70)
    rows = []
    for j, (name, true_val) in enumerate(zip(PARAM_NAMES, TRUE_VALUES)):
        within_2sd = abs(means[j] - true_val) < 2 * stds[j]
        print(
            f"  {name:10s}: true={true_val:.3f}, "
            f"mean={means[j]:.4f}, std={stds[j]:.4f}, "
            f"CI=[{ci[j, 0]:.4f}, {ci[j, 1]:.4f}], "
            f"within_2sd={'Yes' if within_2sd else 'No'}"
        )
        rows.append({
            "parameter": name,
            "true_value": true_val,
            "smc_mean": means[j],
            "smc_std": stds[j],
            "ci_lower": ci[j, 0],
            "ci_upper": ci[j, 1],
            "within_2sd": within_2sd,
        })

    # Add log-evidence and metadata
    meta_row = {
        "parameter": "log_evidence",
        "true_value": np.nan,
        "smc_mean": results.log_evidence,
        "smc_std": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "within_2sd": np.isfinite(results.log_evidence),
    }
    rows.append(meta_row)

    meta_row2 = {
        "parameter": "n_steps",
        "true_value": np.nan,
        "smc_mean": results.n_steps,
        "smc_std": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "within_2sd": True,
    }
    rows.append(meta_row2)

    # Save CSV
    out_path = Path(__file__).resolve().parent / "results_smc_sampler.csv"
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # Verify log-evidence is finite
    assert np.isfinite(results.log_evidence), "Log-evidence is not finite!"
    print(f"\nLog marginal likelihood (evidence): {results.log_evidence:.4f}")
    print("PASSED: Log-evidence is finite.")

    if results.acceptance_rates:
        print(f"Mean MCMC acceptance rate: {np.mean(results.acceptance_rates):.3f}")


if __name__ == "__main__":
    main()
