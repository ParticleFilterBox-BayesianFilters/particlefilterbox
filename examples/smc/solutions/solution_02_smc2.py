#!/usr/bin/env python
"""Solution 02: SMC^2 for Joint State-Parameter Estimation in Stochastic Volatility.

Runs SMC^2 on the stochastic volatility model, estimating parameters (mu, phi, sigma_h)
and latent log-volatility states jointly. Saves results to results_smc2.csv.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from particlefilterbox.smc import SMCSquared

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
N_THETA = 500
N_X = 200
N_MCMC_MOVES = 3
T_OBS = 200  # Use first 200 observations

TRUE_MU = -1.0
TRUE_PHI = 0.97
TRUE_SIGMA_H = 0.15

PARAM_NAMES = ["mu", "phi", "sigma_h"]
TRUE_VALUES = [TRUE_MU, TRUE_PHI, TRUE_SIGMA_H]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    data_path = Path(__file__).resolve().parent.parent / "data" / "simulated_sv.csv"
    df = pd.read_csv(data_path)
    y_obs: NDArray[np.float64] = df["y_obs"].values[:T_OBS]
    h_true: NDArray[np.float64] = df["h_true"].values[:T_OBS]
    print(f"Loaded {len(y_obs)} observations from {data_path.name}")
    return y_obs, h_true


# ---------------------------------------------------------------------------
# Stochastic Volatility Model (StateSpaceModel protocol)
# ---------------------------------------------------------------------------
class SVModel:
    """Stochastic Volatility model for SMC^2.

    theta = [mu, phi, sigma_h]
    """

    def initial_state_sample(
        self, theta: NDArray[np.float64], n_particles: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        mu, phi, sigma_h = theta[0], theta[1], theta[2]
        if abs(phi) < 0.999:
            std = sigma_h / np.sqrt(1.0 - phi**2)
        else:
            std = sigma_h * 5.0
        h0 = rng.normal(mu, std, size=(n_particles, 1))
        return h0

    def transition_sample(
        self, x_prev: NDArray[np.float64], theta: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        mu, phi, sigma_h = theta[0], theta[1], theta[2]
        noise = rng.normal(0, sigma_h, size=x_prev.shape)
        h_new = mu + phi * (x_prev - mu) + noise
        return h_new

    def observation_logpdf(
        self, y_t: NDArray[np.float64], x_t: NDArray[np.float64], theta: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        h = x_t[:, 0]
        scale = np.exp(h / 2.0)
        log_pdf = -0.5 * np.log(2 * np.pi) - h / 2.0 - 0.5 * (y_t[0] / scale) ** 2
        return log_pdf


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------
class SVPrior:
    """Prior for SV model parameters.

    mu ~ N(-1, 1), phi ~ Beta(20, 1.5), sigma_h ~ HalfNormal(0.5)
    """

    def logpdf(self, theta: NDArray[np.float64]) -> float:
        mu, phi, sigma_h = theta[0], theta[1], theta[2]
        if phi <= 0 or phi >= 1 or sigma_h <= 0:
            return -np.inf
        lp = stats.norm.logpdf(mu, loc=-1.0, scale=1.0)
        lp += stats.beta.logpdf(phi, a=20, b=1.5)
        lp += stats.halfnorm.logpdf(sigma_h, scale=0.5)
        return float(lp)

    def sample(self, rng: np.random.Generator) -> NDArray[np.float64]:
        mu = rng.normal(-1.0, 1.0)
        phi = rng.beta(20, 1.5)
        sigma_h = abs(rng.normal(0, 0.5))
        if sigma_h < 0.01:
            sigma_h = 0.01
        return np.array([mu, phi, sigma_h])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    y_obs, h_true = load_data()

    sv_prior = SVPrior()

    # Verify prior at true params
    theta_true = np.array(TRUE_VALUES)
    print(f"Prior logpdf at true params: {sv_prior.logpdf(theta_true):.4f}")

    # Run SMC^2
    print(f"\nRunning SMC^2 (N_theta={N_THETA}, N_x={N_X}, MCMC moves={N_MCMC_MOVES})...")
    smc2 = SMCSquared(
        model_class=SVModel,
        n_theta=N_THETA,
        n_x=N_X,
        prior=sv_prior,
        n_mcmc_moves=N_MCMC_MOVES,
        ess_threshold=0.5,
        seed=SEED,
    )

    endog = y_obs.reshape(-1, 1)

    t0 = time.time()
    results = smc2.run(endog=endog)
    elapsed = time.time() - t0

    results.param_names = PARAM_NAMES

    print(f"SMC^2 completed in {elapsed:.1f}s")
    print(f"Time steps processed: {results.n_steps}")
    print(f"Log marginal likelihood: {results.log_evidence:.4f}")
    print(f"Rejuvenation events: {len(results.acceptance_rates)}")
    if results.acceptance_rates:
        print(f"Mean acceptance rate: {np.mean(results.acceptance_rates):.3f}")

    # Posterior summaries
    means = results.posterior_mean()
    stds = results.posterior_std()
    ci = results.credible_interval(0.95)

    print("\nSMC^2 Parameter Estimates vs True Values")
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
            "smc2_mean": means[j],
            "smc2_std": stds[j],
            "ci_lower": ci[j, 0],
            "ci_upper": ci[j, 1],
            "within_2sd": within_2sd,
        })

    # Add metadata rows
    rows.append({
        "parameter": "log_evidence",
        "true_value": np.nan,
        "smc2_mean": results.log_evidence,
        "smc2_std": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "within_2sd": np.isfinite(results.log_evidence),
    })
    rows.append({
        "parameter": "n_rejuvenations",
        "true_value": np.nan,
        "smc2_mean": len(results.acceptance_rates),
        "smc2_std": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "within_2sd": True,
    })
    rows.append({
        "parameter": "elapsed_seconds",
        "true_value": np.nan,
        "smc2_mean": elapsed,
        "smc2_std": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "within_2sd": True,
    })

    # Save CSV
    out_path = Path(__file__).resolve().parent / "results_smc2.csv"
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # Validate: parameters within reasonable intervals
    print("\nValidation:")
    for j, (name, true_val) in enumerate(zip(PARAM_NAMES, TRUE_VALUES)):
        within = ci[j, 0] <= true_val <= ci[j, 1]
        status = "PASS" if within else "WARN (true outside 95% CI)"
        print(f"  {name}: {status}")

    print(f"\nLog marginal likelihood: {results.log_evidence:.4f}")
    assert np.isfinite(results.log_evidence), "Log-evidence is not finite!"
    print("PASSED: All checks completed.")


if __name__ == "__main__":
    main()
