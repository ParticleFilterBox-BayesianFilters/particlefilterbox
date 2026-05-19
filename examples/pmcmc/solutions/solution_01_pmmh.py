"""Solution 01 - PMMH on Stochastic Volatility model.

Runs Particle Marginal Metropolis-Hastings on the basic SV model,
saves posterior chain and summary to results_pmmh.csv.

Usage:
    python solution_01_pmmh.py          # quick mode (default)
    python solution_01_pmmh.py --full   # full run with more iterations
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from particlefilterbox.models.stochastic_volatility import StochasticVolatility
from particlefilterbox.pmcmc import PMMH


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
    """Minimal particle filter result."""
    log_likelihood: float
    filtered_means: np.ndarray | None = None


class SVModelWrapper:
    """Adapt StochasticVolatility to the PMCMC interface."""

    def __init__(self, variant: str = "basic") -> None:
        self.sv = StochasticVolatility(variant=variant)
        self.variant = variant
        self.param_names = list(self.sv.param_names)
        self._rng_internal = np.random.default_rng(0)

    def set_params(self, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=np.float64)
        for i, name in enumerate(self.param_names):
            self.sv.params[name] = float(theta[i])

    def get_params(self) -> np.ndarray:
        return np.array([self.sv.params[n] for n in self.param_names])

    def filter(
        self,
        endog: np.ndarray,
        n_particles: int = 200,
        rng: np.random.Generator | None = None,
    ) -> FilterResult:
        if rng is None:
            rng = self._rng_internal
        T = len(endog)
        mu = self.sv.params["mu"]
        phi = self.sv.params["phi"]
        sigma = self.sv.params["sigma"]

        if abs(phi) >= 1.0 or sigma <= 0:
            return FilterResult(log_likelihood=-np.inf)

        var_stat = sigma**2 / (1.0 - phi**2)
        particles = rng.normal(mu, np.sqrt(max(var_stat, 1e-10)), size=n_particles)
        log_lik = 0.0
        filtered_means_arr = np.zeros(T)

        for t in range(T):
            vol = np.exp(particles / 2.0)
            log_w = -0.5 * np.log(2 * np.pi) - np.log(vol) - 0.5 * (endog[t] / vol) ** 2
            max_lw = np.max(log_w)
            w = np.exp(log_w - max_lw)
            sw = np.sum(w)
            if sw < 1e-300:
                return FilterResult(log_likelihood=-np.inf)
            log_lik += max_lw + np.log(sw) - np.log(n_particles)
            w /= sw
            filtered_means_arr[t] = np.sum(w * particles)
            idx = rng.choice(n_particles, size=n_particles, p=w)
            particles = particles[idx]
            particles = mu + phi * (particles - mu) + sigma * rng.standard_normal(n_particles)

        return FilterResult(log_likelihood=log_lik, filtered_means=filtered_means_arr)

    def initial_sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        mu = self.sv.params["mu"]
        phi = self.sv.params["phi"]
        sigma = self.sv.params["sigma"]
        if abs(phi) >= 1.0 or sigma <= 0:
            return rng.standard_normal(n)
        var_stat = sigma**2 / (1.0 - phi**2)
        return rng.normal(mu, np.sqrt(max(var_stat, 1e-10)), size=n)

    def transition_sample(self, x_prev: float | np.ndarray, rng: np.random.Generator) -> float:
        mu = self.sv.params["mu"]
        phi = self.sv.params["phi"]
        sigma = self.sv.params["sigma"]
        return float(mu + phi * (float(x_prev) - mu) + sigma * rng.standard_normal())

    def transition_logpdf(self, x_new: float | np.ndarray, x_old: float | np.ndarray) -> float:
        mu = self.sv.params["mu"]
        phi = self.sv.params["phi"]
        sigma = self.sv.params["sigma"]
        if sigma <= 0:
            return -np.inf
        mean = mu + phi * (float(x_old) - mu)
        return float(stats.norm.logpdf(float(x_new), loc=mean, scale=sigma))

    def observation_logpdf(self, y: float, x: float | np.ndarray) -> float:
        h = float(x)
        vol = np.exp(h / 2.0)
        if vol < 1e-300:
            return -np.inf
        return float(-0.5 * np.log(2 * np.pi) - np.log(vol) - 0.5 * (float(y) / vol) ** 2)


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------

class SVPrior:
    """Prior: mu~N(-1,1), phi~Beta(20,1.5), sigma~InvGamma(2.5,0.025)."""

    def __init__(self) -> None:
        self.loc_mu = -1.0
        self.scale_mu = 1.0
        self.a_phi = 20.0
        self.b_phi = 1.5
        self.a_sig = 2.5
        self.b_sig = 0.025

    def logpdf(self, theta: np.ndarray) -> float:
        mu_val, phi_val, sig_val = theta[0], theta[1], theta[2]
        if not (0 < phi_val < 1) or sig_val <= 0:
            return -np.inf
        lp = float(stats.norm.logpdf(mu_val, loc=self.loc_mu, scale=self.scale_mu))
        lp += float(stats.beta.logpdf(phi_val, self.a_phi, self.b_phi))
        lp += float(stats.invgamma.logpdf(sig_val, a=self.a_sig, scale=self.b_sig))
        return lp

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        mu_s = rng.normal(self.loc_mu, self.scale_mu)
        phi_s = float(stats.beta.rvs(self.a_phi, self.b_phi, random_state=rng))
        sig_s = float(stats.invgamma.rvs(self.a_sig, scale=self.b_sig, random_state=rng))
        return np.array([mu_s, phi_s, sig_s])

    @property
    def cov(self) -> np.ndarray:
        c = np.zeros(3)
        c[0] = self.scale_mu ** 2
        a, b = self.a_phi, self.b_phi
        c[1] = (a * b) / ((a + b) ** 2 * (a + b + 1))
        c[2] = self.b_sig ** 2 / ((self.a_sig - 1) ** 2 * (self.a_sig - 2))
        return c


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PMMH on SV model")
    parser.add_argument("--full", action="store_true", help="Full run with more iterations")
    args = parser.parse_args()

    # Configuration
    seed = 42
    true_params = np.array([-1.0, 0.97, 0.15])

    if args.full:
        n_particles = 500
        n_iterations = 5000
        burnin = 1000
        T_use = 200
    else:
        n_particles = 200
        n_iterations = 2000
        burnin = 500
        T_use = 200

    # Load data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "simulated_sv.csv")
    df = pd.read_csv(data_path)
    y_obs = df["y_obs"].values[:T_use]
    print(f"Loaded {len(df)} observations, using first {T_use}")
    print(f"True parameters: mu={true_params[0]}, phi={true_params[1]}, sigma_h={true_params[2]}")

    # Set up model and prior
    model = SVModelWrapper(variant="basic")
    prior = SVPrior()

    # Run PMMH
    pmmh = PMMH(
        model=model,
        prior=prior,
        n_particles=n_particles,
        n_iterations=n_iterations,
        proposal_cov="adaptive",
        target_acceptance=0.234,
        burnin=burnin,
        thin=1,
        seed=seed,
    )

    print(f"\nRunning PMMH (N={n_particles}, {n_iterations} iterations, burn-in={burnin})...")
    t0 = time.time()
    results = pmmh.run(endog=y_obs, theta_init=true_params, verbose=500)
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")

    # Diagnostics
    acc_rate = results.acceptance_rate()
    print(f"Acceptance rate: {acc_rate:.3f}")
    print(results.summary())

    post_mean = results.posterior_mean()
    post_std = results.posterior_std()
    print(f"\nPosterior means: mu={post_mean[0]:.4f}, phi={post_mean[1]:.4f}, sigma_h={post_mean[2]:.4f}")
    print(f"True values:     mu={true_params[0]:.4f}, phi={true_params[1]:.4f}, sigma_h={true_params[2]:.4f}")

    # Verify acceptance rate
    assert 0.05 <= acc_rate <= 0.50, f"Acceptance rate {acc_rate:.3f} outside expected range [0.05, 0.50]"

    # Save posterior samples to CSV
    samples = results.posterior_samples
    param_names = ["mu", "phi", "sigma_h"]
    df_out = pd.DataFrame(samples, columns=param_names)
    out_path = os.path.join(os.path.dirname(__file__), "results_pmmh.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved {len(df_out)} posterior samples to {out_path}")

    # Summary row
    summary_path = out_path  # samples are the full output
    print(f"\n--- Summary ---")
    print(f"  Posterior mean  : {dict(zip(param_names, post_mean))}")
    print(f"  Posterior std   : {dict(zip(param_names, post_std))}")
    print(f"  Acceptance rate : {acc_rate:.3f}")
    for i, name in enumerate(param_names):
        ess = results.effective_sample_size(i)
        print(f"  ESS({name:>7s})   : {ess:.1f}")
    print("Done.")


if __name__ == "__main__":
    main()
