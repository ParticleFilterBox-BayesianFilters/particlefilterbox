"""Solution 02 - Particle Gibbs on Stochastic Volatility model.

Runs Particle Gibbs sampler on the basic SV model,
saves posterior chain and diagnostics to results_particle_gibbs.csv.

Usage:
    python solution_02_particle_gibbs.py          # quick mode (default)
    python solution_02_particle_gibbs.py --full   # full run with more iterations
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from particlefilterbox.models.stochastic_volatility import StochasticVolatility
from particlefilterbox.pmcmc import ParticleGibbs


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
# Parameter sampler (component-wise MH for Gibbs)
# ---------------------------------------------------------------------------

def sv_param_sampler(
    model: SVModelWrapper,
    states: np.ndarray,
    endog: np.ndarray,
    theta_current: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Component-wise MH parameter sampler for the SV model.

    Given states h_{0:T}, samples (mu, phi, sigma) using single-component
    random walk MH updates with tuned step sizes.
    """
    prior = SVPrior()
    # Step sizes tuned for the SV conditional posterior
    step_sizes = np.array([0.05, 0.01, 0.01])
    n_sub_iter = 3  # multiple sweeps per Gibbs iteration

    theta = theta_current.copy()

    for _ in range(n_sub_iter):
        for j in range(len(theta)):
            theta_prop = theta.copy()
            theta_prop[j] += step_sizes[j] * rng.standard_normal()

            # Check prior support
            lp_prop = prior.logpdf(theta_prop)
            if not np.isfinite(lp_prop):
                continue

            lp_curr = prior.logpdf(theta)

            # Compute conditional log-likelihood p(h|theta) + p(y|h)
            ll_curr = _conditional_loglik(model, endog, states, theta)
            ll_prop = _conditional_loglik(model, endog, states, theta_prop)

            log_alpha = (lp_prop + ll_prop) - (lp_curr + ll_curr)
            if np.log(rng.random()) < log_alpha:
                theta = theta_prop

    return theta


def _conditional_loglik(
    model: SVModelWrapper,
    endog: np.ndarray,
    states: np.ndarray,
    theta: np.ndarray,
) -> float:
    """Compute log p(y, h | theta) for basic SV model."""
    mu, phi, sigma = theta[0], theta[1], theta[2]
    if abs(phi) >= 1.0 or sigma <= 0:
        return -np.inf

    T = len(endog)
    loglik = 0.0

    # Initial state distribution: h_0 ~ N(mu, sigma^2/(1-phi^2))
    var_stat = sigma**2 / (1.0 - phi**2)
    loglik += float(stats.norm.logpdf(states[0], loc=mu, scale=np.sqrt(var_stat)))

    for t in range(T):
        h_t = float(states[t])
        # Observation: y_t ~ N(0, exp(h_t))
        vol = np.exp(h_t / 2.0)
        if vol < 1e-300:
            return -np.inf
        loglik += -0.5 * np.log(2 * np.pi) - np.log(vol) - 0.5 * (endog[t] / vol) ** 2

        # Transition: h_{t+1} | h_t ~ N(mu + phi*(h_t - mu), sigma^2)
        if t < T - 1:
            h_next = float(states[t + 1])
            mean_t = mu + phi * (h_t - mu)
            loglik += float(stats.norm.logpdf(h_next, loc=mean_t, scale=sigma))

    return loglik


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Particle Gibbs on SV model")
    parser.add_argument("--full", action="store_true", help="Full run with more iterations")
    args = parser.parse_args()

    # Configuration
    seed = 42
    true_params = np.array([-1.0, 0.97, 0.15])

    if args.full:
        n_particles = 100
        n_iterations = 2000
        burnin = 500
        T_use = 200
    else:
        n_particles = 50
        n_iterations = 1000
        burnin = 300
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

    # Run Particle Gibbs with custom parameter sampler
    pg = ParticleGibbs(
        model=model,
        prior=prior,
        n_particles=n_particles,
        n_iterations=n_iterations,
        param_sampler=sv_param_sampler,
        burnin=burnin,
        thin=1,
        seed=seed,
    )

    print(f"\nRunning Particle Gibbs (N={n_particles}, {n_iterations} iterations, burn-in={burnin})...")
    t0 = time.time()
    results = pg.run(endog=y_obs, theta_init=true_params, verbose=200)
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")

    # Diagnostics
    print(results.summary())

    post_mean = results.posterior_mean()
    post_std = results.posterior_std()
    param_names = ["mu", "phi", "sigma_h"]
    print(f"\nPosterior means: mu={post_mean[0]:.4f}, phi={post_mean[1]:.4f}, sigma_h={post_mean[2]:.4f}")
    print(f"True values:     mu={true_params[0]:.4f}, phi={true_params[1]:.4f}, sigma_h={true_params[2]:.4f}")

    # ESS diagnostics
    print("\n--- ESS Diagnostics ---")
    for i, name in enumerate(param_names):
        ess = results.effective_sample_size(i)
        print(f"  ESS({name:>7s}): {ess:.1f}")

    # Save posterior samples to CSV
    samples = results.posterior_samples
    df_out = pd.DataFrame(samples, columns=param_names)
    out_path = os.path.join(os.path.dirname(__file__), "results_particle_gibbs.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved {len(df_out)} posterior samples to {out_path}")

    # Summary
    print(f"\n--- Summary ---")
    print(f"  Posterior mean  : {dict(zip(param_names, post_mean))}")
    print(f"  Posterior std   : {dict(zip(param_names, post_std))}")
    acc_rate = results.acceptance_rate()
    print(f"  Acceptance rate : {acc_rate:.3f}")
    print("Done.")


if __name__ == "__main__":
    main()
