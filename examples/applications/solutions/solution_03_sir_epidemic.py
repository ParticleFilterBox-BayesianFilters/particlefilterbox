"""Solution script for Notebook 3: Stochastic SIR epidemic with partial obs.

Runs the Bootstrap particle filter on simulated SIR data with Poisson
reported infectives. Writes filtered S, I, R means, 90% credible band
for I, posterior of R_0 = beta / gamma and epidemic metrics to
``results_sir_epidemic.csv``.

Usage
-----
    python solution_03_sir_epidemic.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import poisson

from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel
from particlefilterbox.filters import BootstrapPF

SEED: int = 20260422
N_PARTICLES: int = 5000
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_PATH = Path(__file__).resolve().parent / "results_sir_epidemic.csv"


class StochasticSIR(ParticleFilterModel):
    """Stochastic SIR with Poisson reported infectives.

    State vector x = (S, I, R, beta, gamma).  beta and gamma are carried
    in the state so that R_0 = beta / gamma has a particle representation
    at every time step.
    """

    k_states = 5
    k_obs = 1

    def __init__(
        self,
        N: int = 10000,
        beta: float = 0.3,
        gamma: float = 0.1,
        obs_rate: float = 0.5,
        beta_prior: tuple[float, float] = (0.30, 0.05),
        gamma_prior: tuple[float, float] = (0.10, 0.02),
    ) -> None:
        self.N = int(N)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.obs_rate = float(obs_rate)
        self.beta_prior = beta_prior
        self.gamma_prior = gamma_prior

    @property
    def params(self) -> dict[str, float]:
        return {
            "N": float(self.N),
            "beta": self.beta,
            "gamma": self.gamma,
            "obs_rate": self.obs_rate,
        }

    def initial_distribution(
        self,
        n_particles: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        s0 = np.full(n_particles, self.N - 10, dtype=float)
        i0 = np.full(n_particles, 10.0)
        r0 = np.zeros(n_particles)
        beta = np.clip(rng.normal(*self.beta_prior, size=n_particles), 0.01, None)
        gamma = np.clip(rng.normal(*self.gamma_prior, size=n_particles), 0.01, None)
        return np.column_stack([s0, i0, r0, beta, gamma])

    def transition(
        self,
        particles: NDArray[np.float64],
        t: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        s = particles[:, 0]
        i = particles[:, 1]
        r = particles[:, 2]
        beta = particles[:, 3]
        gamma = particles[:, 4]
        lam_inf = np.clip(beta * s * i / self.N, 0.0, None)
        lam_rec = np.clip(gamma * i, 0.0, None)
        new_inf = rng.poisson(lam_inf).astype(float)
        new_rec = rng.poisson(lam_rec).astype(float)
        new_inf = np.minimum(new_inf, s)
        new_rec = np.minimum(new_rec, i)
        s_new = s - new_inf
        i_new = i + new_inf - new_rec
        r_new = r + new_rec
        return np.column_stack([s_new, i_new, r_new, beta, gamma])

    def log_observation_likelihood(
        self,
        particles: NDArray[np.float64],
        y_t: NDArray[np.float64],
        t: int,
    ) -> NDArray[np.float64]:
        y = float(np.asarray(y_t).ravel()[0])
        i = particles[:, 1]
        rate = np.clip(self.obs_rate * i, 1e-6, None)
        return poisson.logpmf(y, mu=rate)


def weighted_quantile(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    q: float,
) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    total = w.sum()
    if total <= 0.0:
        return float(np.nan)
    cw = np.cumsum(w) / total
    return float(np.interp(q, cw, v))


def main() -> None:
    df = pd.read_csv(DATA_DIR / "simulated_sir.csv")
    y_obs = df["y_obs"].to_numpy(dtype=float)
    s_true = df["S_true"].to_numpy(dtype=float)
    i_true = df["I_true"].to_numpy(dtype=float)
    r_true = df["R_true"].to_numpy(dtype=float)

    model = StochasticSIR()
    config = PFConfig(
        n_particles=N_PARTICLES,
        resampling="systematic",
        ess_threshold=0.5,
        seed=SEED,
    )

    T = len(y_obs)
    s_hat = np.zeros(T)
    i_hat = np.zeros(T)
    r_hat = np.zeros(T)
    i_q05 = np.zeros(T)
    i_q50 = np.zeros(T)
    i_q95 = np.zeros(T)
    r0_mean = np.zeros(T)
    r0_q05 = np.zeros(T)
    r0_q95 = np.zeros(T)
    ess_hist = np.zeros(T)
    ll_inc = np.zeros(T)

    t_start = time.perf_counter()

    pf = BootstrapPF(model=model, config=config)
    rng = pf._get_rng()  # noqa: SLF001 - reuse the seeded generator for online mode
    cloud = pf.initialize(rng)

    total_ll = 0.0
    for t in range(T):
        cloud, ll_t = pf.filter_step(cloud, np.array([y_obs[t]]), t)
        total_ll += float(ll_t)
        ll_inc[t] = float(ll_t)

        w = cloud.normalized_weights
        particles = cloud.particles
        s_part = particles[:, 0]
        i_part = particles[:, 1]
        r_part = particles[:, 2]
        beta_part = particles[:, 3]
        gamma_part = particles[:, 4]
        r0_part = beta_part / gamma_part

        s_hat[t] = float(np.sum(w * s_part))
        i_hat[t] = float(np.sum(w * i_part))
        r_hat[t] = float(np.sum(w * r_part))
        i_q05[t] = weighted_quantile(i_part, w, 0.05)
        i_q50[t] = weighted_quantile(i_part, w, 0.50)
        i_q95[t] = weighted_quantile(i_part, w, 0.95)
        r0_mean[t] = float(np.sum(w * r0_part))
        r0_q05[t] = weighted_quantile(r0_part, w, 0.05)
        r0_q95[t] = weighted_quantile(r0_part, w, 0.95)
        ess_hist[t] = float(1.0 / np.sum(w ** 2))

    elapsed = time.perf_counter() - t_start

    out = pd.DataFrame({
        "t": df["t"].to_numpy(),
        "y_obs": y_obs,
        "S_true": s_true,
        "I_true": i_true,
        "R_true": r_true,
        "S_filt": s_hat,
        "I_filt": i_hat,
        "R_filt": r_hat,
        "I_q05": i_q05,
        "I_q50": i_q50,
        "I_q95": i_q95,
        "R0_mean": r0_mean,
        "R0_q05": r0_q05,
        "R0_q95": r0_q95,
        "ess": ess_hist,
        "log_lik_inc": ll_inc,
    })
    out.to_csv(OUT_PATH, index=False)

    rmse_i = float(np.sqrt(np.mean((i_hat - i_true) ** 2)))
    rmse_s = float(np.sqrt(np.mean((s_hat - s_true) ** 2)))
    rmse_r = float(np.sqrt(np.mean((r_hat - r_true) ** 2)))
    true_r0 = model.beta / model.gamma

    print("=" * 64)
    print("SIR particle filter - solution_03_sir_epidemic.py")
    print("=" * 64)
    print(f"observations        : {T}")
    print(f"particles           : {N_PARTICLES}")
    print(f"log-likelihood      : {total_ll:.2f}")
    print(f"mean ESS            : {ess_hist.mean():.1f}")
    print(f"elapsed             : {elapsed:.2f} s")
    print("-" * 64)
    print(f"RMSE S              : {rmse_s:.2f}")
    print(f"RMSE I              : {rmse_i:.2f}")
    print(f"RMSE R              : {rmse_r:.2f}")
    print("-" * 64)
    print(f"true R0             : {true_r0:.3f}")
    print(f"posterior mean R0 (final) : {r0_mean[-1]:.3f}")
    print(f"90% R0 CI (final)   : [{r0_q05[-1]:.3f}, {r0_q95[-1]:.3f}]")
    print(f"CSV written to      : {OUT_PATH}")


if __name__ == "__main__":
    main()
