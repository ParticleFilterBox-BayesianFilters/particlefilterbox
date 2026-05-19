"""Solution: SIR Particle Filter on Stochastic Volatility model.

Runs SIR with multiple particle counts on the SV model.
Saves filtered log-volatility and log-likelihood to CSV.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel
from particlefilterbox.filters.sir import SIR


class SVModel(ParticleFilterModel):
    """Stochastic Volatility model for particle filtering."""

    k_states = 1
    k_obs = 1

    def __init__(self, mu: float = -1.0, phi: float = 0.97, sigma: float = 0.15):
        self.mu = mu
        self.phi = phi
        self.sigma = sigma

    @property
    def params(self) -> dict[str, float]:
        return {"mu": self.mu, "phi": self.phi, "sigma": self.sigma}

    def initial_distribution(self, n_particles: int, rng: np.random.Generator) -> np.ndarray:
        std = self.sigma / np.sqrt(1.0 - self.phi**2)
        return rng.normal(self.mu, std, size=(n_particles, 1))

    def transition(self, particles: np.ndarray, t: int, rng: np.random.Generator) -> np.ndarray:
        h = particles[:, 0]
        noise = rng.standard_normal(len(h))
        h_new = self.mu + self.phi * (h - self.mu) + self.sigma * noise
        return h_new.reshape(-1, 1)

    def log_observation_likelihood(
        self, particles: np.ndarray, y_t: np.ndarray, t: int
    ) -> np.ndarray:
        h = particles[:, 0]
        vol = np.exp(h / 2.0)
        return -0.5 * np.log(2.0 * np.pi) - np.log(vol) - 0.5 * (y_t[0] / vol) ** 2


def main() -> None:
    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    df = pd.read_csv(os.path.join(data_dir, "simulated_sv.csv"))
    y_obs = df["y_obs"].values
    h_true = df["h_true"].values
    T = len(df)

    # SIR with various N
    results: list[dict[str, float]] = []
    N_values = [100, 500, 1000, 5000]

    for N in N_values:
        config = PFConfig(n_particles=N, resampling="systematic", seed=42)
        sir = SIR(SVModel(), config)

        t0 = time.time()
        res = sir.filter(y_obs)
        elapsed = time.time() - t0

        h_filt = res.filtered_means[:, 0]
        rmse = float(np.sqrt(np.mean((h_filt - h_true) ** 2)))
        mean_ess = float(np.mean(res.ess_history))
        log_lik = float(res.log_likelihood)

        results.append({
            "N": N,
            "RMSE": rmse,
            "ESS_mean": mean_ess,
            "ESS_N_pct": mean_ess / N * 100,
            "log_likelihood": log_lik,
            "time_s": elapsed,
        })
        print(
            f"N={N:5d}: RMSE={rmse:.4f}, ESS_mean={mean_ess:.1f}, "
            f"log_lik={log_lik:.2f}, time={elapsed:.2f}s"
        )

    # Verify log-likelihood is finite and reasonable
    for r in results:
        ll = r["log_likelihood"]
        if not np.isfinite(ll):
            print(f"WARNING: log-likelihood is not finite for N={r['N']}: {ll}")
        elif ll > 0:
            print(f"WARNING: log-likelihood is positive for N={r['N']}: {ll}")
        else:
            print(f"OK: log-likelihood for N={int(r['N'])} is finite and negative: {ll:.2f}")

    # Save results
    out_dir = os.path.dirname(__file__)
    out_path = os.path.join(out_dir, "results_sir.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
