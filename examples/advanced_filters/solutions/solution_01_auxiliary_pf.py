"""Solution: Auxiliary PF vs Bootstrap PF on Stochastic Volatility model.

Runs APF and BPF with multiple particle counts, compares ESS, RMSE, and
log-likelihood. Saves results to CSV.
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
from particlefilterbox.filters.auxiliary import AuxiliaryPF
from particlefilterbox.filters.bootstrap import BootstrapPF


class SVModel(ParticleFilterModel):
    """Stochastic Volatility model with transition_mean for APF."""

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
        var_stat = self.sigma**2 / (1.0 - self.phi**2)
        return rng.normal(self.mu, np.sqrt(var_stat), size=(n_particles, 1))

    def transition(self, particles: np.ndarray, t: int, rng: np.random.Generator) -> np.ndarray:
        eta = rng.standard_normal(size=particles.shape)
        return self.mu + self.phi * (particles - self.mu) + self.sigma * eta

    def transition_mean(self, particles: np.ndarray, t: int) -> np.ndarray:
        """Deterministic transition mean (used by APF first-stage weights)."""
        return self.mu + self.phi * (particles - self.mu)

    def log_observation_likelihood(
        self, particles: np.ndarray, y_t: np.ndarray, t: int
    ) -> np.ndarray:
        h = particles[:, 0]
        vol = np.exp(h / 2.0)
        return -0.5 * np.log(2.0 * np.pi) - np.log(vol) - 0.5 * (float(y_t[0]) / vol) ** 2


def main() -> None:
    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "bootstrap_sir", "data")
    df = pd.read_csv(os.path.join(data_dir, "simulated_sv.csv"))
    y_obs = df["y_obs"].values
    h_true = df["h_true"].values
    T = len(df)

    model = SVModel(mu=-1.0, phi=0.97, sigma=0.15)
    print(f"SV Model parameters: {model.params}")
    print(f"Loaded {T} observations")

    # Run APF and BPF with multiple N
    results: list[dict[str, object]] = []
    N_values = [500, 1000, 2000]
    seed = 42

    for N in N_values:
        for filter_name, filter_cls in [("APF", AuxiliaryPF), ("BPF", BootstrapPF)]:
            config = PFConfig(n_particles=N, resampling="systematic", seed=seed)
            pf = filter_cls(model, config)

            t0 = time.time()
            res = pf.filter(y_obs)
            elapsed = time.time() - t0

            h_filt = res.filtered_means[:, 0]
            rmse = float(np.sqrt(np.mean((h_filt - h_true) ** 2)))
            mean_ess = float(np.mean(res.ess_history))
            min_ess = float(np.min(res.ess_history))
            log_lik = float(res.log_likelihood)

            results.append({
                "Filter": filter_name,
                "N": N,
                "RMSE": round(rmse, 6),
                "ESS_mean": round(mean_ess, 2),
                "ESS_min": round(min_ess, 2),
                "ESS_N_pct": round(mean_ess / N * 100, 2),
                "log_likelihood": round(log_lik, 4),
                "time_s": round(elapsed, 4),
            })
            print(
                f"{filter_name} N={N:5d}: RMSE={rmse:.4f}, ESS_mean={mean_ess:.1f} "
                f"({mean_ess / N * 100:.1f}%), log_lik={log_lik:.2f}, time={elapsed:.2f}s"
            )

    # Verify: APF should have higher ESS than BPF for same N
    print("\n--- ESS Comparison ---")
    for N in N_values:
        apf_row = [r for r in results if r["Filter"] == "APF" and r["N"] == N][0]
        bpf_row = [r for r in results if r["Filter"] == "BPF" and r["N"] == N][0]
        apf_ess = apf_row["ESS_mean"]
        bpf_ess = bpf_row["ESS_mean"]
        if apf_ess > bpf_ess:
            print(f"  N={N}: APF ESS ({apf_ess:.1f}) > BPF ESS ({bpf_ess:.1f}) -- OK")
        else:
            print(f"  N={N}: APF ESS ({apf_ess:.1f}) <= BPF ESS ({bpf_ess:.1f}) -- UNEXPECTED")

    # Save results
    out_dir = os.path.dirname(__file__)
    out_path = os.path.join(out_dir, "results_auxiliary_pf.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
