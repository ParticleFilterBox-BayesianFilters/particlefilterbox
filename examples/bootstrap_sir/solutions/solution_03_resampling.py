"""Solution: Comparison of 4 resampling methods on the SV model.

Compares multinomial, systematic, stratified, and residual resampling.
Saves comparative table to CSV.
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
from particlefilterbox.filters.bootstrap import BootstrapPF
from particlefilterbox.resampling import get_resampling_fn


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


class ConfigurableBootstrapPF(BootstrapPF):
    """Bootstrap PF that uses the resampling method from config."""

    def _maybe_resample(
        self,
        cloud: object,
        rng: np.random.Generator,
    ) -> tuple[object, bool]:
        from particlefilterbox.core.cloud import ParticleCloud

        assert isinstance(cloud, ParticleCloud)
        ess = self._compute_ess(cloud)
        threshold = self.config.ess_threshold * self.n_particles
        if ess < threshold:
            resample_fn = get_resampling_fn(self.config.resampling)
            weights = cloud.normalized_weights
            indices = resample_fn(weights, rng=rng)
            cloud.resample(indices)
            return cloud, True
        return cloud, False


def main() -> None:
    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    df = pd.read_csv(os.path.join(data_dir, "simulated_sv.csv"))
    y_obs = df["y_obs"].values
    h_true = df["h_true"].values

    # Compare resampling methods
    methods = ["multinomial", "systematic", "stratified", "residual"]
    N = 1000
    n_runs = 5

    rows: list[dict[str, float | str]] = []

    for method in methods:
        rmses: list[float] = []
        ess_means: list[float] = []
        times: list[float] = []

        for run_i in range(n_runs):
            seed = 42 + run_i
            config = PFConfig(n_particles=N, resampling=method, seed=seed)
            pf = ConfigurableBootstrapPF(SVModel(), config)

            t0 = time.time()
            res = pf.filter(y_obs)
            elapsed = time.time() - t0

            h_filt = res.filtered_means[:, 0]
            rmse = float(np.sqrt(np.mean((h_filt - h_true) ** 2)))
            mean_ess = float(np.mean(res.ess_history))

            rmses.append(rmse)
            ess_means.append(mean_ess)
            times.append(elapsed)

        row = {
            "method": method,
            "RMSE_mean": np.mean(rmses),
            "RMSE_std": np.std(rmses),
            "ESS_mean": np.mean(ess_means),
            "ESS_N_pct": np.mean(ess_means) / N * 100,
            "time_s": np.mean(times),
        }
        rows.append(row)
        print(
            f"{method:>12s}: RMSE={row['RMSE_mean']:.4f} (std={row['RMSE_std']:.4f}), "
            f"ESS_mean={row['ESS_mean']:.1f} ({row['ESS_N_pct']:.1f}%), "
            f"time={row['time_s']:.3f}s"
        )

    print(f"\n({n_runs} runs averaged per method, N={N})")

    # Verify: systematic ESS should be comparable or superior to multinomial
    ess_systematic = next(r["ESS_mean"] for r in rows if r["method"] == "systematic")
    ess_multinomial = next(r["ESS_mean"] for r in rows if r["method"] == "multinomial")
    rmse_systematic = next(r["RMSE_mean"] for r in rows if r["method"] == "systematic")
    rmse_multinomial = next(r["RMSE_mean"] for r in rows if r["method"] == "multinomial")

    print(f"\nSystematic vs Multinomial:")
    print(f"  ESS: {ess_systematic:.1f} vs {ess_multinomial:.1f}")
    print(f"  RMSE: {rmse_systematic:.4f} vs {rmse_multinomial:.4f}")
    if ess_systematic >= ess_multinomial * 0.95:
        print("  OK: Systematic ESS is comparable or superior to multinomial")
    else:
        print("  WARNING: Systematic ESS is significantly lower than multinomial")

    # Save results
    out_dir = os.path.dirname(__file__)
    out_path = os.path.join(out_dir, "results_resampling.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
