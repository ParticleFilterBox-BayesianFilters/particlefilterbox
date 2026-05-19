"""Solution script for Notebook 2: Merton jump-diffusion jump detection.

Runs the Auxiliary Particle Filter on simulated Merton jump-diffusion
returns, computes the posterior probability of a jump at each time step
and writes posterior probabilities, detected jumps and classification
metrics to ``results_jump_diffusion.csv``.

Usage
-----
    python solution_02_jump_diffusion.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm

from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel
from particlefilterbox.filters import AuxiliaryPF

SEED: int = 20260422
N_PARTICLES: int = 1000
THRESHOLD: float = 0.5
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_PATH = Path(__file__).resolve().parent / "results_jump_diffusion.csv"


class MertonJumpDiffusion(ParticleFilterModel):
    """State (jump_indicator, jump_size); iid in time."""

    k_states = 2
    k_obs = 1

    def __init__(
        self,
        mu: float = 0.0005,
        sigma: float = 0.01,
        lam: float = 0.05,
        mu_j: float = -0.02,
        sigma_j: float = 0.03,
    ) -> None:
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.lam = float(lam)
        self.mu_j = float(mu_j)
        self.sigma_j = float(sigma_j)

    @property
    def params(self) -> dict[str, float]:
        return {
            "mu": self.mu,
            "sigma": self.sigma,
            "lam": self.lam,
            "mu_j": self.mu_j,
            "sigma_j": self.sigma_j,
        }

    def initial_distribution(
        self,
        n_particles: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        j0 = rng.binomial(1, self.lam, size=n_particles).astype(float)
        z0 = rng.normal(self.mu_j, self.sigma_j, size=n_particles)
        return np.column_stack([j0, j0 * z0])

    def transition(
        self,
        particles: NDArray[np.float64],
        t: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        n = particles.shape[0]
        j = rng.binomial(1, self.lam, size=n).astype(float)
        z = rng.normal(self.mu_j, self.sigma_j, size=n)
        return np.column_stack([j, j * z])

    def transition_mean(
        self,
        particles: NDArray[np.float64],
        t: int,
    ) -> NDArray[np.float64]:
        n = particles.shape[0]
        return np.column_stack([
            np.full(n, self.lam),
            np.full(n, self.lam * self.mu_j),
        ])

    def log_observation_likelihood(
        self,
        particles: NDArray[np.float64],
        y_t: NDArray[np.float64],
        t: int,
    ) -> NDArray[np.float64]:
        r = float(np.asarray(y_t).ravel()[0])
        mean = self.mu + particles[:, 1]
        return norm.logpdf(r, loc=mean, scale=self.sigma)


def precision_recall(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
) -> dict[str, float]:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    df = pd.read_csv(DATA_DIR / "simulated_jump_diff.csv")
    y = df["returns"].to_numpy(dtype=float)
    jump_true = df["jump_true"].to_numpy(dtype=int)

    model = MertonJumpDiffusion()
    config = PFConfig(
        n_particles=N_PARTICLES,
        resampling="systematic",
        ess_threshold=0.5,
        seed=SEED,
    )
    apf = AuxiliaryPF(model=model, config=config)

    t_start = time.perf_counter()
    result = apf.filter(y)
    elapsed = time.perf_counter() - t_start

    p_jump = result.filtered_means[:, 0]
    z_hat = result.filtered_means[:, 1]
    jump_pred = (p_jump > THRESHOLD).astype(int)

    metrics = precision_recall(jump_true, jump_pred)

    out = pd.DataFrame({
        "t": df["t"].to_numpy(),
        "returns": y,
        "jump_true": jump_true,
        "jump_size_true": df["jump_size_true"].to_numpy(),
        "p_jump": p_jump,
        "jump_size_hat": z_hat,
        "jump_pred": jump_pred,
        "ess": result.ess_history,
        "resampled": result.resampled.astype(int),
    })
    out.to_csv(OUT_PATH, index=False)

    metrics_path = OUT_PATH.with_name("results_jump_diffusion_metrics.csv")
    metrics_df = pd.DataFrame(
        [
            {"metric": "log_likelihood", "value": float(result.log_likelihood)},
            {"metric": "mean_ess", "value": float(result.ess_history.mean())},
            {"metric": "threshold", "value": THRESHOLD},
            *[{"metric": k, "value": v} for k, v in metrics.items()],
        ]
    )
    metrics_df.to_csv(metrics_path, index=False)

    print("=" * 64)
    print("Jump-diffusion particle filter - solution_02_jump_diffusion.py")
    print("=" * 64)
    print(f"observations        : {len(df)}")
    print(f"particles           : {N_PARTICLES}")
    print(f"log-likelihood      : {result.log_likelihood:.2f}")
    print(f"mean ESS            : {result.ess_history.mean():.1f}")
    print(f"true jumps          : {int(jump_true.sum())}")
    print(f"detected jumps      : {int(jump_pred.sum())}")
    print(f"precision           : {metrics['precision']:.3f}")
    print(f"recall              : {metrics['recall']:.3f}")
    print(f"F1                  : {metrics['f1']:.3f}")
    print(f"elapsed             : {elapsed:.2f} s")
    print(f"CSV (per-step)      : {OUT_PATH}")
    print(f"CSV (metrics)       : {metrics_path}")


if __name__ == "__main__":
    main()
