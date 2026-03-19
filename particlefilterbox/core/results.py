"""ParticleFilterResults - Container for particle filter output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class ParticleFilterResults:
    """Container for particle filter results.

    Attributes
    ----------
    filtered_mean : ndarray, shape (T, k_states)
        Weighted mean E[x_t | y_{1:t}] at each time step.
    filtered_cov : ndarray, shape (T, k_states, k_states)
        Weighted covariance at each time step.
    filtered_quantiles : dict
        Quantiles {q: ndarray shape (T, k_states)} for q in [0.025, 0.5, 0.975].
    log_likelihood : float
        Total log-likelihood: sum_t log p(y_t | y_{1:t-1}).
    log_likelihood_increments : ndarray, shape (T,)
        Individual log-likelihood contributions.
    ess_history : ndarray, shape (T,)
        Effective Sample Size at each time step.
    resampled : ndarray, shape (T,), dtype bool
        Whether resampling occurred at each time step.
    n_particles : int
        Number of particles used.
    nobs : int
        Number of observations T.
    computation_time : float
        Computation time in seconds.
    particle_history : ndarray or None, shape (T, N, k_states)
        Full particle history (optional, memory intensive).
    weight_history : ndarray or None, shape (T, N)
        Full weight history (optional).
    ancestor_history : ndarray or None, shape (T, N), dtype int
        Full ancestor history (optional).
    """

    filtered_mean: NDArray[np.float64]
    filtered_cov: NDArray[np.float64]
    filtered_quantiles: dict[float, NDArray[np.float64]]
    log_likelihood: float
    log_likelihood_increments: NDArray[np.float64]
    ess_history: NDArray[np.float64]
    resampled: NDArray[np.bool_]
    n_particles: int
    nobs: int
    computation_time: float = 0.0
    particle_history: NDArray[np.float64] | None = None
    weight_history: NDArray[np.float64] | None = None
    ancestor_history: NDArray[np.intp] | None = None

    def summary(self) -> str:
        """Generate a formatted summary table.

        Returns
        -------
        str
            Summary string with key metrics.
        """
        mean_ess = float(np.mean(self.ess_history))
        min_ess = float(np.min(self.ess_history))
        n_resampled = int(np.sum(self.resampled))
        lines = [
            "=" * 60,
            "Particle Filter Results",
            "=" * 60,
            f"  Observations (T):      {self.nobs}",
            f"  Particles (N):         {self.n_particles}",
            f"  Log-likelihood:        {self.log_likelihood:.4f}",
            f"  Mean ESS:              {mean_ess:.1f} ({mean_ess / self.n_particles * 100:.1f}%)",
            f"  Min ESS:               {min_ess:.1f} ({min_ess / self.n_particles * 100:.1f}%)",
            f"  Resampling steps:      {n_resampled}/{self.nobs}",
            f"  Computation time:      {self.computation_time:.3f}s",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns for each state dimension's mean, std, quantiles, ESS.
        """
        k_states = self.filtered_mean.shape[1]
        data: dict[str, Any] = {"t": np.arange(self.nobs)}

        for k in range(k_states):
            prefix = f"state_{k}" if k_states > 1 else "state"
            data[f"{prefix}_mean"] = self.filtered_mean[:, k]
            data[f"{prefix}_std"] = np.sqrt(self.filtered_cov[:, k, k])
            for q, vals in self.filtered_quantiles.items():
                data[f"{prefix}_q{q:.3f}"] = vals[:, k]

        data["ess"] = self.ess_history
        data["log_lik_inc"] = self.log_likelihood_increments
        data["resampled"] = self.resampled

        return pd.DataFrame(data)

    def save(self, path: str | Path) -> None:
        """Save results to a .npz file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        """
        path = Path(path)
        save_dict: dict[str, Any] = {
            "filtered_mean": self.filtered_mean,
            "filtered_cov": self.filtered_cov,
            "log_likelihood": np.array(self.log_likelihood),
            "log_likelihood_increments": self.log_likelihood_increments,
            "ess_history": self.ess_history,
            "resampled": self.resampled,
            "n_particles": np.array(self.n_particles),
            "nobs": np.array(self.nobs),
            "computation_time": np.array(self.computation_time),
        }
        # Save quantiles
        for q, vals in self.filtered_quantiles.items():
            save_dict[f"quantile_{q:.3f}"] = vals

        # Save optional histories
        if self.particle_history is not None:
            save_dict["particle_history"] = self.particle_history
        if self.weight_history is not None:
            save_dict["weight_history"] = self.weight_history
        if self.ancestor_history is not None:
            save_dict["ancestor_history"] = self.ancestor_history

        np.savez_compressed(path, **save_dict)

    @staticmethod
    def load(path: str | Path) -> ParticleFilterResults:
        """Load results from a .npz file.

        Parameters
        ----------
        path : str or Path
            Input file path.

        Returns
        -------
        ParticleFilterResults
            Loaded results.
        """
        path = Path(path)
        data = np.load(path, allow_pickle=False)

        # Reconstruct quantiles
        quantiles: dict[float, NDArray[np.float64]] = {}
        for key in data.files:
            if key.startswith("quantile_"):
                q = float(key.replace("quantile_", ""))
                quantiles[q] = data[key]

        return ParticleFilterResults(
            filtered_mean=data["filtered_mean"],
            filtered_cov=data["filtered_cov"],
            filtered_quantiles=quantiles,
            log_likelihood=float(data["log_likelihood"]),
            log_likelihood_increments=data["log_likelihood_increments"],
            ess_history=data["ess_history"],
            resampled=data["resampled"],
            n_particles=int(data["n_particles"]),
            nobs=int(data["nobs"]),
            computation_time=float(data["computation_time"]),
            particle_history=data.get("particle_history"),
            weight_history=data.get("weight_history"),
            ancestor_history=data.get("ancestor_history"),
        )
