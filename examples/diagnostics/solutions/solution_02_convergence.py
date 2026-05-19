"""Solution 02 - convergence diagnostics and model comparison.

Runs a Bootstrap PF on the simulated SV dataset at several particle counts,
estimates:

- RMSE of the filtered mean against the data-generating h_t and against a
  gold-standard large-N reference;
- variance of the log-likelihood estimator across repeated runs;
- marginal log-likelihood for three competing models (SV-true, SV-misspecified,
  GARCH-like) for model comparison.

All metrics are saved in ``results_convergence.csv``. The rows are per-N for the
convergence study; model-comparison rows are appended with ``metric=model_logL``
and ``N`` set to the particle count used for the comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel
from particlefilterbox.filters.bootstrap import BootstrapPF


class SVModel(ParticleFilterModel):
    """Basic stochastic volatility model (Kim, Shephard & Chib, 1998)."""

    k_states = 1
    k_obs = 1

    def __init__(self, mu: float = -1.0, phi: float = 0.97, sigma: float = 0.15) -> None:
        self.mu = mu
        self.phi = phi
        self.sigma = sigma

    @property
    def params(self) -> dict[str, float]:
        return {"mu": self.mu, "phi": self.phi, "sigma": self.sigma}

    def initial_distribution(
        self, n_particles: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        std = self.sigma / np.sqrt(1.0 - self.phi**2)
        return rng.normal(self.mu, std, size=(n_particles, 1))

    def transition(
        self,
        particles: NDArray[np.float64],
        t: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        h = particles[:, 0]
        h_new = self.mu + self.phi * (h - self.mu) + self.sigma * rng.standard_normal(len(h))
        return h_new.reshape(-1, 1)

    def log_observation_likelihood(
        self,
        particles: NDArray[np.float64],
        y_t: NDArray[np.float64],
        t: int,
    ) -> NDArray[np.float64]:
        h = particles[:, 0]
        vol = np.exp(h / 2.0)
        return -0.5 * np.log(2.0 * np.pi) - np.log(vol) - 0.5 * (float(y_t[0]) / vol) ** 2


class GARCHLikeModel(ParticleFilterModel):
    """GARCH(1,1)-style variance recursion wrapped as a PF model."""

    k_states = 1
    k_obs = 1

    def __init__(
        self,
        omega: float = 0.05,
        alpha: float = 0.10,
        beta: float = 0.85,
        sigma_jitter: float = 0.05,
    ) -> None:
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.sigma_jitter = sigma_jitter
        self._last_y = 0.0
        self._last_var = omega / max(1e-8, 1.0 - alpha - beta)

    @property
    def params(self) -> dict[str, float]:
        return {"omega": self.omega, "alpha": self.alpha, "beta": self.beta}

    def initial_distribution(
        self, n_particles: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        lh0 = np.log(self._last_var)
        return rng.normal(lh0, self.sigma_jitter, size=(n_particles, 1))

    def transition(
        self,
        particles: NDArray[np.float64],
        t: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        lh = particles[:, 0]
        v_det = self.omega + self.alpha * self._last_y**2 + self.beta * np.exp(lh)
        v_det = np.clip(v_det, 1e-8, None)
        lh_new = np.log(v_det) + self.sigma_jitter * rng.standard_normal(len(lh))
        return lh_new.reshape(-1, 1)

    def log_observation_likelihood(
        self,
        particles: NDArray[np.float64],
        y_t: NDArray[np.float64],
        t: int,
    ) -> NDArray[np.float64]:
        self._last_y = float(y_t[0])
        v = np.exp(particles[:, 0])
        return -0.5 * np.log(2.0 * np.pi * v) - 0.5 * float(y_t[0]) ** 2 / v


def run_pf(
    model: ParticleFilterModel,
    y: NDArray[np.float64],
    N: int,
    seed: int,
    ess_threshold: float = 0.5,
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    cfg = PFConfig(
        n_particles=N,
        resampling="systematic",
        ess_threshold=ess_threshold,
        seed=seed,
    )
    pf = BootstrapPF(model, cfg)
    res = pf.filter(y)
    return res.filtered_means[:, 0], float(res.log_likelihood), res.log_likelihoods


def gold_standard(
    y: NDArray[np.float64], N_ref: int, reps: int
) -> tuple[NDArray[np.float64], float]:
    T = len(y)
    ref_mean = np.zeros(T)
    ll_samples = np.empty(reps)
    for rep in range(reps):
        fm, ll, _ = run_pf(SVModel(), y, N_ref, seed=99 + rep)
        ref_mean += fm
        ll_samples[rep] = ll
    ref_mean /= reps
    return ref_mean, float(ll_samples.mean())


def convergence_study(
    y: NDArray[np.float64],
    h_true: NDArray[np.float64],
    ref_mean: NDArray[np.float64],
    N_list: list[int],
    n_reps: int,
) -> pd.DataFrame:
    rows = []
    for N in N_list:
        rmse_truth = np.empty(n_reps)
        rmse_mc = np.empty(n_reps)
        lls = np.empty(n_reps)
        for rep in range(n_reps):
            fm, ll, _ = run_pf(SVModel(), y, N, seed=1000 + rep)
            rmse_truth[rep] = np.sqrt(np.mean((fm - h_true) ** 2))
            rmse_mc[rep] = np.sqrt(np.mean((fm - ref_mean) ** 2))
            lls[rep] = ll
        rows.append(
            {
                "metric": "convergence",
                "N": N,
                "rmse_truth_mean": float(rmse_truth.mean()),
                "rmse_truth_std": float(rmse_truth.std()),
                "rmse_mc_mean": float(rmse_mc.mean()),
                "rmse_mc_std": float(rmse_mc.std()),
                "logL_mean": float(lls.mean()),
                "logL_std": float(lls.std()),
                "logL_var": float(lls.var()),
                "n_reps": n_reps,
            }
        )
    return pd.DataFrame(rows)


def model_comparison(
    y: NDArray[np.float64],
    factories: dict[str, Callable[[], ParticleFilterModel]],
    N: int,
    n_reps: int,
) -> pd.DataFrame:
    rows = []
    for name, factory in factories.items():
        lls = np.empty(n_reps)
        for rep in range(n_reps):
            _, ll, _ = run_pf(factory(), y, N, seed=2000 + rep)
            lls[rep] = ll
        rows.append(
            {
                "metric": "model_logL",
                "N": N,
                "model": name,
                "logL_mean": float(lls.mean()),
                "logL_std": float(lls.std()),
                "n_reps": n_reps,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir.parent / "data" / "simulated_sv.csv"
    df = pd.read_csv(data_path)
    y_obs = df["y_obs"].to_numpy(dtype=np.float64)
    h_true = df["h_true"].to_numpy(dtype=np.float64)
    T = len(y_obs)

    N_list = [50, 100, 200, 500, 1000, 2000, 5000]
    N_REPS = 10
    N_REF = 20_000
    N_REF_REPS = 3

    print(f"T={T}, N_list={N_list}, n_reps={N_REPS}")
    print(f"Computing gold-standard filter mean with N_ref={N_REF} x {N_REF_REPS} reps...")
    ref_mean, ll_ref = gold_standard(y_obs, N_REF, N_REF_REPS)
    print(f"  gold-standard logL = {ll_ref:.3f}")

    conv = convergence_study(y_obs, h_true, ref_mean, N_list, N_REPS)
    print("\nConvergence results:")
    for _, r in conv.iterrows():
        print(
            f"  N={int(r.N):>5d}  RMSE_truth={r.rmse_truth_mean:.4f}  "
            f"RMSE_MC={r.rmse_mc_mean:.4f}+/-{r.rmse_mc_std:.4f}  "
            f"logL_std={r.logL_std:.3f}"
        )

    # Empirical slope (RMSE vs N in log-log), theory: -0.5
    N_arr = conv["N"].to_numpy(dtype=float)
    slope_rmse = float(
        np.polyfit(np.log(N_arr), np.log(conv["rmse_mc_mean"].to_numpy()), 1)[0]
    )
    slope_var = float(
        np.polyfit(np.log(N_arr), np.log(conv["logL_var"].to_numpy()), 1)[0]
    )
    print(f"\nFitted MC-RMSE slope    = {slope_rmse:+.3f} (theory -0.5)")
    print(f"Fitted logL-var slope   = {slope_var:+.3f} (theory -1.0)")

    # Model comparison
    factories: dict[str, Callable[[], ParticleFilterModel]] = {
        "SV_true": lambda: SVModel(mu=-1.0, phi=0.97, sigma=0.15),
        "SV_misspecified": lambda: SVModel(mu=0.0, phi=0.50, sigma=0.80),
        "GARCH_like": lambda: GARCHLikeModel(omega=0.05, alpha=0.10, beta=0.85),
    }
    N_CMP = 2000
    N_REPS_CMP = 10
    print(f"\nModel comparison at N={N_CMP}, reps={N_REPS_CMP}:")
    mc = model_comparison(y_obs, factories, N_CMP, N_REPS_CMP)
    for _, r in mc.iterrows():
        print(f"  logL[{r.model:<17s}] = {r.logL_mean:.2f} +/- {r.logL_std:.2f}")

    best_pos = int(np.argmax(mc["logL_mean"].to_numpy()))
    best_model = str(mc["model"].iloc[best_pos])
    best_logL = float(mc["logL_mean"].iloc[best_pos])
    print(f"Best model: {best_model} (logL={best_logL:.2f})")
    assert best_model == "SV_true", "SV_true should have highest log-evidence"

    # Assemble final CSV: convergence rows + model-comparison rows + summary row
    summary = pd.DataFrame(
        [
            {
                "metric": "slope",
                "N": np.nan,
                "model": "",
                "slope_rmse_mc": slope_rmse,
                "slope_logL_var": slope_var,
                "n_reps": np.nan,
            }
        ]
    )
    mc["model"] = mc["model"].astype(str)
    conv["model"] = ""
    out = pd.concat([conv, mc, summary], ignore_index=True, sort=False)

    out_path = script_dir / "results_convergence.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path.relative_to(script_dir.parent.parent)}")


if __name__ == "__main__":
    main()
