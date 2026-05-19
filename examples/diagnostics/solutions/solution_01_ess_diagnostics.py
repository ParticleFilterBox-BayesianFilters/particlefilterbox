"""Solution 01 - ESS and weight diagnostics.

Runs a Bootstrap PF on the simulated SV dataset and records per-timestep
diagnostics (ESS, max weight, weight entropy). Also compares against the
Auxiliary PF to show the effect of the proposal. Results are saved in
``results_ess_diagnostics.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel
from particlefilterbox.filters.auxiliary import AuxiliaryPF
from particlefilterbox.filters.bootstrap import BootstrapPF


class SVModel(ParticleFilterModel):
    """Basic stochastic volatility model h_t = mu + phi (h_{t-1}-mu) + sigma*eta."""

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


def weight_entropy(w: NDArray[np.float64]) -> float:
    mask = w > 0
    return float(-np.sum(w[mask] * np.log(w[mask])))


def run_pf_collect(
    pf: BootstrapPF | AuxiliaryPF,
    y: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Run PF step-by-step and collect ESS, max weight, entropy per timestep."""
    rng = pf._get_rng()
    cloud = pf.initialize(rng)
    T = len(y)
    N = pf.n_particles

    ess = np.zeros(T)
    max_w = np.zeros(T)
    entropy = np.zeros(T)
    ll_inc = np.zeros(T)
    resampled = np.zeros(T, dtype=bool)
    filtered_mean = np.zeros(T)

    prev_ess = float(N)
    for t in range(T):
        y_t = np.atleast_1d(y[t])
        cloud, ll_t = pf.filter_step(cloud, y_t, t)
        w = cloud.normalized_weights
        ess[t] = float(cloud.ess)
        max_w[t] = float(w.max())
        entropy[t] = weight_entropy(w)
        ll_inc[t] = float(ll_t)
        filtered_mean[t] = float(cloud.weighted_mean()[0])
        # Resampling is triggered when ESS rises sharply back to ~ N after being low.
        # Re-infer: when ESS at current step is much higher than expected from small
        # perturbation at previous step AND prev_ess < threshold * N.
        resampled[t] = prev_ess < 0.5 * N and ess[t] > 0.9 * N
        prev_ess = ess[t]

    return {
        "ess": ess,
        "max_w": max_w,
        "entropy": entropy,
        "ll_inc": ll_inc,
        "resampled": resampled,
        "filtered_mean": filtered_mean,
    }


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir.parent / "data" / "simulated_sv.csv"
    df = pd.read_csv(data_path)
    y_obs = df["y_obs"].to_numpy(dtype=np.float64)
    T = len(y_obs)

    N = 1000
    model_bpf = SVModel()
    cfg_bpf = PFConfig(n_particles=N, resampling="systematic", ess_threshold=0.5, seed=42)
    bpf = BootstrapPF(model_bpf, cfg_bpf)
    bpf_out = run_pf_collect(bpf, y_obs)

    model_apf = SVModel()
    cfg_apf = PFConfig(n_particles=N, resampling="systematic", ess_threshold=0.5, seed=42)
    apf = AuxiliaryPF(model_apf, cfg_apf)
    apf_out = run_pf_collect(apf, y_obs)

    out = pd.DataFrame(
        {
            "t": np.arange(T),
            "y_obs": y_obs,
            "abs_y": np.abs(y_obs),
            "bpf_ess": bpf_out["ess"],
            "bpf_max_w": bpf_out["max_w"],
            "bpf_entropy": bpf_out["entropy"],
            "bpf_rel_entropy": bpf_out["entropy"] / np.log(N),
            "bpf_ll_inc": bpf_out["ll_inc"],
            "bpf_filtered_mean": bpf_out["filtered_mean"],
            "apf_ess": apf_out["ess"],
            "apf_max_w": apf_out["max_w"],
            "apf_entropy": apf_out["entropy"],
            "apf_rel_entropy": apf_out["entropy"] / np.log(N),
            "apf_ll_inc": apf_out["ll_inc"],
            "apf_filtered_mean": apf_out["filtered_mean"],
        }
    )

    out_path = script_dir / "results_ess_diagnostics.csv"
    out.to_csv(out_path, index=False)

    bpf_ess_mean = float(bpf_out["ess"].mean())
    bpf_ess_min = float(bpf_out["ess"].min())
    apf_ess_mean = float(apf_out["ess"].mean())
    bpf_ll_total = float(bpf_out["ll_inc"].sum())
    corr_abs_y_ess = float(np.corrcoef(np.abs(y_obs), bpf_out["ess"])[0, 1])

    print(f"T={T}, N={N}")
    print(f"[Bootstrap PF] mean ESS = {bpf_ess_mean:.1f} / {N}  (>N/5={N / 5:.0f}? "
          f"{'yes' if bpf_ess_mean > N / 5 else 'no'})")
    print(f"[Bootstrap PF] min  ESS = {bpf_ess_min:.1f}")
    print(f"[Bootstrap PF] mean entropy = {bpf_out['entropy'].mean():.3f}  "
          f"(log N = {np.log(N):.3f})")
    print(f"[Bootstrap PF] mean rel entropy = "
          f"{(bpf_out['entropy'] / np.log(N)).mean():.3f}")
    print(f"[Bootstrap PF] total log-lik = {bpf_ll_total:.2f}")
    print(f"[Auxiliary PF] mean ESS = {apf_ess_mean:.1f} / {N}")
    print(f"corr(|y_t|, ESS_bpf)     = {corr_abs_y_ess:+.3f}")
    print(f"Saved -> {out_path.relative_to(script_dir.parent.parent)}")


if __name__ == "__main__":
    main()
