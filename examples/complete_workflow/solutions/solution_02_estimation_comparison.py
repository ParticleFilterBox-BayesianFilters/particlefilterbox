"""Solution 02 — Compare SMC sampler, PMMH and PGAS on simulated SV data.

The dataset (`simulated_sv.csv`) was generated from the basic SV model with
true parameters (mu, phi, sigma_h) = (-1.0, 0.97, 0.15), so we can report
bias and RMSE for each method.

Output:
  - results_estimation_comparison.csv  : one row per (method, parameter)
                                         plus a global "all" row per method
                                         with summary metrics (bias, RMSE,
                                         ESS, time, log p(y) where applicable)
"""
from __future__ import annotations

import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from particlefilterbox.models.stochastic_volatility import StochasticVolatility  # noqa: E402
from particlefilterbox.pmcmc import PMMH, PGAS  # noqa: E402
from particlefilterbox.smc import SMCSampler  # noqa: E402

SEED = 42

TRUE_MU = -1.0
TRUE_PHI = 0.97
TRUE_SIGMA = 0.15
THETA_TRUE = np.array([TRUE_MU, TRUE_PHI, TRUE_SIGMA])
PARAM_NAMES = ["mu", "phi", "sigma_h"]

T_USE = 200

N_SMC = 300
MCMC_MOVES_SMC = 2
N_PF_SMC = 80

N_PF_PMMH = 250
ITER_PMMH = 1500
BURN_PMMH = 400

N_PF_PGAS = 35
ITER_PGAS = 500
BURN_PGAS = 120


@dataclass
class FilterResult:
    log_likelihood: float
    filtered_means: np.ndarray | None = None


class SVBasicWrapper:
    def __init__(self) -> None:
        self.sv = StochasticVolatility(variant="basic")
        self.param_names = list(self.sv.param_names)
        self._rng_internal = np.random.default_rng(0)

    def set_params(self, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=np.float64)
        for i, name in enumerate(self.param_names):
            self.sv.params[name] = float(theta[i])

    def get_params(self) -> np.ndarray:
        return np.array([self.sv.params[n] for n in self.param_names])

    def filter(self, endog: np.ndarray, n_particles: int = 200,
               rng: np.random.Generator | None = None) -> FilterResult:
        rng = rng if rng is not None else self._rng_internal
        mu = self.sv.params["mu"]; phi = self.sv.params["phi"]; sigma = self.sv.params["sigma"]
        if abs(phi) >= 1.0 or sigma <= 0:
            return FilterResult(log_likelihood=-np.inf)
        var_stat = sigma ** 2 / (1.0 - phi ** 2)
        particles = rng.normal(mu, np.sqrt(max(var_stat, 1e-10)), size=n_particles)
        log_lik = 0.0
        fm = np.zeros(len(endog))
        for t in range(len(endog)):
            vol = np.exp(particles / 2.0)
            log_w = -0.5 * np.log(2 * np.pi) - np.log(vol) - 0.5 * (endog[t] / vol) ** 2
            m = float(np.max(log_w))
            w = np.exp(log_w - m)
            s = float(np.sum(w))
            if s < 1e-300:
                return FilterResult(log_likelihood=-np.inf)
            log_lik += m + np.log(s) - np.log(n_particles)
            w /= s
            fm[t] = float(np.sum(w * particles))
            idx = rng.choice(n_particles, size=n_particles, p=w)
            particles = particles[idx]
            particles = mu + phi * (particles - mu) + sigma * rng.standard_normal(n_particles)
        return FilterResult(log_likelihood=log_lik, filtered_means=fm)

    def initial_sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        mu = self.sv.params["mu"]; phi = self.sv.params["phi"]; sigma = self.sv.params["sigma"]
        if abs(phi) >= 1.0 or sigma <= 0:
            return rng.standard_normal(n)
        var_stat = sigma ** 2 / (1.0 - phi ** 2)
        return rng.normal(mu, np.sqrt(max(var_stat, 1e-10)), size=n)

    def transition_sample(self, x_prev, rng):
        mu = self.sv.params["mu"]; phi = self.sv.params["phi"]; sigma = self.sv.params["sigma"]
        hp = float(x_prev) if np.ndim(x_prev) == 0 else float(np.asarray(x_prev).flatten()[0])
        return float(mu + phi * (hp - mu) + sigma * rng.standard_normal())

    def transition_logpdf(self, x_new, x_old):
        mu = self.sv.params["mu"]; phi = self.sv.params["phi"]; sigma = self.sv.params["sigma"]
        if sigma <= 0:
            return -np.inf
        hn = float(x_new) if np.ndim(x_new) == 0 else float(np.asarray(x_new).flatten()[0])
        ho = float(x_old) if np.ndim(x_old) == 0 else float(np.asarray(x_old).flatten()[0])
        return float(stats.norm.logpdf(hn, loc=mu + phi * (ho - mu), scale=sigma))

    def observation_logpdf(self, y, x):
        h = float(x) if np.ndim(x) == 0 else float(np.asarray(x).flatten()[0])
        vol = np.exp(h / 2.0)
        if vol < 1e-300:
            return -np.inf
        return float(-0.5 * np.log(2 * np.pi) - np.log(vol) - 0.5 * (float(y) / vol) ** 2)


class SVBasicPrior:
    def logpdf(self, theta: np.ndarray) -> float:
        mu, phi, sig = float(theta[0]), float(theta[1]), float(theta[2])
        if not (0 < phi < 1) or sig <= 0:
            return -np.inf
        return float(stats.norm.logpdf(mu, loc=-1.0, scale=1.0)
                     + stats.beta.logpdf(phi, 20.0, 1.5)
                     + stats.invgamma.logpdf(sig, a=2.5, scale=0.025))

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return np.array([
            float(rng.normal(-1.0, 1.0)),
            float(stats.beta.rvs(20.0, 1.5, random_state=rng)),
            float(stats.invgamma.rvs(2.5, scale=0.025, random_state=rng)),
        ])

    @property
    def cov(self) -> np.ndarray:
        c = np.zeros(3)
        c[0] = 1.0
        a, b = 20.0, 1.5
        c[1] = (a * b) / ((a + b) ** 2 * (a + b + 1))
        c[2] = 0.025 ** 2 / ((2.5 - 1) ** 2 * (2.5 - 2))
        return c


def main() -> None:
    print("=" * 78)
    print("SOLUTION 02 — SMC vs PMMH vs PGAS on simulated SV data")
    print("=" * 78)

    data_path = HERE.parent / "data" / "simulated_sv.csv"
    df = pd.read_csv(data_path).iloc[:T_USE].reset_index(drop=True)
    y_obs = df["y_obs"].to_numpy(dtype=np.float64)
    T = len(y_obs)
    print(f"T = {T}   true theta = (mu={TRUE_MU}, phi={TRUE_PHI}, sigma_h={TRUE_SIGMA})")

    # ---------------- SMC sampler --------------------------------------
    prior_for_closures = SVBasicPrior()
    model_for_closures = SVBasicWrapper()
    _pf_rng = np.random.default_rng(10_000)

    def _log_lik_pf(theta: np.ndarray) -> float:
        model_for_closures.set_params(theta)
        return float(model_for_closures.filter(y_obs, n_particles=N_PF_SMC, rng=_pf_rng).log_likelihood)

    def log_target(theta: np.ndarray) -> float:
        lp = prior_for_closures.logpdf(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + _log_lik_pf(theta)

    def log_prior_fn(theta: np.ndarray) -> float:
        return prior_for_closures.logpdf(theta)

    def sample_prior_fn(rng: np.random.Generator) -> np.ndarray:
        return prior_for_closures.sample(rng)

    print(f"\n[SMC sampler] N={N_SMC}, MCMC moves={MCMC_MOVES_SMC}, PF N={N_PF_SMC} ...")
    t0 = time.perf_counter()
    smc = SMCSampler(
        target_logpdf=log_target,
        prior_logpdf=log_prior_fn,
        prior_sample=sample_prior_fn,
        n_particles=N_SMC,
        n_mcmc_moves=MCMC_MOVES_SMC,
        ess_target_ratio=0.5,
        seed=SEED,
    )
    smc_res = smc.run()
    elapsed_smc = time.perf_counter() - t0
    smc_mean = smc_res.posterior_mean()
    smc_std = smc_res.posterior_std()
    smc_ci = smc_res.credible_interval(0.95)
    smc_logZ = float(smc_res.log_evidence)
    smc_final_ess = float(smc_res.ess_history[-1]) if smc_res.ess_history else float(smc_res.n_particles)
    print(f"  finished in {elapsed_smc:.1f}s   tempering stages = {smc_res.n_steps}   "
          f"final ESS = {smc_final_ess:.0f}/{smc_res.n_particles}   log p(y) = {smc_logZ:+.3f}")
    for j, n in enumerate(PARAM_NAMES):
        print(f"  {n:<7} true={THETA_TRUE[j]:+.4f}  mean={smc_mean[j]:+.4f}  "
              f"std={smc_std[j]:.4f}  CI=[{smc_ci[j, 0]:+.4f}, {smc_ci[j, 1]:+.4f}]")

    # ---------------- PMMH ---------------------------------------------
    print(f"\n[PMMH] N_pf={N_PF_PMMH}, iter={ITER_PMMH}, burn={BURN_PMMH} ...")
    pmmh = PMMH(
        model=SVBasicWrapper(),
        prior=SVBasicPrior(),
        n_particles=N_PF_PMMH,
        n_iterations=ITER_PMMH,
        proposal_cov="adaptive",
        target_acceptance=0.234,
        burnin=BURN_PMMH,
        thin=1,
        seed=SEED,
    )
    t0 = time.perf_counter()
    pmmh_res = pmmh.run(endog=y_obs, theta_init=THETA_TRUE.copy(), verbose=0)
    elapsed_pmmh = time.perf_counter() - t0
    pmmh_mean = pmmh_res.posterior_mean()
    pmmh_std = pmmh_res.posterior_std()
    pmmh_ci_lo, pmmh_ci_hi = pmmh_res.credible_interval(0.05)
    pmmh_ess = np.array([pmmh_res.effective_sample_size(j) for j in range(3)])
    print(f"  finished in {elapsed_pmmh:.1f}s   acceptance={pmmh_res.acceptance_rate():.3f}")
    for j, n in enumerate(PARAM_NAMES):
        print(f"  {n:<7} true={THETA_TRUE[j]:+.4f}  mean={pmmh_mean[j]:+.4f}  "
              f"std={pmmh_std[j]:.4f}  CI=[{pmmh_ci_lo[j]:+.4f}, {pmmh_ci_hi[j]:+.4f}]  "
              f"ESS={pmmh_ess[j]:.1f}")

    # ---------------- PGAS ---------------------------------------------
    def param_sampler_pgas(model, states, endog, theta_current, rng):
        pr = SVBasicPrior()
        step = np.array([0.05, 0.008, 0.010])
        n_sub = 2
        h = states.reshape(-1)
        T_ = len(endog)

        def cond_loglik(th: np.ndarray) -> float:
            mu_, phi_, sig_ = float(th[0]), float(th[1]), float(th[2])
            if not (0 < phi_ < 1) or sig_ <= 0:
                return -np.inf
            vs = sig_ ** 2 / (1.0 - phi_ ** 2)
            ll = float(stats.norm.logpdf(h[0], loc=mu_, scale=np.sqrt(max(vs, 1e-10))))
            vol = np.exp(h / 2.0)
            ll += float(np.sum(-0.5 * np.log(2 * np.pi) - np.log(vol) - 0.5 * (endog / vol) ** 2))
            if T_ > 1:
                m = mu_ + phi_ * (h[:-1] - mu_)
                ll += float(np.sum(stats.norm.logpdf(h[1:], loc=m, scale=sig_)))
            return ll

        th = theta_current.copy()
        for _ in range(n_sub):
            for j in range(len(th)):
                prop = th.copy()
                prop[j] += step[j] * rng.standard_normal()
                lp_prop = pr.logpdf(prop)
                if not np.isfinite(lp_prop):
                    continue
                lp_curr = pr.logpdf(th)
                ll_curr = cond_loglik(th)
                ll_prop = cond_loglik(prop)
                if np.log(rng.random()) < (lp_prop + ll_prop) - (lp_curr + ll_curr):
                    th = prop
        return th

    print(f"\n[PGAS] N_pf={N_PF_PGAS}, iter={ITER_PGAS}, burn={BURN_PGAS} ...")
    pgas = PGAS(
        model=SVBasicWrapper(),
        prior=SVBasicPrior(),
        n_particles=N_PF_PGAS,
        n_iterations=ITER_PGAS,
        param_sampler=param_sampler_pgas,
        burnin=BURN_PGAS,
        thin=1,
        seed=SEED,
    )
    t0 = time.perf_counter()
    pgas_res = pgas.run(endog=y_obs, theta_init=THETA_TRUE.copy(), verbose=0)
    elapsed_pgas = time.perf_counter() - t0
    pgas_mean = pgas_res.posterior_mean()
    pgas_std = pgas_res.posterior_std()
    pgas_ci_lo, pgas_ci_hi = pgas_res.credible_interval(0.05)
    pgas_ess = np.array([pgas_res.effective_sample_size(j) for j in range(3)])
    print(f"  finished in {elapsed_pgas:.1f}s")
    for j, n in enumerate(PARAM_NAMES):
        print(f"  {n:<7} true={THETA_TRUE[j]:+.4f}  mean={pgas_mean[j]:+.4f}  "
              f"std={pgas_std[j]:.4f}  CI=[{pgas_ci_lo[j]:+.4f}, {pgas_ci_hi[j]:+.4f}]  "
              f"ESS={pgas_ess[j]:.1f}")

    # ---------------- Comparison table ---------------------------------
    rows: list[dict] = []
    for j, name in enumerate(PARAM_NAMES):
        true = float(THETA_TRUE[j])
        # SMC (the SMC sampler does not produce per-parameter ESS; report final ESS)
        rows.append({
            "method": "SMC",
            "param": name,
            "true": true,
            "posterior_mean": float(smc_mean[j]),
            "posterior_std": float(smc_std[j]),
            "ci95_lower": float(smc_ci[j, 0]),
            "ci95_upper": float(smc_ci[j, 1]),
            "bias": float(smc_mean[j] - true),
            "abs_bias": abs(float(smc_mean[j] - true)),
            "sq_error": float((smc_mean[j] - true) ** 2),
            "within_ci": bool(smc_ci[j, 0] <= true <= smc_ci[j, 1]),
            "ess": smc_final_ess,
            "time_seconds": elapsed_smc,
            "log_evidence": smc_logZ,
        })
        rows.append({
            "method": "PMMH",
            "param": name,
            "true": true,
            "posterior_mean": float(pmmh_mean[j]),
            "posterior_std": float(pmmh_std[j]),
            "ci95_lower": float(pmmh_ci_lo[j]),
            "ci95_upper": float(pmmh_ci_hi[j]),
            "bias": float(pmmh_mean[j] - true),
            "abs_bias": abs(float(pmmh_mean[j] - true)),
            "sq_error": float((pmmh_mean[j] - true) ** 2),
            "within_ci": bool(pmmh_ci_lo[j] <= true <= pmmh_ci_hi[j]),
            "ess": float(pmmh_ess[j]),
            "time_seconds": elapsed_pmmh,
            "log_evidence": np.nan,
        })
        rows.append({
            "method": "PGAS",
            "param": name,
            "true": true,
            "posterior_mean": float(pgas_mean[j]),
            "posterior_std": float(pgas_std[j]),
            "ci95_lower": float(pgas_ci_lo[j]),
            "ci95_upper": float(pgas_ci_hi[j]),
            "bias": float(pgas_mean[j] - true),
            "abs_bias": abs(float(pgas_mean[j] - true)),
            "sq_error": float((pgas_mean[j] - true) ** 2),
            "within_ci": bool(pgas_ci_lo[j] <= true <= pgas_ci_hi[j]),
            "ess": float(pgas_ess[j]),
            "time_seconds": elapsed_pgas,
            "log_evidence": np.nan,
        })

    # Append summary rows aggregated across the three parameters.
    def _summary_row(method: str, mean: np.ndarray, ess: float | np.ndarray,
                     elapsed: float, log_ev: float) -> dict:
        bias_avg = float(np.mean(mean - THETA_TRUE))
        rmse = float(np.sqrt(np.mean((mean - THETA_TRUE) ** 2)))
        ess_min = float(np.min(np.atleast_1d(ess)))
        return {
            "method": method,
            "param": "all",
            "true": np.nan,
            "posterior_mean": np.nan,
            "posterior_std": np.nan,
            "ci95_lower": np.nan,
            "ci95_upper": np.nan,
            "bias": bias_avg,
            "abs_bias": float(np.mean(np.abs(mean - THETA_TRUE))),
            "sq_error": rmse ** 2,
            "within_ci": np.nan,
            "ess": ess_min,
            "time_seconds": elapsed,
            "log_evidence": log_ev,
        }

    rows.append(_summary_row("SMC", smc_mean, smc_final_ess, elapsed_smc, smc_logZ))
    rows.append(_summary_row("PMMH", pmmh_mean, pmmh_ess, elapsed_pmmh, np.nan))
    rows.append(_summary_row("PGAS", pgas_mean, pgas_ess, elapsed_pgas, np.nan))

    out_df = pd.DataFrame(rows)
    out_path = HERE / "results_estimation_comparison.csv"
    out_df.to_csv(out_path, index=False)

    print("\n" + "=" * 78)
    print("FINAL COMPARISON  (true theta = "
          f"{THETA_TRUE.tolist()},  T = {T})")
    print("=" * 78)
    summary = out_df[out_df["param"] == "all"][
        ["method", "bias", "abs_bias", "sq_error", "ess", "time_seconds", "log_evidence"]
    ].rename(columns={"sq_error": "MSE", "ess": "ESS_min", "time_seconds": "time_s"})
    summary.insert(2, "RMSE", np.sqrt(summary["MSE"]))
    print(summary.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
