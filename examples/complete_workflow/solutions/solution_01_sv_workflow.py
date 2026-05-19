"""Solution 01 — End-to-end SV workflow on S&P 500 daily returns.

Pipeline:
  1. Bootstrap particle filter at plug-in theta.
  2. Bayesian estimation of (mu, phi, sigma_h) via PGAS.
  3. FFBSi smoothing at the PGAS posterior mean.
  4. Volatility forecasting from the final particle cloud.

Outputs (written to the same directory as this script):
  - results_sv_workflow_filtering.csv  (per-t filtered & smoothed volatility)
  - results_sv_workflow_params.csv     (PGAS posterior summary)
  - results_sv_workflow_forecast.csv   (forecast cone over horizons 1..H)
"""
from __future__ import annotations

import sys
import time
import warnings
from dataclasses import dataclass, field
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
from particlefilterbox.pmcmc import PGAS  # noqa: E402
from particlefilterbox.smoothers import FFBSi  # noqa: E402

SEED = 42

T_USE = 250
N_FILTER = 1500
N_PGAS = 30
ITER_PGAS = 400
BURN_PGAS = 100
M_TRAJ = 300
H_FORECAST = 25


@dataclass
class FilterResult:
    log_likelihood: float
    filtered_means: np.ndarray | None = None
    filtered_vars: np.ndarray | None = None
    ess_history: np.ndarray | None = None
    final_particles: np.ndarray | None = None
    final_weights: np.ndarray | None = None
    particles_history: list = field(default_factory=list)
    weights_history: list = field(default_factory=list)
    ancestor_indices: list = field(default_factory=list)


class SVBasicWrapper:
    """Adapter for basic StochasticVolatility -> PMCMC / smoother interfaces."""

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

    def filter(
        self,
        endog: np.ndarray,
        n_particles: int = 200,
        rng: np.random.Generator | None = None,
        store_history: bool = False,
    ) -> FilterResult:
        rng = rng if rng is not None else self._rng_internal
        T = len(endog)
        mu = self.sv.params["mu"]; phi = self.sv.params["phi"]; sigma = self.sv.params["sigma"]
        if abs(phi) >= 1.0 or sigma <= 0:
            return FilterResult(log_likelihood=-np.inf)
        var_stat = sigma ** 2 / (1.0 - phi ** 2)
        particles = rng.normal(mu, np.sqrt(max(var_stat, 1e-10)), size=n_particles)

        log_lik = 0.0
        f_mean = np.zeros(T); f_var = np.zeros(T); ess_hist = np.zeros(T)
        part_hist: list = []; w_hist: list = []; anc_hist: list = []

        for t in range(T):
            vol = np.exp(particles / 2.0)
            log_w = -0.5 * np.log(2 * np.pi) - np.log(vol) - 0.5 * (endog[t] / vol) ** 2
            max_lw = float(np.max(log_w))
            w = np.exp(log_w - max_lw)
            sw = float(np.sum(w))
            if sw < 1e-300:
                return FilterResult(log_likelihood=-np.inf)
            log_lik += max_lw + np.log(sw) - np.log(n_particles)
            w_norm = w / sw
            f_mean[t] = float(np.sum(w_norm * particles))
            f_var[t] = float(np.sum(w_norm * (particles - f_mean[t]) ** 2))
            ess_hist[t] = 1.0 / float(np.sum(w_norm ** 2))

            if store_history:
                part_hist.append(particles.reshape(-1, 1).copy())
                w_hist.append(w_norm.copy())

            if ess_hist[t] < n_particles / 2:
                cumsum = np.cumsum(w_norm)
                u = (rng.random() + np.arange(n_particles)) / n_particles
                idx = np.clip(np.searchsorted(cumsum, u), 0, n_particles - 1)
                if store_history:
                    anc_hist.append(idx.copy())
                particles = particles[idx]
            else:
                if store_history:
                    anc_hist.append(np.arange(n_particles))

            particles = mu + phi * (particles - mu) + sigma * rng.standard_normal(n_particles)

        return FilterResult(
            log_likelihood=log_lik,
            filtered_means=f_mean,
            filtered_vars=f_var,
            ess_history=ess_hist,
            final_particles=particles,
            final_weights=np.ones(n_particles) / n_particles,
            particles_history=part_hist,
            weights_history=w_hist,
            ancestor_indices=anc_hist,
        )

    def initial_sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        mu = self.sv.params["mu"]; phi = self.sv.params["phi"]; sigma = self.sv.params["sigma"]
        if abs(phi) >= 1.0 or sigma <= 0:
            return rng.standard_normal(n)
        var_stat = sigma ** 2 / (1.0 - phi ** 2)
        return rng.normal(mu, np.sqrt(max(var_stat, 1e-10)), size=n)

    def transition_sample(self, x_prev, rng) -> float:
        mu = self.sv.params["mu"]; phi = self.sv.params["phi"]; sigma = self.sv.params["sigma"]
        h_prev = float(x_prev) if np.ndim(x_prev) == 0 else float(np.asarray(x_prev).flatten()[0])
        return float(mu + phi * (h_prev - mu) + sigma * rng.standard_normal())

    def transition_logpdf(self, x_new, x_old) -> float:
        mu = self.sv.params["mu"]; phi = self.sv.params["phi"]; sigma = self.sv.params["sigma"]
        if sigma <= 0:
            return -np.inf
        h_new = float(x_new) if np.ndim(x_new) == 0 else float(np.asarray(x_new).flatten()[0])
        h_old = float(x_old) if np.ndim(x_old) == 0 else float(np.asarray(x_old).flatten()[0])
        mean = mu + phi * (h_old - mu)
        return float(stats.norm.logpdf(h_new, loc=mean, scale=sigma))

    def observation_logpdf(self, y: float, x) -> float:
        h = float(x) if np.ndim(x) == 0 else float(np.asarray(x).flatten()[0])
        vol = np.exp(h / 2.0)
        if vol < 1e-300:
            return -np.inf
        return float(-0.5 * np.log(2 * np.pi) - np.log(vol) - 0.5 * (float(y) / vol) ** 2)


class SmootherModel:
    """Matches the interface expected by FFBSi.smooth()."""

    def __init__(self, mu: float, phi: float, sigma: float) -> None:
        self.mu = mu; self.phi = phi; self.sigma = sigma

    def log_transition_density(self, x_new: np.ndarray, x_old: np.ndarray, t: int) -> np.ndarray:
        if x_new.ndim == 1:
            x_new = x_new.reshape(1, -1)
        if x_old.ndim == 1:
            x_old = x_old.reshape(1, -1)
        mean = self.mu + self.phi * (x_old[:, 0] - self.mu)
        return (-0.5 * np.log(2 * np.pi * self.sigma ** 2)
                - 0.5 * ((x_new[:, 0] - mean) / self.sigma) ** 2)

    def log_observation_density(self, y, x: np.ndarray, t: int) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        vol = np.exp(x[:, 0] / 2.0)
        y_val = float(np.asarray(y).flatten()[0]) if hasattr(y, "__len__") else float(y)
        return -0.5 * np.log(2 * np.pi) - np.log(vol) - 0.5 * (y_val / vol) ** 2


class SVBasicPrior:
    """Data-adaptive SV prior."""

    def __init__(self, loc_mu: float = 0.0, scale_mu: float = 5.0) -> None:
        self.loc_mu = loc_mu; self.scale_mu = scale_mu
        self.a_phi = 20.0; self.b_phi = 1.5
        self.a_sig = 2.5; self.b_sig = 0.025

    def logpdf(self, theta: np.ndarray) -> float:
        mu_v, phi_v, sig_v = float(theta[0]), float(theta[1]), float(theta[2])
        if not (0 < phi_v < 1) or sig_v <= 0:
            return -np.inf
        lp = float(stats.norm.logpdf(mu_v, loc=self.loc_mu, scale=self.scale_mu))
        lp += float(stats.beta.logpdf(phi_v, self.a_phi, self.b_phi))
        lp += float(stats.invgamma.logpdf(sig_v, a=self.a_sig, scale=self.b_sig))
        return lp

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        mu_s = float(rng.normal(self.loc_mu, self.scale_mu))
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


def make_param_sampler(mu_loc: float):
    """Closure that returns a PGAS parameter sampler with data-centred prior on mu."""

    def sampler(model, states, endog, theta_current, rng):
        prior = SVBasicPrior(loc_mu=mu_loc, scale_mu=3.0)
        step = np.array([0.05, 0.008, 0.010])
        n_sub = 3

        def cond_loglik(theta: np.ndarray) -> float:
            mu_, phi_, sig_ = float(theta[0]), float(theta[1]), float(theta[2])
            if not (0 < phi_ < 1) or sig_ <= 0:
                return -np.inf
            h = states.reshape(-1)
            T_ = len(endog)
            var_stat = sig_ ** 2 / (1.0 - phi_ ** 2)
            ll = float(stats.norm.logpdf(h[0], loc=mu_, scale=np.sqrt(max(var_stat, 1e-10))))
            vol = np.exp(h / 2.0)
            ll += float(np.sum(-0.5 * np.log(2 * np.pi) - np.log(vol) - 0.5 * (endog / vol) ** 2))
            if T_ > 1:
                m = mu_ + phi_ * (h[:-1] - mu_)
                ll += float(np.sum(stats.norm.logpdf(h[1:], loc=m, scale=sig_)))
            return ll

        theta = theta_current.copy()
        for _ in range(n_sub):
            for j in range(len(theta)):
                prop = theta.copy()
                prop[j] += step[j] * rng.standard_normal()
                lp_prop = prior.logpdf(prop)
                if not np.isfinite(lp_prop):
                    continue
                lp_curr = prior.logpdf(theta)
                ll_curr = cond_loglik(theta)
                ll_prop = cond_loglik(prop)
                if np.log(rng.random()) < (lp_prop + ll_prop) - (lp_curr + ll_curr):
                    theta = prop
        return theta

    return sampler


class _SmootherInput:
    """Adapter exposing the attributes that FFBSi.smooth() expects."""

    def __init__(self, fr: FilterResult, observations: np.ndarray) -> None:
        self.particles_history = fr.particles_history
        self.weights_history = fr.weights_history
        self.filtered_mean = (fr.filtered_means.reshape(-1, 1)
                              if fr.filtered_means is not None else np.zeros((0, 1)))
        self.filtered_cov = (fr.filtered_vars.reshape(-1, 1, 1)
                             if fr.filtered_vars is not None else np.zeros((0, 1, 1)))
        self.observations = observations.reshape(-1, 1)
        self.ancestor_indices = fr.ancestor_indices


def main() -> None:
    print("=" * 78)
    print("SOLUTION 01 — Stochastic Volatility complete workflow on S&P 500")
    print("=" * 78)

    data_path = HERE.parent / "data" / "sp500_returns.csv"
    df_full = pd.read_csv(data_path, parse_dates=["date"])
    df = df_full.tail(T_USE).reset_index(drop=True)
    y_obs = df["returns"].to_numpy(dtype=np.float64)
    dates = pd.to_datetime(df["date"]).to_numpy()
    T = len(y_obs)
    print(f"Loaded {len(df_full)} rows; using last {T} observations")
    print(f"Range: {pd.Timestamp(dates[0]).date()} -> {pd.Timestamp(dates[-1]).date()}")

    # ---- Step 1: Bootstrap PF at plug-in theta -----------------------------
    mu_plugin = float(2.0 * np.log(y_obs.std(ddof=1)))
    theta_plugin = np.array([mu_plugin, 0.95, 0.20])
    print(f"\nPlug-in theta = {theta_plugin}")

    model = SVBasicWrapper()
    model.set_params(theta_plugin)
    rng_filter = np.random.default_rng(SEED)

    t0 = time.perf_counter()
    filter_res = model.filter(y_obs, n_particles=N_FILTER, rng=rng_filter, store_history=True)
    elapsed_filter = time.perf_counter() - t0
    assert filter_res.ess_history is not None and filter_res.filtered_means is not None
    ess_history = filter_res.ess_history
    print(f"Bootstrap PF (N={N_FILTER}) in {elapsed_filter:.2f}s   "
          f"loglik = {filter_res.log_likelihood:+.2f}   "
          f"ESS mean/min = {ess_history.mean():.0f}/{ess_history.min():.0f}")

    filtered_h = filter_res.filtered_means
    filtered_vol_plugin = np.exp(filtered_h / 2.0)

    # ---- Step 2: PGAS estimation ------------------------------------------
    print(f"\nRunning PGAS (N={N_PGAS}, iter={ITER_PGAS}, burn={BURN_PGAS}) ...")
    pgas = PGAS(
        model=SVBasicWrapper(),
        prior=SVBasicPrior(loc_mu=mu_plugin, scale_mu=3.0),
        n_particles=N_PGAS,
        n_iterations=ITER_PGAS,
        param_sampler=make_param_sampler(mu_plugin),
        burnin=BURN_PGAS,
        thin=1,
        seed=SEED,
    )
    t0 = time.perf_counter()
    pgas_res = pgas.run(endog=y_obs, theta_init=theta_plugin, verbose=0)
    elapsed_pgas = time.perf_counter() - t0
    post_mean = pgas_res.posterior_mean()
    post_std = pgas_res.posterior_std()
    ci_lo, ci_hi = pgas_res.credible_interval(0.05)
    print(f"PGAS finished in {elapsed_pgas:.1f}s")
    for j, name in enumerate(["mu", "phi", "sigma_h"]):
        print(f"  {name:<8} mean={post_mean[j]:+.4f}  std={post_std[j]:.4f}  "
              f"95% CI=[{ci_lo[j]:+.4f}, {ci_hi[j]:+.4f}]")

    # ---- Step 3: FFBSi smoothing at posterior mean ------------------------
    print("\nRunning FFBSi smoothing at posterior mean ...")
    model_pm = SVBasicWrapper()
    model_pm.set_params(post_mean)
    rng_sm = np.random.default_rng(SEED + 1)
    filter_pm = model_pm.filter(y_obs, n_particles=N_FILTER, rng=rng_sm, store_history=True)
    smoother_model = SmootherModel(mu=float(post_mean[0]),
                                   phi=float(post_mean[1]),
                                   sigma=float(post_mean[2]))
    ffbsi = FFBSi(seed=SEED)
    t0 = time.perf_counter()
    assert filter_pm.filtered_means is not None and filter_pm.filtered_vars is not None
    assert filter_pm.final_particles is not None
    smooth_res = ffbsi.smooth(_SmootherInput(filter_pm, y_obs), smoother_model,
                              n_trajectories=M_TRAJ)
    elapsed_smooth = time.perf_counter() - t0
    smoothed_h = smooth_res.smoothed_mean[:, 0]
    smoothed_vol = np.exp(smoothed_h / 2.0)
    filtered_vol_pm = np.exp(filter_pm.filtered_means / 2.0)
    smoothed_var = smooth_res.smoothed_cov[:, 0, 0]
    filtered_var_pm = filter_pm.filtered_vars
    var_red = (1.0 - smoothed_var.mean() / filtered_var_pm.mean()) * 100.0
    traj = smooth_res.trajectories[:, :, 0]
    smoothed_h_lo = np.quantile(traj, 0.05, axis=0)
    smoothed_h_hi = np.quantile(traj, 0.95, axis=0)
    smoothed_vol_lo = np.exp(smoothed_h_lo / 2.0)
    smoothed_vol_hi = np.exp(smoothed_h_hi / 2.0)
    print(f"FFBSi (M={M_TRAJ}) in {elapsed_smooth:.2f}s   "
          f"variance reduction = {var_red:+.1f}%")

    # ---- Step 4: Volatility forecast --------------------------------------
    print(f"\nForecasting {H_FORECAST} business days ahead ...")
    rng_fc = np.random.default_rng(SEED + 2)
    final_particles = filter_pm.final_particles.copy()
    n_fc = len(final_particles)
    h_paths = np.zeros((n_fc, H_FORECAST))
    h_curr = final_particles.copy()
    mu_pm, phi_pm, sig_pm = float(post_mean[0]), float(post_mean[1]), float(post_mean[2])
    for step in range(H_FORECAST):
        eta = rng_fc.standard_normal(n_fc)
        h_curr = mu_pm + phi_pm * (h_curr - mu_pm) + sig_pm * eta
        h_paths[:, step] = h_curr
    vol_paths = np.exp(h_paths / 2.0)
    q05_h = np.quantile(h_paths, 0.05, axis=0)
    q50_h = np.quantile(h_paths, 0.50, axis=0)
    q95_h = np.quantile(h_paths, 0.95, axis=0)
    q05 = np.quantile(vol_paths, 0.05, axis=0)
    q50 = np.quantile(vol_paths, 0.50, axis=0)
    q95 = np.quantile(vol_paths, 0.95, axis=0)
    future_dates = pd.date_range(
        start=pd.Timestamp(dates[-1]) + pd.Timedelta(days=1),
        periods=H_FORECAST,
        freq="B",
    )

    # ---- Persist outputs --------------------------------------------------
    filtering_df = pd.DataFrame({
        "t": np.arange(T),
        "date": dates,
        "y": y_obs,
        "filtered_h_plugin": filtered_h,
        "filtered_vol_plugin": filtered_vol_plugin,
        "filtered_h_postmean": filter_pm.filtered_means,
        "filtered_vol_postmean": filtered_vol_pm,
        "filtered_var_postmean": filtered_var_pm,
        "smoothed_h": smoothed_h,
        "smoothed_vol": smoothed_vol,
        "smoothed_var": smoothed_var,
        "smoothed_vol_q05": smoothed_vol_lo,
        "smoothed_vol_q95": smoothed_vol_hi,
        "ess_filter_plugin": ess_history,
    })
    out_filtering = HERE / "results_sv_workflow_filtering.csv"
    filtering_df.to_csv(out_filtering, index=False)

    rhat = pgas_res.r_hat() if hasattr(pgas_res, "r_hat") else [np.nan] * 3
    rhat = np.atleast_1d(rhat)

    geweke_z = np.zeros(3)
    geweke_p = np.zeros(3)
    for j in range(3):
        z, p = pgas_res.geweke_test(param_idx=j)
        geweke_z[j] = z
        geweke_p[j] = p

    params_df = pd.DataFrame({
        "param": ["mu", "phi", "sigma_h"],
        "posterior_mean": post_mean,
        "posterior_std": post_std,
        "ci95_lower": ci_lo,
        "ci95_upper": ci_hi,
        "ess": [pgas_res.effective_sample_size(j) for j in range(3)],
        "r_hat": rhat[:3] if len(rhat) >= 3 else [np.nan] * 3,
        "geweke_z": geweke_z,
        "geweke_p": geweke_p,
        "plugin_init": theta_plugin,
    })
    params_df.attrs["acceptance_rate"] = pgas_res.acceptance_rate()
    out_params = HERE / "results_sv_workflow_params.csv"
    params_df.to_csv(out_params, index=False)

    forecast_df = pd.DataFrame({
        "horizon": np.arange(1, H_FORECAST + 1),
        "date": future_dates,
        "h_q05": q05_h,
        "h_q50": q50_h,
        "h_q95": q95_h,
        "vol_q05": q05,
        "vol_q50": q50,
        "vol_q95": q95,
    })
    out_forecast = HERE / "results_sv_workflow_forecast.csv"
    forecast_df.to_csv(out_forecast, index=False)

    print("\n" + "=" * 78)
    print("OUTPUTS")
    print("=" * 78)
    print(f"  {out_filtering}  ({filtering_df.shape[0]} rows x {filtering_df.shape[1]} cols)")
    print(f"  {out_params}     ({params_df.shape[0]} rows x {params_df.shape[1]} cols)")
    print(f"  {out_forecast}    ({forecast_df.shape[0]} rows x {forecast_df.shape[1]} cols)")
    print("\nForecast summary:")
    for h in [1, 5, 20]:
        print(f"  h={h:>2d} days: median vol = {q50[h - 1]:.4f}  "
              f"90% CI = [{q05[h - 1]:.4f}, {q95[h - 1]:.4f}]")
    print(f"\nTotal wall time: "
          f"{elapsed_filter + elapsed_pgas + elapsed_smooth:.1f}s "
          f"(filter {elapsed_filter:.1f} + PGAS {elapsed_pgas:.1f} + FFBSi {elapsed_smooth:.1f})")


if __name__ == "__main__":
    main()
