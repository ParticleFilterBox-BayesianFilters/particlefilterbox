"""Solution script for SV with Jumps (SV-J) and Factor SV.

Pipeline:
    Part 1 — SV with Jumps:
        1. Simulate SV-J data via particlefilterbox.
        2. Filter log-volatility and jump indicators with an Auxiliary PF.
        3. Report jump detection metrics (precision, recall, counts).
    Part 2 — Factor SV:
        1. Simulate a 3-series Factor SV dataset.
        2. Filter common factor with RBPF.
        3. Report RMSE of filtered common factor.

Both parts are persisted to results_sv_jumps_factor.csv with a `section` column.

Run:
    python solution_03_sv_jumps_factor.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from particlefilterbox.models.stochastic_volatility import StochasticVolatility

SEED = 42
T_JUMPS = 500
T_FACTOR = 500
N_APF = 1000
N_RBPF = 500
JUMP_THRESHOLD = 0.3

PARAMS_J = {
    "mu": -1.0,
    "phi": 0.97,
    "sigma": 0.15,
    "lambda_jump": 0.05,
    "mu_jump": -0.5,
    "sigma_jump": 1.0,
}

PARAMS_F = {
    "mu": -1.0,
    "phi": 0.97,
    "sigma": 0.15,
    "phi_0": 0.95,
    "sigma_0": 0.2,
    "beta_0": 1.0,
    "phi_1": 0.90,
    "sigma_1": 0.25,
    "beta_1": 0.8,
    "phi_2": 0.93,
    "sigma_2": 0.18,
    "beta_2": 1.2,
}


def auxiliary_pf_svj(
    y: np.ndarray,
    params: dict[str, float],
    n_particles: int = 1000,
    seed: int = 42,
) -> dict:
    """Auxiliary PF for SV-J: marginal over h, binary jump indicator q."""
    rng = np.random.default_rng(seed)
    T = len(y)
    mu = params["mu"]
    phi = params["phi"]
    sigma = params["sigma"]
    lam = params["lambda_jump"]
    mu_j = params["mu_jump"]
    sigma_j = params["sigma_jump"]

    var_stat = sigma**2 / (1.0 - phi**2)
    h_particles = rng.normal(mu, np.sqrt(max(var_stat, 1e-10)), size=n_particles)
    q_particles = rng.binomial(1, lam, size=n_particles).astype(float)

    filtered_h_mean = np.zeros(T)
    filtered_h_std = np.zeros(T)
    filtered_h_q05 = np.zeros(T)
    filtered_h_q95 = np.zeros(T)
    filtered_q_prob = np.zeros(T)
    log_lik = 0.0

    for t in range(T):
        # Stage 1: predictive adjustment weights via mixture likelihood at h_pred
        if t > 0:
            h_pred = mu + phi * (h_particles - mu)
        else:
            h_pred = h_particles
        vol_pred = np.exp(h_pred / 2.0)
        var_no = vol_pred**2
        var_yes = vol_pred**2 + sigma_j**2
        log_p_no = -0.5 * np.log(2 * np.pi * var_no) - 0.5 * y[t] ** 2 / var_no
        log_p_yes = -0.5 * np.log(2 * np.pi * var_yes) - 0.5 * (y[t] - mu_j) ** 2 / var_yes
        max_p = np.maximum(log_p_no, log_p_yes)
        first_w = (1 - lam) * np.exp(log_p_no - max_p) + lam * np.exp(log_p_yes - max_p)
        first_w_sum = float(np.sum(first_w))
        if first_w_sum < 1e-300:
            first_w = np.ones(n_particles) / n_particles
        else:
            first_w /= first_w_sum

        idx = rng.choice(n_particles, size=n_particles, p=first_w)
        h_particles = h_particles[idx]

        # Stage 2: propagate h, sample jump indicator
        if t > 0:
            h_particles = (
                mu + phi * (h_particles - mu) + sigma * rng.standard_normal(n_particles)
            )
        q_particles = rng.binomial(1, lam, size=n_particles).astype(float)

        vol = np.exp(h_particles / 2.0)
        var_no_2 = vol**2
        var_yes_2 = vol**2 + sigma_j**2
        log_p_actual = np.where(
            q_particles > 0.5,
            -0.5 * np.log(2 * np.pi * var_yes_2) - 0.5 * (y[t] - mu_j) ** 2 / var_yes_2,
            -0.5 * np.log(2 * np.pi * var_no_2) - 0.5 * y[t] ** 2 / var_no_2,
        )
        max_lw = float(np.max(log_p_actual))
        w = np.exp(log_p_actual - max_lw)
        sw = float(np.sum(w))
        if sw < 1e-300:
            w = np.ones(n_particles) / n_particles
            log_lik += -100.0
        else:
            log_lik += max_lw + np.log(sw) - np.log(n_particles)
            w /= sw

        filtered_h_mean[t] = float(np.sum(w * h_particles))
        filtered_h_std[t] = float(
            np.sqrt(np.sum(w * (h_particles - filtered_h_mean[t]) ** 2))
        )
        idx_sort = np.argsort(h_particles)
        cumw = np.cumsum(w[idx_sort])
        filtered_h_q05[t] = float(h_particles[idx_sort][np.searchsorted(cumw, 0.05)])
        q95_idx = int(np.clip(np.searchsorted(cumw, 0.95, side="right"), 0, n_particles - 1))
        filtered_h_q95[t] = float(h_particles[idx_sort][q95_idx])
        filtered_q_prob[t] = float(np.sum(w * q_particles))

        idx2 = rng.choice(n_particles, size=n_particles, p=w)
        h_particles = h_particles[idx2]
        q_particles = q_particles[idx2]

    return {
        "h_mean": filtered_h_mean,
        "h_std": filtered_h_std,
        "h_q05": filtered_h_q05,
        "h_q95": filtered_h_q95,
        "q_prob": filtered_q_prob,
        "log_likelihood": log_lik,
    }


def rbpf_factor_sv(
    y: np.ndarray,
    params: dict[str, float],
    K: int,
    n_particles: int = 500,
    seed: int = 42,
) -> dict:
    """Particle filter for Factor SV targeting the common factor.

    State: (h_common, h_0, h_1, ..., h_{K-1}).
    Emission per series k: N(0, exp(beta_k * h_common + h_k)).
    """
    rng = np.random.default_rng(seed)
    T = y.shape[0]
    mu = params["mu"]
    phi = params["phi"]
    sigma = params["sigma"]

    var_stat = sigma**2 / (1.0 - phi**2)
    h_common = rng.normal(mu, np.sqrt(max(var_stat, 1e-10)), size=n_particles)
    h_idio = np.zeros((n_particles, K))
    for k in range(K):
        phi_k = params[f"phi_{k}"]
        sigma_k = params[f"sigma_{k}"]
        var_k = sigma_k**2 / (1.0 - phi_k**2)
        h_idio[:, k] = rng.normal(0.0, np.sqrt(max(var_k, 1e-10)), size=n_particles)

    filtered_h_common = np.zeros(T)
    filtered_h_common_std = np.zeros(T)
    log_lik = 0.0

    for t in range(T):
        log_w = np.zeros(n_particles)
        for k in range(K):
            beta_k = params[f"beta_{k}"]
            vol_k = np.exp((beta_k * h_common + h_idio[:, k]) / 2.0)
            log_w += -0.5 * np.log(2 * np.pi) - np.log(vol_k) - 0.5 * (y[t, k] / vol_k) ** 2
        max_lw = float(np.max(log_w))
        w = np.exp(log_w - max_lw)
        sw = float(np.sum(w))
        if sw < 1e-300:
            w = np.ones(n_particles) / n_particles
        else:
            log_lik += max_lw + np.log(sw) - np.log(n_particles)
            w /= sw

        filtered_h_common[t] = float(np.sum(w * h_common))
        filtered_h_common_std[t] = float(
            np.sqrt(np.sum(w * (h_common - filtered_h_common[t]) ** 2))
        )

        idx = rng.choice(n_particles, size=n_particles, p=w)
        h_common = h_common[idx]
        h_idio = h_idio[idx]

        h_common = mu + phi * (h_common - mu) + sigma * rng.standard_normal(n_particles)
        for k in range(K):
            phi_k = params[f"phi_{k}"]
            sigma_k = params[f"sigma_{k}"]
            h_idio[:, k] = phi_k * h_idio[:, k] + sigma_k * rng.standard_normal(n_particles)

    return {
        "h_common_mean": filtered_h_common,
        "h_common_std": filtered_h_common_std,
        "log_likelihood": log_lik,
    }


def part_1_svj() -> tuple[pd.DataFrame, dict]:
    print("\n" + "-" * 70)
    print("PART 1: SV with Jumps (SV-J) + Auxiliary PF")
    print("-" * 70)

    sv_jumps = StochasticVolatility(variant="jumps", params=PARAMS_J)
    sim = sv_jumps.simulate(T=T_JUMPS, seed=SEED)
    y = np.asarray(sim["observations"])[:, 0]
    h_true = np.asarray(sim["states"])[:, 0]
    q_true = np.asarray(sim["states"])[:, 1]
    n_jumps_true = int(q_true.sum())
    print(f"Simulated {T_JUMPS} observations with {n_jumps_true} true jumps ({n_jumps_true / T_JUMPS:.2%}).")

    print(f"\nRunning Auxiliary PF (N={N_APF}) ...")
    t0 = time.time()
    apf = auxiliary_pf_svj(y, PARAMS_J, n_particles=N_APF, seed=SEED)
    print(f"  done in {time.time() - t0:.2f}s, log-lik={apf['log_likelihood']:.2f}")

    detected = apf["q_prob"] > JUMP_THRESHOLD
    true_jumps = q_true > 0.5
    tp = int(np.sum(detected & true_jumps))
    fp = int(np.sum(detected & ~true_jumps))
    fn = int(np.sum(~detected & true_jumps))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    rmse_h = float(np.sqrt(np.mean((apf["h_mean"] - h_true) ** 2)))
    print(f"  Jumps detected (P>{JUMP_THRESHOLD}): {int(detected.sum())}  (true={n_jumps_true})")
    print(f"  Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    print(f"  RMSE of filtered h_t vs true: {rmse_h:.4f}")

    jumps_df = pd.DataFrame({
        "section": "svj_filtered",
        "t": np.arange(T_JUMPS),
        "y": y,
        "h_true": h_true,
        "q_true": q_true,
        "h_mean": apf["h_mean"],
        "h_std": apf["h_std"],
        "h_q05": apf["h_q05"],
        "h_q95": apf["h_q95"],
        "q_prob": apf["q_prob"],
        "vol_mean": np.exp(apf["h_mean"] / 2.0),
        "detected": detected.astype(int),
    })

    metrics = {
        "log_likelihood": apf["log_likelihood"],
        "n_jumps_true": n_jumps_true,
        "n_jumps_detected": int(detected.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rmse_h": rmse_h,
    }
    return jumps_df, metrics


def part_2_factor() -> tuple[pd.DataFrame, dict]:
    print("\n" + "-" * 70)
    print("PART 2: Factor SV (K=3) + RBPF")
    print("-" * 70)

    sv_factor = StochasticVolatility(variant="factor", k_factor_series=3, params=PARAMS_F)
    sim = sv_factor.simulate(T=T_FACTOR, seed=SEED + 1)
    y = np.asarray(sim["observations"])  # (T, 3)
    states = np.asarray(sim["states"])    # (T, 4)
    h_common_true = states[:, 0]
    print(f"Simulated {y.shape[0]} x {y.shape[1]} observations.")
    corr_mat = np.corrcoef(y.T)
    print(f"  Return cross-correlations: off-diag mean = {corr_mat[np.triu_indices(3, k=1)].mean():.3f}")

    print(f"\nRunning RBPF on common factor (N={N_RBPF}) ...")
    t0 = time.time()
    rbpf = rbpf_factor_sv(y, PARAMS_F, K=3, n_particles=N_RBPF, seed=SEED)
    print(f"  done in {time.time() - t0:.2f}s, log-lik={rbpf['log_likelihood']:.2f}")
    rmse_common = float(np.sqrt(np.mean((rbpf["h_common_mean"] - h_common_true) ** 2)))
    print(f"  RMSE of common factor h_t: {rmse_common:.4f}")

    factor_df = pd.DataFrame({
        "section": "factor_filtered",
        "t": np.arange(T_FACTOR),
        "y0": y[:, 0],
        "y1": y[:, 1],
        "y2": y[:, 2],
        "h_common_true": h_common_true,
        "h_common_mean": rbpf["h_common_mean"],
        "h_common_std": rbpf["h_common_std"],
        "vol_common_mean": np.exp(rbpf["h_common_mean"] / 2.0),
    })

    metrics = {
        "log_likelihood": rbpf["log_likelihood"],
        "rmse_h_common": rmse_common,
        "mean_offdiag_corr": float(corr_mat[np.triu_indices(3, k=1)].mean()),
    }
    return factor_df, metrics


def main() -> None:
    print("=" * 70)
    print("SOLUTION 03: SV with Jumps + Factor SV")
    print("=" * 70)

    svj_df, svj_metrics = part_1_svj()
    factor_df, factor_metrics = part_2_factor()

    print("\nWriting results_sv_jumps_factor.csv ...")
    params_j_df = pd.DataFrame({
        "section": "svj_parameters",
        "parameter": list(PARAMS_J.keys()),
        "value": list(PARAMS_J.values()),
    })
    params_f_df = pd.DataFrame({
        "section": "factor_parameters",
        "parameter": list(PARAMS_F.keys()),
        "value": list(PARAMS_F.values()),
    })

    svj_metric_rows = pd.DataFrame({
        "section": "svj_metrics",
        "parameter": list(svj_metrics.keys()),
        "value": list(svj_metrics.values()),
    })
    factor_metric_rows = pd.DataFrame({
        "section": "factor_metrics",
        "parameter": list(factor_metrics.keys()),
        "value": list(factor_metrics.values()),
    })

    combined = pd.concat(
        [svj_df, factor_df, params_j_df, params_f_df, svj_metric_rows, factor_metric_rows],
        ignore_index=True,
        sort=False,
    )
    out_path = os.path.join(SCRIPT_DIR, "results_sv_jumps_factor.csv")
    combined.to_csv(out_path, index=False)
    print(f"  saved {len(combined)} rows to {out_path}")

    print("\n--- SV-J Metrics ---")
    print(pd.Series(svj_metrics).to_string())
    print("\n--- Factor SV Metrics ---")
    print(pd.Series(factor_metrics).to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()
