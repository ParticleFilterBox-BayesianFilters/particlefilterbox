#!/usr/bin/env python3
"""End-to-end verification script for FASE 2.

Runs all critical tests and reports results.
Usage: python -m tests.filters.verify_fase2
"""

from __future__ import annotations

import sys
import time

import numpy as np


def run_verification() -> bool:
    """Run all verification checks.

    Returns True if all checks pass.
    """
    from particlefilterbox.core.config import PFConfig
    from particlefilterbox.filters.bootstrap import BootstrapPF
    from particlefilterbox.filters.sir import SIR
    from tests.filters.conftest import (
        LinearGaussianModel,
        StochasticVolatilityModel,
        kalman_filter,
    )

    all_passed = True
    results_summary: list[tuple[str, bool, str]] = []

    print("=" * 70)
    print("FASE 2 - End-to-End Verification")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Test 1: Bootstrap PF on Linear Gaussian
    # -----------------------------------------------------------------------
    print("\n[1/6] Bootstrap PF - Linear Gaussian convergence...")
    t0 = time.time()

    rng = np.random.default_rng(42)
    model_lg = LinearGaussianModel()
    states, obs = model_lg.simulate(n_steps=200, rng=rng)

    kf_means, kf_vars, kf_ll = kalman_filter(
        obs,
        phi=model_lg.phi,
        sigma_eta=model_lg.sigma_eta,
        sigma_eps=model_lg.sigma_eps,
    )

    config = PFConfig(n_particles=5000, seed=123, ess_threshold=0.5)
    pf = BootstrapPF(model_lg, config)  # type: ignore[arg-type]
    pf_results = pf.filter(obs)

    pf_means = pf_results.filtered_means[:, 0]
    corr = float(np.corrcoef(pf_means, kf_means)[0, 1])
    ll_diff = abs(pf_results.log_likelihood - kf_ll)

    check1 = corr > 0.99
    check2 = ll_diff < 2.0
    dt = time.time() - t0

    msg1 = f"corr={corr:.4f} (>0.99)"
    msg2 = f"|ll_diff|={ll_diff:.4f} (<2.0)"
    results_summary.append(("Bootstrap corr vs Kalman", check1, msg1))
    results_summary.append(("Bootstrap ll vs Kalman", check2, msg2))
    print(f"  corr = {corr:.4f} {'PASS' if check1 else 'FAIL'}")
    print(f"  |ll_diff| = {ll_diff:.4f} {'PASS' if check2 else 'FAIL'}")
    print(f"  Time: {dt:.2f}s")
    all_passed = all_passed and check1 and check2

    # -----------------------------------------------------------------------
    # Test 2: SIR = Bootstrap (fallback)
    # -----------------------------------------------------------------------
    print("\n[2/6] SIR (bootstrap fallback) matches BootstrapPF...")
    t0 = time.time()

    config_b = PFConfig(n_particles=500, seed=42, ess_threshold=0.5)
    config_s = PFConfig(n_particles=500, seed=42, ess_threshold=0.5)
    pf_b = BootstrapPF(model_lg, config_b)  # type: ignore[arg-type]
    pf_s = SIR(model_lg, config_s)  # type: ignore[arg-type]

    res_b = pf_b.filter(obs)
    res_s = pf_s.filter(obs)

    means_match = bool(
        np.allclose(res_b.filtered_means, res_s.filtered_means, rtol=1e-10)
    )
    ll_match = bool(
        np.isclose(res_b.log_likelihood, res_s.log_likelihood, rtol=1e-10)
    )
    check3 = means_match and ll_match
    dt = time.time() - t0

    results_summary.append((
        "SIR = Bootstrap (fallback)",
        check3,
        f"means_match={means_match}, ll_match={ll_match}",
    ))
    print(
        f"  means match: {means_match}, ll match: {ll_match} "
        f"{'PASS' if check3 else 'FAIL'}"
    )
    print(f"  Time: {dt:.2f}s")
    all_passed = all_passed and check3

    # -----------------------------------------------------------------------
    # Test 3: Missing data
    # -----------------------------------------------------------------------
    print("\n[3/6] Missing data handling...")
    t0 = time.time()

    obs_missing = obs.copy()
    obs_missing[[10, 20, 30, 50]] = np.nan

    config_m = PFConfig(n_particles=1000, seed=42, ess_threshold=0.5)
    pf_m = BootstrapPF(model_lg, config_m)  # type: ignore[arg-type]
    res_m = pf_m.filter(obs_missing)

    check4 = bool(
        np.isfinite(res_m.log_likelihood)
        and np.all(res_m.log_likelihoods[[10, 20, 30, 50]] == 0.0)
        and np.all(np.isfinite(res_m.filtered_means))
    )
    dt = time.time() - t0

    results_summary.append((
        "Missing data",
        check4,
        f"ll_finite={np.isfinite(res_m.log_likelihood)}",
    ))
    print(f"  {'PASS' if check4 else 'FAIL'}")
    print(f"  Time: {dt:.2f}s")
    all_passed = all_passed and check4

    # -----------------------------------------------------------------------
    # Test 4: Reproducibility
    # -----------------------------------------------------------------------
    print("\n[4/6] Reproducibility (same seed)...")
    t0 = time.time()

    config_r1 = PFConfig(n_particles=500, seed=99, ess_threshold=0.5)
    config_r2 = PFConfig(n_particles=500, seed=99, ess_threshold=0.5)

    res_r1 = BootstrapPF(model_lg, config_r1).filter(obs)  # type: ignore[arg-type]
    res_r2 = BootstrapPF(model_lg, config_r2).filter(obs)  # type: ignore[arg-type]

    check5 = bool(
        np.array_equal(res_r1.filtered_means, res_r2.filtered_means)
        and res_r1.log_likelihood == res_r2.log_likelihood
    )
    dt = time.time() - t0

    results_summary.append(("Reproducibility", check5, ""))
    print(f"  {'PASS' if check5 else 'FAIL'}")
    print(f"  Time: {dt:.2f}s")
    all_passed = all_passed and check5

    # -----------------------------------------------------------------------
    # Test 5: SV model
    # -----------------------------------------------------------------------
    print("\n[5/6] SV model - Bootstrap PF tracking...")
    t0 = time.time()

    rng_sv = np.random.default_rng(100)
    model_sv = StochasticVolatilityModel()
    h_true, obs_sv = model_sv.simulate(n_steps=500, rng=rng_sv)

    config_sv = PFConfig(n_particles=2000, seed=123, ess_threshold=0.5)
    pf_sv = BootstrapPF(model_sv, config_sv)  # type: ignore[arg-type]
    res_sv = pf_sv.filter(obs_sv)

    sv_corr = float(np.corrcoef(res_sv.filtered_means[:, 0], h_true)[0, 1])
    check6 = sv_corr > 0.7
    dt = time.time() - t0

    results_summary.append(("SV tracking", check6, f"corr={sv_corr:.4f} (>0.7)"))
    print(f"  corr = {sv_corr:.4f} {'PASS' if check6 else 'FAIL'}")
    print(f"  Time: {dt:.2f}s")
    all_passed = all_passed and check6

    # -----------------------------------------------------------------------
    # Test 6: Step-by-step = Batch
    # -----------------------------------------------------------------------
    print("\n[6/6] Step-by-step consistency...")
    t0 = time.time()

    config_batch = PFConfig(n_particles=500, seed=42, ess_threshold=0.5)
    config_step = PFConfig(n_particles=500, seed=42, ess_threshold=0.5)

    pf_batch = BootstrapPF(model_lg, config_batch)  # type: ignore[arg-type]
    res_batch = pf_batch.filter(obs)

    pf_step = BootstrapPF(model_lg, config_step)  # type: ignore[arg-type]
    rng_step = pf_step._get_rng()
    cloud = pf_step.initialize(rng_step)
    step_ll = 0.0

    for t in range(len(obs)):
        y_t = np.atleast_1d(obs[t])
        cloud, ll_t = pf_step.filter_step(cloud, y_t, t)
        step_ll += ll_t

    check7 = bool(np.isclose(step_ll, res_batch.log_likelihood, rtol=1e-10))
    dt = time.time() - t0

    results_summary.append((
        "Step-by-step = Batch",
        check7,
        f"step_ll={step_ll:.4f}, batch_ll={res_batch.log_likelihood:.4f}",
    ))
    print(f"  {'PASS' if check7 else 'FAIL'}")
    print(f"  Time: {dt:.2f}s")
    all_passed = all_passed and check7

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed, detail in results_summary:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {detail}")

    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
