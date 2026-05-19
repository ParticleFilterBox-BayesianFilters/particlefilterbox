---
title: "Troubleshooting"
description: "Common problems in particlefilterbox — low ESS, filter divergence, PMMH acceptance rate, Numba compile errors, GPU OOM, NaN weights, path degeneracy, and how to fix each."
---

# Troubleshooting Guide

Step-by-step solutions for the most common problems encountered running particle filters, SMC, and PMCMC with **particlefilterbox**.

!!! tip "Related pages"
    - **Conceptual questions**: [General FAQ](general.md)
    - **Advanced methods**: [Advanced FAQ](advanced.md)
    - **Diagnostic reference**: [Diagnostics section](../diagnostics/index.md)

---

## Effective Sample Size

??? question "ESS is always very low (< N / 10)"

    Symptoms: `res.ess.mean()` far below $N/2$; filter runs but results are noisy and resampling is triggered every step.

    **Causes, ranked by frequency:**

    1. **Observation noise is tiny compared to state noise.** The transition produces a diffuse particle cloud, but $p(y_t \mid x_t)$ is sharp — most particles land in the tails. Switch to a data-informed proposal:

        ```python
        from particlefilterbox.filters.auxiliary import AuxiliaryFilter
        pf = AuxiliaryFilter(model=model, n_particles=2000)  # default APF
        ```

        If that isn't enough, use [Guided PF](../user-guide/filters/guided.md) or [Locally Optimal PF](../user-guide/filters/locally-optimal.md).

    2. **Not enough particles.** Double $N$ and re-check. If ESS scales linearly with $N$ the fix is just more particles; if it doesn't, the proposal is the problem.

    3. **Model mis-specification.** A wrong transition or likelihood forces the filter to explain data it cannot. Validate by simulating from your model and checking whether its simulated data looks like the real data:

        ```python
        x_sim, y_sim = model.simulate(T=len(y), rng=np.random.default_rng(0))
        # plot y vs y_sim — if they look unrelated, the model is wrong
        ```

    4. **Initial distribution too wide.** Many particles waste their first few steps drifting into the data region. Narrow `model.initial` if you have prior information.

??? question "ESS drops suddenly at one time step"

    Symptom: ESS is healthy (> $N/2$) until $t^*$, where it crashes to near 0.

    This is an **outlier** or a **regime shift** the model cannot accommodate. Plot the log-weights at $t^*$:

    ```python
    import matplotlib.pyplot as plt
    plt.hist(res.log_weights[t_star], bins=50)
    plt.xlabel("log weight")
    ```

    If one particle dominates, that observation is far from all predictions. Options:

    - Use a heavier-tailed observation density (Student-$t$).
    - Add a jump component to the state (see `models.jump_diffusion`).
    - Accept the outlier and monitor — a single low-ESS step rarely breaks estimates, since the filter can recover once the next observation arrives.

## Filter Divergence

??? question "The particle filter diverges (filtered mean drifts away from data)"

    Symptoms: `res.filtered_mean` drifts monotonically; ESS collapses after a few dozen steps.

    Check in order:

    1. **Resampling is disabled or mis-configured.** Adaptive resampling should trigger whenever ESS drops below a threshold (commonly $N/2$):

        ```python
        pf = BootstrapFilter(model=model, n_particles=1000,
                             resample_threshold=0.5)  # fraction of N
        ```

    2. **Weight normalization bug.** If you wrote a custom filter, ensure you normalize weights in **log space** using `logsumexp` before exponentiating:

        ```python
        from scipy.special import logsumexp
        log_w = log_likelihoods - logsumexp(log_likelihoods)
        w = np.exp(log_w)
        ```

    3. **Sign error in `log_likelihood`.** The most common custom-model bug. Compare against a simulated example where you know the answer, or validate against `kalmanbox` on a linear-Gaussian toy.

    4. **Degeneracy** — all particles collapse to one point. See path degeneracy below.

## PMMH Acceptance Rate

??? question "PMMH acceptance rate is extremely low (< 5%)"

    Either the proposal is too aggressive or the log-likelihood estimator is too noisy. Diagnose in this order:

    1. **Measure log-likelihood variance** at the current chain state:

        ```python
        from particlefilterbox.pmcmc.tuning import likelihood_variance
        var_ll = likelihood_variance(model, y, theta_current, N=pf_n, n_repeats=100)
        ```

        - `var_ll > 2` → increase $N$ (target is ~1.0).
        - `var_ll < 0.5` → you have $N$ to spare; shrink to save compute.

    2. **Check proposal scale.** For a Gaussian random-walk on $\theta$, the optimal scale is typically $\sigma_\text{prop} \approx 2.38 / \sqrt{d_\theta} \cdot \hat{\Sigma}^{1/2}$ (Roberts & Rosenthal, 2001). Use adaptive scaling:

        ```python
        from particlefilterbox.pmcmc.pmmh import PMMH
        sampler = PMMH(model=model, n_particles=500, n_iter=20_000,
                       adapt_proposal=True, target_acceptance=0.25)
        ```

    3. **Parameterize on an unconstrained scale.** Sampling `phi` directly hits boundary issues; sample `logit(phi)` instead.

??? question "PMMH acceptance rate is suspiciously high (> 70%)"

    High acceptance usually means **step size too small** — the chain barely moves and mixing is poor. Widen `proposal_scale`. Target ≈ 25% for moderate-dim (2–10) parameters; ≈ 45% for 1-D.

    If acceptance is high **and** chains still mix well (ESS/iter > 0.5), you may have hit a flat posterior region; no action needed.

## Numerical Issues

??? question "I get `NaN` or `Inf` in the weights"

    Most commonly:

    1. **Log-likelihood returned `-inf`** for every particle (e.g. $p(y_t \mid x_t) = 0$ due to a hard boundary). Fix by adding a small regularization or checking the support of your likelihood.

    2. **Arithmetic underflow** from raw (non-log) weights. Always compute in log-space:

        ```python
        # good
        log_w = model.log_likelihood(x, y_t, t)

        # bad — underflows to 0 for any non-trivial likelihood
        w = np.exp(-0.5 * (y_t - x) ** 2 / sigma ** 2)
        ```

    3. **NaN in the state.** Propagation produced `NaN` from a `log`, `sqrt`, or division. Add an assertion:

        ```python
        def transition(self, x_prev, t, rng):
            x_next = ...
            assert np.isfinite(x_next).all(), f"Bad state at t={t}"
            return x_next
        ```

??? question "Results change every run even with a fixed seed"

    Check each:

    1. You are passing `rng=` or `seed=` consistently to **every** stochastic component (model, filter, smoother, PMCMC).
    2. You are not using a GPU backend — CuPy and JAX are **not** bit-reproducible across runs due to non-deterministic reduction order.
    3. You are not calling `np.random.seed(...)` (the global legacy API); always use a `Generator`.
    4. Your custom model does not have a hidden call to `np.random.*` or `random.*`.

    Quick test:

    ```python
    rng = np.random.default_rng(42)
    r1 = BootstrapFilter(model=model, n_particles=1000, rng=rng).filter(y)

    rng = np.random.default_rng(42)
    r2 = BootstrapFilter(model=model, n_particles=1000, rng=rng).filter(y)

    assert np.array_equal(r1.filtered_mean, r2.filtered_mean)
    ```

    If this fails, the reproducibility bug is in the call chain — bisect until you find it.

## Backend Errors

??? question "Numba compilation errors on first run"

    Symptoms: `TypingError: Failed in nopython mode`, slow first call, works on second call (JIT compilation).

    Common causes:

    1. **Unsupported NumPy function.** Numba supports a subset — stick to basic arithmetic, `np.exp`, `np.log`, `np.sqrt`, `np.sum`, `np.sort`. For SciPy calls, keep them outside `@njit` blocks.

    2. **Heterogeneous types.** Numba infers types once; mixing `int32` and `float64` or using `list` where it expects `np.ndarray` will fail.

    3. **Object-mode fallback triggered.** Pass `nopython=True` explicitly to catch this early:

        ```python
        from numba import njit

        @njit(cache=True)
        def propagate(x, phi, sigma):
            return phi * x + sigma * np.random.randn(x.size)
        ```

    4. **First call is slow.** That's expected — compilation happens on the first call. Use `cache=True` to persist to disk.

    When the Numba backend fails, you can always fall back to pure NumPy:

    ```python
    pf = BootstrapFilter(model=model, n_particles=1000, backend="numpy")
    ```

??? question "CuPy `OutOfMemoryError` on GPU"

    Particles, weights, and history all live on the device. Typical fixes:

    1. **Reduce $N$** — the obvious one.
    2. **Free history** — don't store full particle trajectories if only the final state matters:

        ```python
        pf = BootstrapFilter(model=model, n_particles=100_000,
                             backend="cupy", store_history=False)
        ```

    3. **Release the CuPy memory pool after each run:**

        ```python
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()
        ```

    4. **Use mixed precision** — `float32` doubles capacity at the cost of precision:

        ```python
        pf = BootstrapFilter(model=model, n_particles=200_000,
                             backend="cupy", dtype="float32")
        ```

    For very large runs, switch to `JAXBackend` which supports multi-device sharding.

??? question "JAX errors: `ConcretizationTypeError` or `TracerBoolConversionError`"

    JAX requires `jit`-able functions — no Python `if`/`while` on traced values, no in-place mutation. If your model uses conditional logic:

    ```python
    # bad (errors under jit)
    def transition(x, ...):
        if x > 0:
            return 2 * x
        return -x

    # good
    def transition(x, ...):
        return jnp.where(x > 0, 2 * x, -x)
    ```

    If porting is too much work, use the **Numba backend** instead — it imposes far fewer restrictions.

## Smoothing and PMCMC

??? question "Path degeneracy in Particle Gibbs (early times have one or two unique ancestors)"

    Classic PG suffers from **path degeneracy**: looking back from $T$, early time steps are dominated by one ancestral particle. Symptoms:

    - `unique_ancestors_at_t_1 == 1` or 2.
    - Gibbs chain barely moves in $\theta$ blocks tied to early $x$.

    **Fix**: use **PG with Ancestor Sampling (PG-AS)**, which resamples the ancestor of the reference trajectory at each step:

    ```python
    from particlefilterbox.pmcmc.pgas import PGAS
    sampler = PGAS(model=model, n_particles=500, n_iter=10_000)
    ```

    PG-AS typically has orders-of-magnitude better mixing for $T > 200$. See [PG-AS](../user-guide/pmcmc/pgas.md).

??? question "Smoothed estimates are identical to filtered estimates"

    You are likely running **Fixed-Lag smoothing with lag 0** or **FFBSm with a degenerate weight**. Check:

    - `lag > 0` in `FixedLagSmoother`;
    - the backward weights in FFBSm aren't all zero (usually indicates a bug in `log_transition_density`).

## Performance

??? question "The filter is much slower than I expect"

    Work through the list:

    1. **Are you using the default NumPy backend with large $N$?** Switch to Numba for a 10–50× speedup on CPU:

        ```python
        pf = BootstrapFilter(model=model, n_particles=5000, backend="numba")
        ```

    2. **Is your `transition` vectorized?** It must accept an `ndarray` of particle states and return one — not loop over particles in Python.

    3. **Are you storing history you don't need?** `store_history=False` halves memory and can double throughput for long series.

    4. **Are you allocating in the hot loop?** Pre-allocate buffers and overwrite:

        ```python
        # bad
        def transition(x, ..., rng):
            return x + rng.normal(size=x.shape)

        # better for very large N with Numba
        @njit
        def transition(x, out, rng_state):
            for i in range(x.size):
                out[i] = x[i] + rand_normal(rng_state)
            return out
        ```

    See the [Acceleration Benchmarks](../benchmarks/acceleration.md) for expected speedups.

## Still Stuck?

- Open a [GitHub issue](https://github.com/nodesecon/particlefilterbox/issues) with a **minimal reproducing example** (simulate data, no private data).
- Paste `pfbox info` output (version + backend info):

    ```bash
    pfbox info
    ```

- Search the [Diagnostics section](../diagnostics/index.md) — most symptoms have a dedicated diagnostic.
