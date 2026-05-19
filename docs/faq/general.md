---
title: "General FAQ"
description: "Frequently asked questions about particlefilterbox — what particle filters do, when to use them, choosing N, integration with kalmanbox, and getting started."
---

# General FAQ

Common questions for new and intermediate users of **particlefilterbox**.

!!! tip "Looking for more?"
    - **Advanced methods** (custom models, RBPF, PMCMC tuning): [Advanced FAQ](advanced.md)
    - **Error messages and debugging**: [Troubleshooting](troubleshooting.md)
    - **Performance**: [Benchmarks](../benchmarks/index.md)

---

## Fundamentals

??? question "What is a particle filter?"

    A **particle filter** (also called **Sequential Monte Carlo**, SMC) is a simulation-based algorithm that approximates the filtering distribution $p(x_t \mid y_{1:t})$ of a state-space model by a cloud of $N$ weighted samples (**particles**):

    $$
    p(x_t \mid y_{1:t}) \approx \sum_{i=1}^{N} w_t^{(i)} \, \delta_{x_t^{(i)}}(x_t), \qquad \sum_{i=1}^{N} w_t^{(i)} = 1
    $$

    At each time step the filter:

    1. **Propagates** each particle through the transition density $p(x_t \mid x_{t-1}^{(i)})$,
    2. **Weights** it by the observation density $p(y_t \mid x_t^{(i)})$,
    3. **Resamples** the cloud to kill low-weight particles and duplicate high-weight ones.

    Unlike the Kalman filter, this procedure does **not** require linearity or Gaussianity — any model you can simulate from and any likelihood you can evaluate works.

??? question "When should I use particlefilterbox vs kalmanbox?"

    Use the table below as a starting rule:

    | Situation | Use |
    |:----------|:---:|
    | Linear transitions + Gaussian noise | **kalmanbox** |
    | Nonlinear transitions (e.g. SV, DSGE, jump-diffusion) | **particlefilterbox** |
    | Non-Gaussian observation noise (Student-$t$, Poisson, etc.) | **particlefilterbox** |
    | Discrete latent states (regime switching with continuous shocks) | **particlefilterbox** |
    | Mixed linear/nonlinear blocks | **RBPF** (both) |
    | Panel of linear-Gaussian units | **kalmanbox** or **panelbox** |

    Rule of thumb: **start with kalmanbox**. If you need the EKF or UKF because the linearization is inaccurate, switch to `particlefilterbox`. If only part of the state is nonlinear, use the [Rao-Blackwellized PF](../user-guide/filters/rbpf.md), which calls into `kalmanbox` for the linear sub-state.

??? question "How many particles do I need?"

    There is no universal answer, but reasonable starting points are:

    | Task | Typical $N$ |
    |:-----|:-----------:|
    | Quick prototype / debugging | 200–500 |
    | State estimation (SV, 1-D) | 1 000 |
    | State estimation (5–10-D) | 2 000–10 000 |
    | Log-likelihood for PMMH | 500–2 000 |
    | High-dimensional DSGE | 10 000+ |
    | Publishable empirical results | ≥ 5 000 with ESS > $N/2$ |

    The right diagnostic is the **Effective Sample Size** (ESS), not $N$:

    ```python
    from particlefilterbox.filters.bootstrap import BootstrapFilter

    pf = BootstrapFilter(model=model, n_particles=1000)
    res = pf.filter(y)
    print(f"Mean ESS: {res.ess.mean():.0f} / 1000")
    print(f"Min ESS:  {res.ess.min():.0f}")
    ```

    If $\text{ESS} < N/10$ at any time, either increase $N$, switch to a better proposal (Guided / Locally Optimal / Auxiliary), or tune the resampling threshold.

??? question "Which filter should I use?"

    The short guide:

    - **Default**: [Bootstrap Particle Filter](../user-guide/filters/bootstrap.md). Always works, easy to implement, competitive when the observation noise is large relative to the state noise.
    - **Strong / informative observations**: [Auxiliary PF](../user-guide/filters/auxiliary.md) or [Guided PF](../user-guide/filters/guided.md) — look-ahead proposals avoid weight collapse.
    - **Mixed linear-nonlinear** (conditionally linear given part of the state): [Rao-Blackwellized PF](../user-guide/filters/rbpf.md) — exact Kalman update for the linear block, particles only for the nonlinear part. Huge variance reduction.
    - **Mildly nonlinear, unimodal posterior**: [Unscented PF](../user-guide/filters/upf.md).
    - **Very diffuse or continuous state with few particles**: [Regularized PF](../user-guide/filters/regularized.md) — KDE-based resampling.
    - **Weather / geoscience-style high dimension**: [Ensemble PF](../user-guide/filters/ensemble.md).

    See the [Choosing a Filter guide](../getting-started/choosing-filter.md) for a decision tree.

## Installation and Setup

??? question "How do I install particlefilterbox?"

    ```bash
    pip install particlefilterbox
    ```

    With optional extras:

    ```bash
    # visualization (matplotlib + seaborn)
    pip install particlefilterbox[viz]

    # CLI
    pip install particlefilterbox[cli]

    # GPU acceleration (CuPy)
    pip install particlefilterbox[gpu]

    # JAX backend
    pip install particlefilterbox[jax]

    # Everything
    pip install particlefilterbox[all]
    ```

    Verify the installation:

    ```python
    import particlefilterbox as pfb
    print(pfb.__version__)
    ```

??? question "Which Python versions are supported?"

    **Python 3.10 and later**. Python 3.12 is recommended for the best NumPy 2 / Numba compatibility.

??? question "Do I need a GPU?"

    **No.** particlefilterbox runs on pure CPU NumPy by default and is fast enough for most applied work — a Bootstrap PF with $N=1\,000$ runs in milliseconds per step. You only benefit from a GPU when:

    - $N \geq 10\,000$ *and* the model is dominated by per-particle arithmetic (not resampling), or
    - You run many parallel chains (SMC², PMMH with multiple restarts, bootstrap CIs).

    Use [Numba JIT](../acceleration/numba.md) as the first acceleration step — it typically gives 10–50× speedups on CPU with zero code changes.

??? question "Do I need kalmanbox installed?"

    Yes — [kalmanbox](https://github.com/nodesecon/kalmanbox) is a required dependency. It provides:

    - Kalman Filter / Smoother primitives used internally by the [RBPF](../user-guide/filters/rbpf.md),
    - The `LinearGaussianSSM` class that RBPF delegates to,
    - Baseline diagnostics (Kalman validation) for linear-Gaussian sub-problems.

    `pip install particlefilterbox` installs `kalmanbox` automatically.

## Data and Models

??? question "What data format does particlefilterbox expect?"

    A **1-D NumPy array** for univariate observations, or a **2-D array of shape `(T, d_y)`** for multivariate observations:

    ```python
    import numpy as np
    from particlefilterbox.models.sv import SVModel
    from particlefilterbox.filters.bootstrap import BootstrapFilter

    y = np.loadtxt("returns.csv")    # shape (T,)
    model = SVModel(mu=0.0, phi=0.97, sigma_eta=0.15)
    res = BootstrapFilter(model=model, n_particles=1000).filter(y)
    ```

    For pandas DataFrames, pass the column as an array:

    ```python
    import pandas as pd
    df = pd.read_csv("returns.csv", parse_dates=["date"])
    y = df["log_return"].to_numpy()
    ```

??? question "Which models come built-in?"

    | Model | Import | Typical use |
    |:------|:-------|:-----------|
    | Stochastic Volatility | `particlefilterbox.models.sv.SVModel` | Financial returns |
    | Linearized DSGE | `particlefilterbox.models.dsge.DSGEModel` | Macro / monetary policy |
    | Jump-Diffusion | `particlefilterbox.models.jump_diffusion.JumpDiffusion` | Asset prices with shocks |
    | Markov Switching | `particlefilterbox.models.regime.RegimeSwitching` | Business-cycle / volatility regimes |
    | Poisson / NB Count | `particlefilterbox.models.count.CountModel` | Integer-valued time series |
    | Bounded state-space | `particlefilterbox.models.bounded.BoundedModel` | Unit intervals, shares |
    | Mixture observation | `particlefilterbox.models.mixture.MixtureModel` | Multimodal emissions |
    | Continuous-time SDE | `particlefilterbox.models.continuous_time.SDEModel` | Euler-discretized diffusions |

    See the [Models Guide](../user-guide/models/index.md) for details.

??? question "Can I use particlefilterbox with real-time / streaming data?"

    Yes. All filters expose a **step API** for online updates:

    ```python
    from particlefilterbox.filters.bootstrap import BootstrapFilter

    pf = BootstrapFilter(model=model, n_particles=1000)
    pf.initialize()                     # draw initial cloud

    for y_t in stream:                  # one observation at a time
        pf.step(y_t)
        estimate = pf.state.filtered_mean()
        yield estimate
    ```

    This is $O(N)$ per observation and does not store the full trajectory by default. For online **parameter** learning, use [SMC² Online](../user-guide/pmcmc/smc2-online.md) or [IBIS](../user-guide/smc/ibis.md).

??? question "Can I handle missing observations?"

    Yes — pass `np.nan` in the observation array. All filters skip the weighting step for missing $y_t$ while still propagating particles:

    ```python
    y = np.array([1.0, 0.5, np.nan, 0.3])  # y_3 is missing
    res = BootstrapFilter(model=model, n_particles=1000).filter(y)
    ```

    The filtered mean at $t=3$ is the **prior predictive** mean $\mathbb{E}[x_3 \mid y_{1:2}]$.

??? question "How do I define my own model?"

    Subclass `StateSpaceModel` and implement three methods:

    ```python
    from particlefilterbox.core.model import StateSpaceModel
    import numpy as np

    class MyModel(StateSpaceModel):
        def __init__(self, theta):
            self.theta = theta

        def initial(self, n_particles, rng):
            return rng.normal(0.0, 1.0, size=n_particles)

        def transition(self, x_prev, t, rng):
            return self.theta * x_prev + rng.normal(size=x_prev.shape)

        def log_likelihood(self, x, y_t, t):
            return -0.5 * (y_t - x) ** 2
    ```

    See the [Custom Models tutorial](../user-guide/models/index.md) for the full interface.

## Estimation and Inference

??? question "How do I estimate parameters, not just states?"

    particlefilterbox provides five PMCMC / SMC approaches:

    | Method | When to use | Import |
    |:-------|:------------|:-------|
    | **PMMH** | Static parameters, any model | `particlefilterbox.pmcmc.pmmh.PMMH` |
    | **Particle Gibbs** | Conjugate block structure | `particlefilterbox.pmcmc.particle_gibbs.ParticleGibbs` |
    | **PG-AS** | Long time series, path degeneracy in PG | `particlefilterbox.pmcmc.pgas.PGAS` |
    | **SMC²** | Online learning, sequential Bayes | `particlefilterbox.smc.smc_squared.SMC2` |
    | **IBIS** | Static model, sequential data | `particlefilterbox.smc.ibis.IBIS` |

    Start with [PMMH](../user-guide/pmcmc/pmmh.md) — it is the most general and the easiest to tune.

??? question "How does particlefilterbox integrate with kalmanbox?"

    Three integration points:

    1. **RBPF** — delegates the linear sub-state to `kalmanbox.LinearGaussianSSM.filter` and samples only the nonlinear block.
    2. **Kalman validation** — run your particle filter against a linear-Gaussian baseline using [`diagnostics.kalman_validation`](../diagnostics/kalman-validation.md); errors between PF and KF on the *same* linear-Gaussian model should scale as $\mathcal{O}(N^{-1/2})$.
    3. **Warm-starts** — use a kalmanbox Kalman Smoother to generate a deterministic proposal (locally-optimal) for the initial SMC pass.

    See the [RBPF tutorial](../tutorials/rbpf.md).

??? question "Can I compute log-marginal likelihood for model comparison?"

    Yes — any particle filter gives an **unbiased** estimate of the marginal likelihood via the product of the normalizing constants:

    $$
    \hat{p}(y_{1:T}) = \prod_{t=1}^{T} \frac{1}{N} \sum_{i=1}^{N} \tilde{w}_t^{(i)}
    $$

    ```python
    res = BootstrapFilter(model=model, n_particles=2000).filter(y)
    print(f"log p(y): {res.log_marginal_likelihood:.3f}")
    ```

    Compute for each competing model, then take the log-Bayes factor. See [Marginal Likelihood diagnostic](../diagnostics/marginal-likelihood.md).

## Performance and Reproducibility

??? question "How do I make runs reproducible?"

    Pass a `seed` or `numpy.random.Generator`:

    ```python
    import numpy as np
    rng = np.random.default_rng(42)

    pf = BootstrapFilter(model=model, n_particles=1000, rng=rng)
    ```

    All filters, smoothers, SMC, and PMCMC samplers accept `rng=` or `seed=`. With a fixed seed and fixed backend (CPU NumPy), results are bitwise reproducible. GPU backends are **not** bitwise reproducible because reduction order varies.

??? question "Where can I learn more about the theory?"

    - [Theory section](../theory/index.md) — mathematical foundations with proofs.
    - [Tutorials](../tutorials/index.md) — worked examples from Bootstrap PF up to PG-AS on DSGE.
    - Chopin & Papaspiliopoulos (2020), *An Introduction to Sequential Monte Carlo*, Springer — the canonical reference.
    - Doucet & Johansen (2011), *A tutorial on particle filtering and smoothing: Fifteen years later*.

---

## See Also

- [Advanced FAQ](advanced.md) — custom proposals, PMMH tuning, marginal likelihood.
- [Troubleshooting](troubleshooting.md) — common errors and diagnostics.
- [Benchmarks](../benchmarks/index.md) — performance comparisons across filters and backends.
