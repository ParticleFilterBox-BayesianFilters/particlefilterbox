---
title: "Advanced FAQ"
description: "Advanced questions on particlefilterbox — custom models, custom proposals, RBPF with kalmanbox, PMMH vs PGAS, marginal likelihood, model comparison, missing data, high dimension."
---

# Advanced FAQ

Questions beyond the basics — writing your own components, tuning PMCMC, coupling to `kalmanbox`, and scaling to high dimensions.

!!! tip "Newer to the library?"
    Start with the [General FAQ](general.md). For error messages see [Troubleshooting](troubleshooting.md).

---

## Custom Models and Proposals

??? question "How do I implement a fully custom model?"

    Subclass `StateSpaceModel` and implement the three-method interface:

    ```python
    import numpy as np
    from particlefilterbox.core.model import StateSpaceModel
    from scipy import stats

    class StochasticVolatilityJumps(StateSpaceModel):
        """SV model with rare Poisson jumps in the log-variance."""

        def __init__(self, mu, phi, sigma_eta, lambda_jump, sigma_jump):
            self.mu, self.phi, self.sigma_eta = mu, phi, sigma_eta
            self.lambda_jump, self.sigma_jump = lambda_jump, sigma_jump

        def initial(self, n_particles, rng):
            # stationary distribution of AR(1)
            var0 = self.sigma_eta ** 2 / (1 - self.phi ** 2)
            return self.mu + rng.normal(size=n_particles) * np.sqrt(var0)

        def transition(self, x_prev, t, rng):
            n = x_prev.size
            noise = rng.normal(size=n) * self.sigma_eta
            jumps = rng.poisson(self.lambda_jump, size=n) * \
                    rng.normal(size=n) * self.sigma_jump
            return self.mu + self.phi * (x_prev - self.mu) + noise + jumps

        def log_likelihood(self, x, y_t, t):
            # y_t | x_t ~ N(0, exp(x_t))
            return stats.norm.logpdf(y_t, 0.0, np.exp(0.5 * x))
    ```

    Any method that uses `rng` must accept it as a parameter to remain reproducible. The same object flows into all filters, smoothers, and PMCMC samplers.

    See [Models Guide](../user-guide/models/index.md).

??? question "How do I implement a custom proposal (importance density)?"

    The **proposal** $q(x_t \mid x_{t-1}, y_t)$ controls how particles are pushed forward. The Bootstrap PF uses the transition as proposal; the Auxiliary / Guided / Locally-Optimal PF use data-informed proposals.

    Subclass `Proposal` with two methods — `sample` and `log_density`:

    ```python
    from particlefilterbox.core.proposal import Proposal
    import numpy as np

    class LaplaceProposal(Proposal):
        """Laplace approximation around the mode of p(x_t | x_{t-1}, y_t)."""

        def __init__(self, model):
            self.model = model

        def sample(self, x_prev, y_t, t, rng):
            mu_hat, var_hat = self._laplace(x_prev, y_t, t)  # Newton step
            return mu_hat + rng.normal(size=x_prev.shape) * np.sqrt(var_hat)

        def log_density(self, x, x_prev, y_t, t):
            mu_hat, var_hat = self._laplace(x_prev, y_t, t)
            return -0.5 * (x - mu_hat) ** 2 / var_hat - 0.5 * np.log(var_hat)

        def _laplace(self, x_prev, y_t, t):
            ...  # problem-specific Newton/Fisher-scoring step
    ```

    Pass it to `GuidedFilter` or `AuxiliaryFilter`:

    ```python
    from particlefilterbox.filters.guided import GuidedFilter
    pf = GuidedFilter(model=my_model, proposal=LaplaceProposal(my_model),
                      n_particles=1000)
    ```

    A good proposal can cut required $N$ by **10–100×** when the observation is informative.

??? question "When is a custom proposal worth the effort?"

    Build one only when all three are true:

    1. **Weights are degenerate with Bootstrap** — ESS drops below $N/50$ or log-weights span > 20 units.
    2. **The observation is tractable** — you can evaluate $p(y_t \mid x_t)$ in closed form and take its gradient or Fisher information.
    3. **$N$ is a bottleneck** — runtime dominated by particles, not by the model evaluation itself.

    If only (1) holds, start by switching to the [Auxiliary PF](../user-guide/filters/auxiliary.md) with a default pre-weight; it often solves the problem without implementing a new proposal.

## RBPF and Coupling with kalmanbox

??? question "How do I use RBPF with my own linear sub-model?"

    Split the state $x_t = (x^{\text{lin}}_t, x^{\text{nl}}_t)$ so that conditional on the nonlinear path $x^{\text{nl}}_{1:t}$, the linear block is a **linear-Gaussian** state-space model. Provide a `kalmanbox.LinearGaussianSSM` for the linear block:

    ```python
    from kalmanbox import LinearGaussianSSM
    from particlefilterbox.filters.rbpf import RBPF

    def linear_ssm(x_nl_t):
        """Returns a conditionally linear-Gaussian SSM given nonlinear state."""
        return LinearGaussianSSM(
            F=F(x_nl_t), Q=Q(x_nl_t),
            H=H(x_nl_t), R=R(x_nl_t),
        )

    pf = RBPF(
        nonlinear_model=nonlinear_ssm,
        linear_factory=linear_ssm,
        n_particles=500,
    )
    res = pf.filter(y)
    ```

    For each of the $N$ nonlinear particles, `RBPF` runs a full Kalman filter on the linear block, giving **$N$ Gaussian mixtures** as the posterior — a massive variance reduction vs sampling everything.

??? question "How do I validate my particle filter against kalmanbox?"

    Take a truly linear-Gaussian model, run both filters with the same seed on the same data, and check error decay:

    ```python
    from kalmanbox import LocalLevel, LinearGaussianSSM
    from particlefilterbox.diagnostics.kalman_validation import validate_against_kf

    # exact solution
    kf_result = LocalLevel(sigma_eps=1.0, sigma_eta=0.3).fit(y)

    # particle filter must converge to KF as N grows
    errors = validate_against_kf(
        pf_class=BootstrapFilter,
        model=LinearGaussianSSM.local_level(sigma_eps=1.0, sigma_eta=0.3),
        y=y,
        N_grid=[100, 500, 1000, 5000],
        n_repeats=20,
    )

    errors.plot_convergence()   # should show O(N^{-1/2}) slope
    ```

    If the error does **not** decay at the expected rate, the issue is almost always either (a) a resampling bug, (b) a sign error in the log-likelihood, or (c) incorrect handling of the initial distribution.

    See [Kalman Validation diagnostic](../diagnostics/kalman-validation.md).

## PMCMC Methods

??? question "How do I choose between PMMH, PG, and PG-AS?"

    | Algorithm | Shines when... | Avoid when... |
    |:----------|:--------------|:--------------|
    | **PMMH** | Few parameters, no conjugate block, irregular likelihood | You need joint parameter + state draws without storing full PF history |
    | **Particle Gibbs** | Conjugate priors for some $\theta$-block; moderate $T$ | $T \gg 500$ — path degeneracy kills mixing |
    | **PG-AS** | Long $T$, path degeneracy in PG | Model has no analytic backward kernel |

    Decision tree:

    ```text
    Is T larger than ~500?
      ├── Yes → PG-AS
      └── No  → Do you have conjugate blocks (e.g. Gaussian AR(1) params)?
                  ├── Yes → Particle Gibbs
                  └── No  → PMMH
    ```

    Always run **two chains from random starts**, check $\hat{R} < 1.01$ and effective sample size > 200 per parameter before trusting results. See [MCMC Convergence](../diagnostics/mcmc-convergence.md).

??? question "How do I tune the number of particles $N$ in PMMH?"

    The variance of the log-likelihood estimator controls acceptance rate. Target:

    $$
    \mathrm{Var}\bigl[\log \hat{p}(y_{1:T} \mid \theta)\bigr] \approx 1
    $$

    Procedure:

    1. Pick a plausible $\theta^*$ (e.g. posterior mean from a pilot run).
    2. Run the filter $R=100$ times with different seeds and compute the variance of $\log \hat{p}$.
    3. If variance > 2, double $N$; if < 0.5, halve it.

    ```python
    from particlefilterbox.pmcmc.tuning import tune_n_particles

    N_opt = tune_n_particles(
        model=model, y=y, theta=theta_hat, N_grid=[200, 500, 1000, 2000],
        target_var=1.0, n_repeats=100,
    )
    print(f"Use N = {N_opt}")
    ```

    Doucet, Pitt, Deligiannidis & Kohn (2015) show this is near-optimal for total compute per effective sample.

??? question "How do I estimate marginal likelihood for model comparison?"

    Three options, in increasing cost and accuracy:

    1. **Single PF run** — unbiased but high-variance:

        ```python
        res = BootstrapFilter(model=M1, n_particles=2000).filter(y)
        logZ = res.log_marginal_likelihood
        ```

    2. **SMC² / IBIS** — sequential Monte Carlo over the parameter space gives $\log Z$ with quantified variance:

        ```python
        from particlefilterbox.smc.smc_squared import SMC2
        smc = SMC2(model=M1, prior=prior, n_theta=500, n_x=500).run(y)
        print(smc.log_marginal_likelihood, smc.log_marginal_se)
        ```

    3. **Thermodynamic integration** via tempered SMC — most robust, highest cost.

    For a Bayes factor between models $M_1$ and $M_2$:

    $$
    \log \mathrm{BF}_{12} = \log \hat{Z}_1 - \log \hat{Z}_2
    $$

    Average $\log Z$ across ≥10 independent PF runs for stable estimates. See [Marginal Likelihood diagnostic](../diagnostics/marginal-likelihood.md).

??? question "How do I do formal model comparison?"

    Combine marginal-likelihood estimates with diagnostic checks:

    ```python
    from particlefilterbox.reports import ModelComparisonReport

    report = ModelComparisonReport(
        models={"SV": sv, "SV-Jumps": svj, "Regime-SV": rsv},
        data=y,
        n_particles=2000,
        n_replicates=10,   # for log-Z uncertainty
    )
    report.summary()
    ```

    The report returns log marginal likelihoods with standard errors, PPCs (posterior predictive checks), and DIC / WAIC. A Bayes factor > $10^2$ is "decisive" (Kass & Raftery 1995).

## Data Issues

??? question "How do I handle missing observations?"

    Use `np.nan` in the observation array — filters automatically skip the weighting step at those times, propagating the prior predictive instead. For structurally missing components in multivariate $y_t$, override `log_likelihood` to mask missing entries:

    ```python
    def log_likelihood(self, x, y_t, t):
        mask = ~np.isnan(y_t)
        if not mask.any():
            return np.zeros(x.shape[0])
        resid = y_t[mask] - self.H[mask] @ x
        return stats.multivariate_normal.logpdf(resid, cov=self.R[mask][:, mask])
    ```

    The log-likelihood of the full data is unaffected by missing entries and still unbiased.

??? question "How do I handle outliers or heavy-tailed observations?"

    Three options:

    1. **Use a heavy-tailed observation density**: swap Gaussian for Student-$t$ or Laplace in `log_likelihood`. This is the cleanest statistical fix.
    2. **Robust re-weighting** — Huberize log-weights:

        ```python
        from particlefilterbox.core.weights import huber_clip
        pf = BootstrapFilter(model=model, n_particles=1000,
                             weight_clip=huber_clip(threshold=5.0))
        ```

    3. **Outlier detection in post-processing** — flag $t$ where ESS < $N/100$ as candidate outliers.

    Prefer (1) — changing the model is more defensible than patching weights.

## Scaling

??? function "How do I scale to high-dimensional state-spaces?"

    Plain particle filters suffer the **curse of dimensionality** — required $N$ grows exponentially with state dimension $d$. Practical remedies:

    - **Factorize** — exploit conditional independence in the state to apply local particle filters (block PF, localized EnKF).
    - **Rao-Blackwellize** — if any sub-block is conditionally linear-Gaussian, use [RBPF](../user-guide/filters/rbpf.md). Removes the "Gaussian" dimensions from the particle count.
    - **Ensemble PF** — for $d > 50$, switch to the [Ensemble PF](../user-guide/filters/ensemble.md), which uses localization and inflation.
    - **Auxiliary SMC samplers** — for static high-dim problems, [Waste-Free SMC](../user-guide/smc/waste-free.md) with tempering.

    Realistic ceilings on a CPU:

    | Approach | Max $d$ (routinely) |
    |:---------|:-------------------:|
    | Bootstrap PF | 3–5 |
    | Auxiliary / Guided PF | 5–10 |
    | Unscented PF | 10–20 |
    | RBPF (with $d_\text{linear}$ large) | 30–50 |
    | Ensemble PF (localized) | 100+ |

??? question "How do I parallelize across particles?"

    Two axes — intra-filter (particles) and inter-chain (independent PMMH runs):

    === "Particles (Numba)"

        ```python
        from particlefilterbox.acceleration.numba import NumbaBackend
        pf = BootstrapFilter(model=model, n_particles=10_000,
                             backend=NumbaBackend(parallel=True))
        ```

    === "Particles (GPU)"

        ```python
        from particlefilterbox.acceleration.gpu import CuPyBackend
        pf = BootstrapFilter(model=model, n_particles=100_000,
                             backend=CuPyBackend())
        ```

    === "Chains"

        ```python
        from particlefilterbox.pmcmc.pmmh import run_parallel_chains
        chains = run_parallel_chains(
            sampler_factory=lambda: PMMH(model=model, n_particles=500, n_iter=20_000),
            n_chains=4, data=y,
        )
        ```

    See [Acceleration Guide](../acceleration/index.md) for backend details and [Benchmarks](../benchmarks/acceleration.md) for measured speedups.

---

## See Also

- [Troubleshooting](troubleshooting.md) — specific errors and ESS / weight diagnostics.
- [Tuning PMCMC](../user-guide/pmcmc/tuning.md) — full guide to tuning $N$, step sizes, acceptance rates.
- [PMCMC Benchmarks](../benchmarks/pmcmc.md) — measured ESS/second across algorithms.
