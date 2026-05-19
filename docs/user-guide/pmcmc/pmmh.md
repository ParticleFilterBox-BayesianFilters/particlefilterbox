---
title: Particle Marginal Metropolis-Hastings (PMMH)
description: PMMH algorithm -- using particle filter likelihood estimates inside Metropolis-Hastings for Bayesian parameter estimation
---

# Particle Marginal Metropolis-Hastings (PMMH)

PMMH is the simplest PMCMC method. It uses a particle filter as a **black-box likelihood estimator** inside a Metropolis-Hastings sampler. The particle filter integrates out the latent states, so the MCMC chain operates only in the parameter space $\theta$.

---

## The Algorithm

At each MCMC iteration, PMMH performs three steps:

1. **Propose** new parameters $\theta'$ from a proposal distribution $q(\theta' \mid \theta^{(m-1)})$
2. **Estimate** the marginal likelihood $\hat{p}(y_{1:T} \mid \theta')$ by running a particle filter with $N$ particles
3. **Accept or reject** using the standard Metropolis-Hastings ratio

### Pseudocode

$$
\boxed{
\begin{aligned}
&\textbf{Algorithm: PMMH} \\
&\textbf{Input:} \text{ observations } y_{1:T}, \text{ prior } p(\theta), \text{ number of particles } N \\
&\text{1. Initialize } \theta^{(0)}, \text{ run PF}(y_{1:T}, \theta^{(0)}, N) \to \hat{p}_0 \\
&\text{2. For } m = 1, \ldots, M: \\
&\quad \text{a. Draw } \theta' \sim q(\theta' \mid \theta^{(m-1)}) \\
&\quad \text{b. Run PF}(y_{1:T}, \theta', N) \to \hat{p}' \\
&\quad \text{c. Compute acceptance probability:} \\
&\quad\quad \alpha = \min\!\left(1, \;\frac{\hat{p}' \cdot p(\theta')}{\hat{p}_{m-1} \cdot p(\theta^{(m-1)})} \cdot \frac{q(\theta^{(m-1)} \mid \theta')}{q(\theta' \mid \theta^{(m-1)})}\right) \\
&\quad \text{d. Draw } u \sim \text{Uniform}(0, 1) \\
&\quad \text{e. If } u < \alpha: \text{ set } \theta^{(m)} = \theta', \; \hat{p}_m = \hat{p}' \\
&\quad \quad \text{Else: set } \theta^{(m)} = \theta^{(m-1)}, \; \hat{p}_m = \hat{p}_{m-1} \\
&\text{3. Return } \{\theta^{(m)}\}_{m=1}^{M}
\end{aligned}
}
$$

### Why It Works

The particle filter estimate $\hat{p}(y_{1:T} \mid \theta)$ is **unbiased**: on average, it equals the true marginal likelihood. Andrieu et al. (2010) proved that substituting an unbiased estimator into the MH ratio yields a chain that targets the **exact** posterior $p(\theta \mid y_{1:T})$ -- not an approximation.

This is the **pseudo-marginal** property. It holds regardless of how noisy the likelihood estimate is, although higher variance estimates lead to lower acceptance rates and poorer mixing.

!!! note "The variance-acceptance tradeoff"
    The variance of $\log \hat{p}(y_{1:T} \mid \theta)$ controls the efficiency of the chain:

    - **Low variance** (many particles): high acceptance rate, but each iteration is expensive
    - **High variance** (few particles): low acceptance rate, chain gets stuck

    The optimal tradeoff is when $\text{Var}[\log \hat{p}] \approx 1$--$3$, which typically
    corresponds to an acceptance rate of **15--30%** (Doucet et al., 2015).

---

## API

```python
from particlefilterbox.pmcmc import PMMH

pmmh = PMMH(
    model,                    # StateSpaceModel instance
    n_particles=500,          # particles per PF run
    n_iterations=10000,       # total MCMC iterations
    proposal='adaptive',      # proposal type: 'random_walk', 'adaptive', 'mala'
    burnin=2000,              # discard first 2000 samples
    thin=1,                   # thinning interval
    target_acceptance=0.234,  # target acceptance rate for adaptive proposals
    seed=42                   # random seed for reproducibility
)

chain = pmmh.sample(observations)
```

### Proposal Strategies

The proposal distribution $q(\theta' \mid \theta)$ has a major impact on mixing. particlefilterbox provides three options:

=== "Random Walk"

    The simplest proposal: a Gaussian centered at the current value.

    $$
    \theta' = \theta^{(m-1)} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \Sigma)
    $$

    ```python
    pmmh = PMMH(model, n_particles=500, n_iterations=10000,
                proposal='random_walk', proposal_scale=0.1)
    ```

    !!! warning "Tuning the scale"
        If `proposal_scale` is too large, almost all proposals are rejected.
        If too small, the chain moves in tiny steps and explores slowly.
        Aim for **20--30%** acceptance rate.

=== "Adaptive (AM)"

    The **Adaptive Metropolis** algorithm (Haario et al., 2001) automatically tunes the proposal covariance using the history of the chain:

    $$
    \Sigma_m = \frac{2.38^2}{d} \, \hat{\Sigma}_m + \epsilon I_d
    $$

    where $\hat{\Sigma}_m$ is the empirical covariance of $\theta^{(0)}, \ldots, \theta^{(m-1)}$ and $d$ is the parameter dimension.

    ```python
    pmmh = PMMH(model, n_particles=500, n_iterations=10000,
                proposal='adaptive', adaptation_start=500)
    ```

    This is the **recommended default** -- it removes the need to manually tune the proposal scale.

=== "MALA"

    The **Metropolis-Adjusted Langevin Algorithm** uses gradient information to propose in high-probability directions:

    $$
    \theta' = \theta^{(m-1)} + \frac{\epsilon^2}{2} \nabla_\theta \log p(\theta^{(m-1)} \mid y_{1:T}) + \epsilon \, \eta, \quad \eta \sim \mathcal{N}(0, I)
    $$

    ```python
    pmmh = PMMH(model, n_particles=500, n_iterations=10000,
                proposal='mala', step_size=0.01)
    ```

    !!! tip "When to use MALA"
        MALA is most beneficial for **high-dimensional** parameter spaces ($d > 5$) where
        random walk proposals become inefficient. It requires the gradient of the log-posterior,
        which particlefilterbox estimates using finite differences if not provided analytically.

---

## Example: Stochastic Volatility Model

The stochastic volatility (SV) model is a classic application of PMMH:

$$
\begin{aligned}
h_t &= \mu + \phi(h_{t-1} - \mu) + \sigma_\eta \, \eta_t, \quad \eta_t \sim \mathcal{N}(0, 1) \\
y_t &= \exp(h_t / 2) \, \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, 1)
\end{aligned}
$$

The parameters to estimate are $\theta = (\mu, \phi, \sigma_\eta)$.

```python
import numpy as np
from particlefilterbox.models import SVModel
from particlefilterbox.pmcmc import PMMH

# True parameters
model = SVModel(mu=0.0, phi=0.97, sigma=0.15)

# Simulate data
np.random.seed(42)
y, h_true = model.simulate(T=1000, return_states=True)

# Set up PMMH with priors
pmmh = PMMH(
    model,
    n_particles=500,
    n_iterations=15000,
    proposal='adaptive',
    burnin=5000,
    priors={
        'mu':    ('normal', 0.0, 1.0),      # N(0, 1)
        'phi':   ('beta', 20.0, 1.5),       # Beta(20, 1.5), concentrated near 1
        'sigma': ('half_cauchy', 0.0, 1.0),  # Half-Cauchy(0, 1)
    }
)

chain = pmmh.sample(y)
```

### Inspecting Results

```python
# Posterior summaries
print(chain.summary())
```

```text
Parameter    Mean     Std     2.5%    97.5%    ESS    R-hat
---------  ------  ------  ------  -------  -----  -------
mu          0.012   0.198  -0.381    0.403   3200    1.001
phi         0.968   0.011   0.944    0.987   2800    1.002
sigma       0.156   0.023   0.115    0.206   2500    1.003
```

```python
# Trace plots and posterior distributions
chain.plot_trace()         # trace plots for all parameters
chain.plot_posterior()     # marginal posterior histograms
chain.plot_autocorr()     # autocorrelation functions
```

---

## Tuning and Diagnostics

### Number of Particles ($N$)

The most important tuning parameter in PMMH is the number of particles. More particles reduce the variance of the likelihood estimate, increasing acceptance rate -- but each iteration becomes more expensive.

| $N$ | $\text{Var}[\log \hat{p}]$ | Acceptance rate | Cost per iteration |
|---|---|---|---|
| 50 | ~10 | < 5% | Low |
| 200 | ~3 | 10--20% | Moderate |
| 500 | ~1 | 20--35% | High |
| 1000 | ~0.5 | 35--50% | Very high |

!!! tip "Calibrating $N$"
    Run a short pilot chain (1000 iterations) and monitor:

    1. **Acceptance rate**: should be 15--30% for random walk, 20--40% for adaptive
    2. **Variance of log-likelihood**: estimate by running the PF multiple times at the same $\theta$

    ```python
    # Pilot run to calibrate N
    pmmh_pilot = PMMH(model, n_particles=200, n_iterations=1000,
                       proposal='adaptive')
    pilot_chain = pmmh_pilot.sample(y)
    print(f"Acceptance rate: {pilot_chain.acceptance_rate:.2%}")
    ```

### Convergence Diagnostics

After running PMMH, check convergence using standard MCMC diagnostics:

| Diagnostic | Target | What it measures |
|---|---|---|
| **Acceptance rate** | 15--30% | Efficiency of proposals |
| **Trace plots** | Stationary, well-mixing | Visual check for convergence |
| **ESS** (of chain) | > 1000 | Effective independent samples |
| **$\hat{R}$ (R-hat)** | < 1.05 | Between-chain vs within-chain variance |
| **Autocorrelation** | Fast decay | How quickly the chain forgets its past |

```python
# Diagnostics
print(f"Acceptance rate: {chain.acceptance_rate:.2%}")
print(f"ESS per parameter: {chain.ess}")
print(f"R-hat: {chain.rhat}")

# Run multiple chains for R-hat
chains = pmmh.sample(y, n_chains=4)
print(f"Multi-chain R-hat: {chains.rhat}")
```

!!! warning "Low acceptance rate"
    If the acceptance rate is below 5%:

    1. **Increase `n_particles`** -- the likelihood estimate is too noisy
    2. **Reduce `proposal_scale`** -- proposals are too ambitious
    3. **Switch to `proposal='adaptive'`** -- let the algorithm tune itself

!!! warning "Slow mixing despite good acceptance"
    If the acceptance rate looks fine but ESS is low:

    1. **Reparameterize** -- highly correlated parameters slow mixing
    2. **Use MALA** -- gradient information helps in correlated spaces
    3. **Consider PGAS** -- if you also need state trajectories, PGAS often mixes better

---

## Relation to Pseudo-Marginal Methods

PMMH is a specific instance of the broader class of **pseudo-marginal MCMC** methods (Beaumont, 2003; Andrieu & Roberts, 2009). Any algorithm that:

1. Targets a posterior $p(\theta \mid y)$ using Metropolis-Hastings
2. Replaces the intractable likelihood $p(y \mid \theta)$ with an **unbiased estimate** $\hat{p}(y \mid \theta)$

is a pseudo-marginal method. PMMH uses the particle filter as the unbiased estimator, but other estimators (importance sampling, bridge sampling) could be used instead.

The key theoretical guarantees are:

- **Exactness**: the chain targets $p(\theta \mid y_{1:T})$ exactly, not an approximation
- **Ergodicity**: under mild conditions, the chain converges to the posterior regardless of $N$ (though small $N$ makes convergence very slow)
- **CLT**: posterior expectations satisfy a central limit theorem with variance that depends on both the MCMC mixing and the PF variance

---

## What's Next?

<div class="grid cards" markdown>

- :material-arrow-right-bold: **[Particle Gibbs](particle-gibbs.md)**

    Joint inference over parameters and states using conditional SMC

- :material-arrow-right-bold: **[PGAS](pgas.md)**

    Improved mixing with ancestor sampling -- the recommended default

- :material-arrow-left-bold: **[PMCMC Overview](index.md)**

    Back to the framework overview and method comparison

</div>
