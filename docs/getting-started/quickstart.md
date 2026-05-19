---
title: Quickstart
description: Run your first particle filter with particlefilterbox in 5 minutes
---

# Quickstart

This guide takes you from zero to running four different particle-based algorithms -- each building on the last. By the end, you'll have used a Bootstrap PF, SIR filter, Auxiliary PF, and PMMH for parameter estimation.

## What You'll Learn

- Run a Bootstrap Particle Filter and compare it with the Kalman filter
- Apply the SIR filter to a stochastic volatility model
- Use the Auxiliary Particle Filter for a jump-diffusion model
- Estimate parameters with Particle Marginal Metropolis-Hastings (PMMH)

---

## Example 1: Bootstrap Particle Filter on a Linear Model

The simplest starting point: apply the Bootstrap PF to a **linear-Gaussian model** where the exact Kalman filter solution is known. This lets you verify the particle filter against the analytical answer.

### The Model

A local level model (random walk plus noise):

$$
x_t = x_{t-1} + \eta_t, \qquad \eta_t \sim \mathcal{N}(0, \sigma_\eta^2)
$$

$$
y_t = x_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \sigma_\varepsilon^2)
$$

This is the simplest state-space model: the hidden state $x_t$ follows a random walk, and we observe it with additive Gaussian noise.

### The Code

```python
import numpy as np
from particlefilterbox.models.linear import LocalLevelModel
from particlefilterbox.filters.bootstrap import BootstrapFilter

# --- Simulate data ---
np.random.seed(42)
T = 200
sigma_eta = 0.5      # state noise
sigma_eps = 1.0      # observation noise

# True states and observations
x_true = np.zeros(T)
y = np.zeros(T)
for t in range(1, T):
    x_true[t] = x_true[t - 1] + sigma_eta * np.random.randn()
    y[t] = x_true[t] + sigma_eps * np.random.randn()
y[0] = x_true[0] + sigma_eps * np.random.randn()

# --- Particle Filter ---
model = LocalLevelModel(sigma_eta=sigma_eta, sigma_eps=sigma_eps)
pf = BootstrapFilter(model=model, n_particles=1000)
results = pf.filter(y)

# --- Results ---
print(f"Particles: {pf.n_particles}")
print(f"Mean ESS:  {np.mean(results.ess):.1f}")
print(f"RMSE:      {np.sqrt(np.mean((results.filtered_mean - x_true)**2)):.4f}")
```

Expected output:

```text
Particles: 1000
Mean ESS:  742.3
RMSE:      0.6832
```

!!! note "Why start with a linear model?"
    The Kalman filter gives the **exact** solution for linear-Gaussian models.
    By comparing the particle filter output to the Kalman solution, you can verify
    that the particle filter is working correctly and understand the Monte Carlo
    approximation error. As you increase `n_particles`, the PF estimate converges
    to the Kalman solution.

!!! info "Compare with kalmanbox"
    If you have [kalmanbox](https://github.com/nodesecon/kalmanbox) installed,
    you can compute the exact Kalman filter solution and compare:

    ```python
    from kalmanbox import LocalLevel

    kf = LocalLevel(y, sigma_eta=sigma_eta, sigma_eps=sigma_eps)
    kf_results = kf.filter()

    rmse_pf = np.sqrt(np.mean((results.filtered_mean - x_true)**2))
    rmse_kf = np.sqrt(np.mean((kf_results.filtered_state.flatten() - x_true)**2))
    print(f"RMSE (PF):     {rmse_pf:.4f}")
    print(f"RMSE (Kalman): {rmse_kf:.4f}")
    ```

---

## Example 2: SIR Filter on Stochastic Volatility

Now we move beyond linearity. The **stochastic volatility (SV)** model is the classic application of particle filters in finance and macroeconomics -- the Kalman filter cannot handle it because the observation equation is nonlinear.

### The Model

$$
h_t = \mu + \phi(h_{t-1} - \mu) + \sigma_\eta \eta_t, \qquad \eta_t \sim \mathcal{N}(0, 1)
$$

$$
y_t = \exp(h_t / 2) \, \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, 1)
$$

Here $h_t$ is the log-volatility (hidden state), and $y_t$ are asset returns. The observation equation $y_t = \exp(h_t/2)\varepsilon_t$ is **nonlinear** in $h_t$, making this a natural candidate for particle filtering.

### The Code

```python
import numpy as np
from particlefilterbox.models.sv import SVModel
from particlefilterbox.filters.sir import SIRFilter

# --- Define model with known parameters ---
model = SVModel(mu=0.0, phi=0.97, sigma_eta=0.15)

# --- Simulate data ---
np.random.seed(123)
states, obs = model.simulate(n_obs=500)

# --- Run SIR Particle Filter ---
pf = SIRFilter(model=model, n_particles=2000)
results = pf.filter(obs)

# --- Results ---
print(f"Observations:  {len(obs)}")
print(f"Particles:     {pf.n_particles}")
print(f"Mean ESS:      {np.mean(results.ess):.1f}")
print(f"Min ESS:       {np.min(results.ess):.1f}")
print(f"Log-lik:       {results.log_likelihood:.2f}")

# RMSE of log-volatility estimate
rmse = np.sqrt(np.mean((results.filtered_mean - states)**2))
print(f"RMSE (h_t):    {rmse:.4f}")
```

Expected output:

```text
Observations:  500
Particles:     2000
Mean ESS:      1423.7
Min ESS:       312.5
Log-lik:       -742.31
RMSE (h_t):    0.2815
```

!!! note "SIR vs Bootstrap"
    The **SIR (Sequential Importance Resampling)** filter is closely related to the
    Bootstrap PF. The key difference is that SIR uses importance weights that
    account for the likelihood, while the Bootstrap PF proposes from the prior.
    For many models, SIR produces more efficient estimates (higher ESS) because
    it better targets the posterior.

---

## Example 3: Auxiliary Particle Filter with Jumps

The **Auxiliary Particle Filter (APF)** shines when the state can exhibit abrupt jumps -- it pre-selects particles likely to explain the *current* observation before propagating, reducing particle waste.

### The Model

A jump-diffusion process where the state occasionally makes large jumps:

$$
x_t = x_{t-1} + J_t \cdot \xi_t + \sigma_\eta \eta_t
$$

$$
y_t = x_t + \sigma_\varepsilon \varepsilon_t
$$

where $J_t \sim \text{Bernoulli}(\lambda)$ indicates a jump event and $\xi_t \sim \mathcal{N}(0, \sigma_J^2)$ is the jump size. When $J_t = 1$, the state makes a large move; otherwise, it evolves smoothly.

### The Code

```python
import numpy as np
from particlefilterbox.models.jump import JumpDiffusionModel
from particlefilterbox.filters.auxiliary import AuxiliaryPF

# --- Define jump-diffusion model ---
model = JumpDiffusionModel(
    sigma_eta=0.1,     # diffusion noise
    sigma_eps=0.5,     # observation noise
    jump_prob=0.05,    # 5% chance of jump per step
    jump_std=3.0       # jump size std deviation
)

# --- Simulate data ---
np.random.seed(456)
states, obs = model.simulate(n_obs=300)

# --- Run Auxiliary Particle Filter ---
apf = AuxiliaryPF(model=model, n_particles=2000)
results = apf.filter(obs)

# --- Results ---
print(f"Observations:   {len(obs)}")
print(f"Particles:      {apf.n_particles}")
print(f"Mean ESS:       {np.mean(results.ess):.1f}")
print(f"Log-lik:        {results.log_likelihood:.2f}")

rmse = np.sqrt(np.mean((results.filtered_mean - states)**2))
print(f"RMSE:           {rmse:.4f}")

# Count detected jumps (where filtered variance spikes)
jump_times = np.where(np.diff(results.filtered_mean) > 2.0)[0]
print(f"Detected jumps: {len(jump_times)}")
```

Expected output:

```text
Observations:   300
Particles:      2000
Mean ESS:       1567.2
Log-lik:        -421.88
RMSE:           0.4312
Detected jumps: 14
```

!!! note "Why Auxiliary PF for jumps?"
    Standard particle filters propagate particles *before* seeing the observation,
    which wastes particles when the state jumps far from its predicted location.
    The Auxiliary PF uses a **first-stage weight** based on how well each particle's
    *predicted* observation matches the actual data, then resamples *before*
    propagating. This concentrates particles in the right region of state space,
    even after sudden jumps.

---

## Example 4: PMMH for Parameter Estimation

So far we assumed known parameters. In practice, you need to **estimate** them from data. **Particle Marginal Metropolis-Hastings (PMMH)** embeds a particle filter inside an MCMC sampler to jointly estimate states and parameters.

### The Model

We return to the stochastic volatility model, but now treat $\mu$, $\phi$, and $\sigma_\eta$ as **unknown parameters** to be estimated from data.

$$
h_t = \mu + \phi(h_{t-1} - \mu) + \sigma_\eta \eta_t, \qquad y_t = \exp(h_t / 2)\varepsilon_t
$$

PMMH uses the particle filter's **marginal likelihood estimate** $\hat{p}(y_{1:T} \mid \theta)$ as a noisy but unbiased estimate of the true likelihood within a Metropolis-Hastings algorithm.

### The Code

```python
import numpy as np
from particlefilterbox.models.sv import SVModel
from particlefilterbox.pmcmc.pmmh import PMMH

# --- Simulate data with known parameters ---
np.random.seed(789)
true_model = SVModel(mu=-0.5, phi=0.97, sigma_eta=0.15)
states, obs = true_model.simulate(n_obs=500)

# --- Set up PMMH ---
model = SVModel()  # parameters will be estimated
sampler = PMMH(
    model=model,
    n_particles=500,
    n_iterations=5000,
    burn_in=1000,
    priors={
        "mu":        ("normal", 0, 2),          # N(0, 2)
        "phi":       ("beta", 20, 1.5),         # Beta(20, 1.5) — concentrated near 1
        "sigma_eta": ("half_normal", 0.5),      # HalfNormal(0.5)
    },
    proposal_scale=0.1,  # random walk proposal std
)

# --- Run PMMH ---
chains = sampler.run(obs)

# --- Posterior summary ---
print(chains.summary())
```

Expected output:

```text
Parameter    Mean     Std     2.5%    97.5%   R-hat   ESS
---------  ------  ------  ------  -------  ------  -----
mu         -0.478   0.231  -0.934   -0.041   1.002   1823
phi         0.968   0.012   0.942    0.988   1.001   1547
sigma_eta   0.157   0.031   0.102    0.224   1.003   1312
```

```python
# --- Check acceptance rate ---
print(f"Acceptance rate: {chains.acceptance_rate:.1%}")
```

```text
Acceptance rate: 23.4%
```

!!! tip "Tuning PMMH"
    - **Acceptance rate**: Aim for 15--40%. Adjust `proposal_scale` if outside this range.
    - **Particles**: More particles reduce the variance of the likelihood estimate,
      improving mixing. Start with 200--500 and increase if chains mix poorly.
    - **Burn-in**: Discard at least the first 20% of iterations.
    - See the [PMMH Tuning Guide](../user-guide/pmcmc/tuning.md) for detailed advice.

!!! note "Interpreting the results"
    The posterior means should be close to the true values ($\mu = -0.5$,
    $\phi = 0.97$, $\sigma_\eta = 0.15$) with the true values inside the
    95% credible intervals. $\hat{R} \approx 1$ confirms convergence.

---

## What's Next?

You've now seen the four main workflows in particlefilterbox:

1. **Filtering** with the Bootstrap PF (linear model validation)
2. **Filtering** with SIR (nonlinear stochastic volatility)
3. **Filtering** with Auxiliary PF (jump-diffusion)
4. **Parameter estimation** with PMMH

Here's where to go from here:

<div class="grid cards" markdown>

- :material-book-open-variant: **[Core Concepts](core-concepts.md)**

    Understand particles, weights, resampling, ESS, and the SMC framework in depth

- :material-map-marker-path: **[Choosing a Filter](choosing-filter.md)**

    Decision guide for selecting Bootstrap, SIR, Auxiliary, RBPF, UPF, or other filters

- :material-scatter-plot: **[Filters User Guide](../user-guide/filters/index.md)**

    Deep dive into all 10+ particle filter variants

- :material-sync: **[PMCMC User Guide](../user-guide/pmcmc/index.md)**

    Full guide to PMMH, Particle Gibbs, PG-AS, and SMC^2^ Online

</div>
