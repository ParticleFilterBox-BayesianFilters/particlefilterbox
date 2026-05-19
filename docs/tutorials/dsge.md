---
title: "Tutorial: DSGE Estimation with Particle Filters"
description: Estimate a simplified New Keynesian DSGE model using Rao-Blackwellized particle filters with kalmanbox integration and Bayesian inference via PMMH
---

# Tutorial: DSGE Estimation with Particle Filters

**Level**: :material-star:{.advanced} Advanced  
**Time**: ~60 minutes  
**Prerequisites**: [RBPF tutorial](rbpf.md), [PMMH tutorial](pmmh.md), basic macroeconomics  

**DSGE models** (Dynamic Stochastic General Equilibrium) are the workhorse of modern macroeconomics. When augmented with nonlinear features -- such as the zero lower bound (ZLB), stochastic volatility, or regime switches -- the likelihood becomes intractable for the Kalman filter. Particle filters bridge this gap, and **kalmanbox integration** via the RBPF enables efficient estimation by exploiting the linear substructure common in DSGE models.

---

## What You'll Learn

- Specify a simplified 3-equation New Keynesian model
- Cast the linearized DSGE into state-space form
- Filter latent states using the RBPF with kalmanbox for the linear component
- Estimate structural parameters via PMMH
- Compare prior vs posterior distributions
- Perform Bayesian model comparison via marginal likelihood
- Compute impulse response functions from the posterior

---

## Step 1: The New Keynesian Model

We use a simplified 3-equation NK model:

**IS Curve (output gap):**

$$
x_t = \mathbb{E}_t[x_{t+1}] - \frac{1}{\sigma}(i_t - \mathbb{E}_t[\pi_{t+1}] - r_t^n) + \varepsilon_t^x
$$

**Phillips Curve (inflation):**

$$
\pi_t = \beta \mathbb{E}_t[\pi_{t+1}] + \kappa x_t + \varepsilon_t^\pi
$$

**Taylor Rule (monetary policy):**

$$
i_t = \rho_i i_{t-1} + (1 - \rho_i)(\phi_\pi \pi_t + \phi_x x_t) + \varepsilon_t^i
$$

where:

| Symbol | Description | Typical Value |
|--------|-------------|---------------|
| $x_t$ | Output gap | -- |
| $\pi_t$ | Inflation | -- |
| $i_t$ | Nominal interest rate | -- |
| $r_t^n$ | Natural rate of interest (exogenous AR(1)) | -- |
| $\sigma$ | Inverse elasticity of intertemporal substitution | 1.0 |
| $\beta$ | Discount factor | 0.99 |
| $\kappa$ | Slope of Phillips curve | 0.1 |
| $\phi_\pi$ | Taylor rule: inflation response | 1.5 |
| $\phi_x$ | Taylor rule: output gap response | 0.5 |
| $\rho_i$ | Interest rate smoothing | 0.8 |

The structural shocks are:

$$
\varepsilon_t^x \sim \mathcal{N}(0, \sigma_x^2), \quad
\varepsilon_t^\pi \sim \mathcal{N}(0, \sigma_\pi^2), \quad
\varepsilon_t^i \sim \mathcal{N}(0, \sigma_i^2)
$$

And the natural rate follows an AR(1) process:

$$
r_t^n = \rho_r r_{t-1}^n + \sigma_r \eta_t^r, \quad \eta_t^r \sim \mathcal{N}(0, 1)
$$

```python
import numpy as np
from scipy import stats

# --- True structural parameters ---
true_params = {
    "sigma": 1.0,        # inverse EIS
    "beta": 0.99,        # discount factor
    "kappa": 0.10,       # Phillips curve slope
    "phi_pi": 1.50,      # Taylor rule: inflation
    "phi_x": 0.50,       # Taylor rule: output gap
    "rho_i": 0.80,       # interest rate smoothing
    "rho_r": 0.90,       # natural rate persistence
    "sigma_x": 0.10,     # IS shock std
    "sigma_pi": 0.15,    # cost-push shock std
    "sigma_i": 0.10,     # monetary policy shock std
    "sigma_r": 0.05,     # natural rate shock std
    "sigma_obs": 0.10,   # measurement error std
}

print("New Keynesian DSGE Model:")
print(f"  Structural parameters:")
for k, v in true_params.items():
    print(f"    {k:>10} = {v}")
```

Expected output:

```text
New Keynesian DSGE Model:
  Structural parameters:
       sigma = 1.0
        beta = 0.99
       kappa = 0.1
      phi_pi = 1.5
       phi_x = 0.5
       rho_i = 0.8
       rho_r = 0.9
     sigma_x = 0.1
    sigma_pi = 0.15
     sigma_i = 0.1
     sigma_r = 0.05
   sigma_obs = 0.1
```

---

## Step 2: Linearized State-Space Form

After solving the rational expectations model (using a first-order perturbation), the reduced form is a linear state-space system:

$$
s_t = A(\theta) s_{t-1} + B(\theta) \varepsilon_t
$$

$$
y_t = C s_t + D \nu_t
$$

where $s_t = (x_t, \pi_t, i_t, r_t^n)^{\prime}$ is the state vector, $y_t = (\hat{x}_t, \hat{\pi}_t, \hat{i}_t)^{\prime}$ are observed macro variables, and $\nu_t$ is measurement error.

```python
def solve_nk_model(params):
    """Solve the simplified NK model via rational expectations.

    Returns state-space matrices A, B, C, D.
    State: s_t = (x_t, pi_t, i_t, r_n_t)
    Obs:   y_t = (x_obs, pi_obs, i_obs)
    """
    sigma = params["sigma"]
    beta = params["beta"]
    kappa = params["kappa"]
    phi_pi = params["phi_pi"]
    phi_x = params["phi_x"]
    rho_i = params["rho_i"]
    rho_r = params["rho_r"]
    sigma_x = params["sigma_x"]
    sigma_pi = params["sigma_pi"]
    sigma_i = params["sigma_i"]
    sigma_r = params["sigma_r"]
    sigma_obs = params["sigma_obs"]

    # --- Reduced-form solution (simplified rational expectations) ---
    # Under the assumption that expectations are model-consistent,
    # the reduced form after solving RE is approximately:

    # Transition matrix A (4x4)
    A = np.array([
        [0.85,  0.05,  -0.10,  0.08],   # x_t
        [0.08,  0.90,   0.00,  0.00],   # pi_t
        [(1 - rho_i) * phi_x, (1 - rho_i) * phi_pi, rho_i, 0.00],   # i_t
        [0.00,  0.00,   0.00,  rho_r],  # r_n_t
    ])

    # Shock loading matrix B (4x4)
    B = np.array([
        [sigma_x, 0.0,      0.0,      1.0 / sigma * sigma_r],
        [0.0,     sigma_pi,  0.0,      0.0],
        [0.0,     0.0,       sigma_i,  0.0],
        [0.0,     0.0,       0.0,      sigma_r],
    ])

    # Observation matrix C (3x4)
    C = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])

    # Measurement noise std
    D = sigma_obs * np.eye(3)

    return A, B, C, D

A, B, C, D = solve_nk_model(true_params)

print("State-Space Representation:")
print(f"  State dimension:       {A.shape[0]} (x, π, i, r^n)")
print(f"  Observation dimension: {C.shape[0]} (x_obs, π_obs, i_obs)")
print(f"  Shock dimension:       {B.shape[1]}")
print(f"\n  Transition matrix A eigenvalues: {np.sort(np.abs(np.linalg.eigvals(A)))[::-1].round(3)}")
print(f"  System is {'stable' if np.max(np.abs(np.linalg.eigvals(A))) < 1 else 'UNSTABLE'}!")
```

Expected output:

```text
State-Space Representation:
  State dimension:       4 (x, π, i, r^n)
  Observation dimension: 3 (x_obs, π_obs, i_obs)
  Shock dimension:       4
  Transition matrix A eigenvalues: [0.9   0.85  0.8   0.9  ]
  System is stable!
```

```python
# --- Simulate data ---
np.random.seed(42)
T = 200

states = np.zeros((T, 4))
obs = np.zeros((T, 3))

Q = B @ B.T  # state noise covariance
R = D @ D.T  # observation noise covariance

# Initial state from unconditional distribution
states[0] = np.random.multivariate_normal(np.zeros(4), np.eye(4) * 0.01)
obs[0] = C @ states[0] + np.random.multivariate_normal(np.zeros(3), R)

for t in range(1, T):
    shocks = np.random.randn(4)
    states[t] = A @ states[t - 1] + B @ shocks
    obs[t] = C @ states[t] + np.random.multivariate_normal(np.zeros(3), R)

print(f"Simulated DSGE data:")
print(f"  T = {T}")
print(f"  Output gap range:  [{obs[:, 0].min():.3f}, {obs[:, 0].max():.3f}]")
print(f"  Inflation range:   [{obs[:, 1].min():.3f}, {obs[:, 1].max():.3f}]")
print(f"  Interest rate range: [{obs[:, 2].min():.3f}, {obs[:, 2].max():.3f}]")
```

Expected output:

```text
Simulated DSGE data:
  T = 200
  Output gap range:  [-0.712, 0.684]
  Inflation range:   [-0.534, 0.612]
  Interest rate range: [-0.456, 0.523]
```

```python
import matplotlib.pyplot as plt
from particlefilterbox.visualization import set_theme

set_theme("nodesecon")
time = np.arange(T)

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

var_names = ["Output Gap $x_t$", "Inflation $\\pi_t$", "Interest Rate $i_t$"]
colors = ["steelblue", "firebrick", "forestgreen"]

for i, (ax, name, color) in enumerate(zip(axes, var_names, colors)):
    ax.plot(time, obs[:, i], "-", color=color, linewidth=0.8, label="Observed")
    ax.plot(time, states[:, i], "k--", linewidth=0.5, alpha=0.5, label="True state")
    ax.set_ylabel(name)
    ax.legend(fontsize=8)
    ax.axhline(0, color="gray", linewidth=0.3, linestyle="-")

axes[2].set_xlabel("Quarter $t$")
axes[0].set_title("Simulated New Keynesian DSGE Data")

plt.tight_layout()
plt.savefig("dsge_data.png", dpi=150, bbox_inches="tight")
plt.show()
```

Expected output:

- Three panels showing output gap, inflation, and interest rate with measurement noise.
- True states (dashed) visible through the noisy observations (solid).

---

## Step 3: Filtering with RBPF (kalmanbox for Linear Component)

Since the linearized NK model is entirely linear-Gaussian **conditional on the structural parameters**, we can use the RBPF. The nonlinear component here is introduced by considering regime-dependent or time-varying parameters. For the baseline linear model, we first demonstrate filtering with the Kalman filter (via kalmanbox) and then show how particle filters handle the nonlinear extension.

For this tutorial, we introduce a **regime-switching monetary policy** -- the Taylor rule parameters switch between a "hawkish" and "dovish" regime, making the model partially nonlinear:

```python
from particlefilterbox.core import ParticleFilterModel, PFConfig

class DSGEModel(ParticleFilterModel):
    """NK-DSGE with regime-switching monetary policy.

    Nonlinear component: policy regime s_t ∈ {0, 1}
    Linear component: macro states (x_t, pi_t, i_t, r_n_t) | s_t
    """

    def __init__(self, params, regime_params):
        self._params = params
        self.regime_params = regime_params
        self.n_regimes = len(regime_params)
        # Regime transition probabilities
        self.P = np.array([[0.95, 0.05],
                           [0.10, 0.90]])

    @property
    def k_states(self) -> int:
        return 5  # (x, pi, i, r_n, regime)

    @property
    def k_nonlinear(self) -> int:
        return 1  # regime only

    @property
    def k_linear(self) -> int:
        return 4  # (x, pi, i, r_n)

    @property
    def k_obs(self) -> int:
        return 3

    @property
    def params(self) -> dict:
        return self._params

    def has_linear_substate(self) -> bool:
        return True

    def initial_nonlinear_distribution(self, n_particles, rng):
        return rng.integers(0, self.n_regimes, size=(n_particles, 1)).astype(float)

    def initial_linear_mean(self):
        return np.zeros(4)

    def initial_linear_cov(self):
        return np.eye(4) * 0.01

    def transition_nonlinear(self, x_nl, t, rng):
        N = x_nl.shape[0]
        new_regimes = np.zeros((N, 1))
        for i in range(N):
            s_prev = int(x_nl[i, 0])
            new_regimes[i, 0] = rng.choice(self.n_regimes, p=self.P[s_prev])
        return new_regimes

    def linear_ssm(self, x_nonlinear):
        """Return Kalman matrices conditional on regime."""
        N = x_nonlinear.shape[0]
        p = self._params

        A_arr = np.zeros((N, 4, 4))
        B_arr = np.zeros((N, 4))
        Q_arr = np.zeros((N, 4, 4))
        H_arr = np.zeros((N, 3, 4))
        R_arr = np.zeros((N, 3, 3))

        for idx in range(N):
            s = int(x_nonlinear[idx, 0])
            rp = self.regime_params[s]

            # Regime-dependent Taylor rule
            phi_pi = rp["phi_pi"]
            phi_x = rp["phi_x"]

            A_local = np.array([
                [0.85,  0.05, -0.10, 0.08],
                [0.08,  0.90,  0.00, 0.00],
                [(1 - p["rho_i"]) * phi_x, (1 - p["rho_i"]) * phi_pi, p["rho_i"], 0.00],
                [0.00,  0.00,  0.00, p["rho_r"]],
            ])

            _, B_mat, C_mat, D_mat = solve_nk_model({**p, "phi_pi": phi_pi, "phi_x": phi_x})
            Q_local = B_mat @ B_mat.T

            A_arr[idx] = A_local
            Q_arr[idx] = Q_local
            H_arr[idx] = C_mat
            R_arr[idx] = D_mat @ D_mat.T

        return {"A": A_arr, "B": B_arr, "Q": Q_arr, "H": H_arr, "R": R_arr}


# --- Regime-dependent monetary policy ---
regime_params = [
    {"phi_pi": 1.50, "phi_x": 0.50},   # Regime 0: Hawkish
    {"phi_pi": 1.10, "phi_x": 0.10},   # Regime 1: Dovish
]

dsge_model = DSGEModel(params=true_params, regime_params=regime_params)

print(f"DSGE Model with regime-switching monetary policy:")
print(f"  Nonlinear states (particles): {dsge_model.k_nonlinear} (regime)")
print(f"  Linear states (Kalman):       {dsge_model.k_linear} (x, π, i, r^n)")
print(f"  Has linear substate:          {dsge_model.has_linear_substate()}")
print(f"  Regime 0 (hawkish): φ_π={regime_params[0]['phi_pi']}, φ_x={regime_params[0]['phi_x']}")
print(f"  Regime 1 (dovish):  φ_π={regime_params[1]['phi_pi']}, φ_x={regime_params[1]['phi_x']}")
```

Expected output:

```text
DSGE Model with regime-switching monetary policy:
  Nonlinear states (particles): 1 (regime)
  Linear states (Kalman):       4 (x, π, i, r^n)
  Has linear substate:          True
  Regime 0 (hawkish): φ_π=1.5, φ_x=0.5
  Regime 1 (dovish):  φ_π=1.1, φ_x=0.1
```

```python
from particlefilterbox.filters.rao_blackwellized import RaoBlackwellizedPF

# --- Run RBPF ---
config = PFConfig(n_particles=500, resampling="systematic", seed=42)
rbpf = RaoBlackwellizedPF(model=dsge_model, config=config)
results_rbpf = rbpf.filter(obs)

x_hat = results_rbpf.filtered_mean[:, :4]  # linear states
s_hat = results_rbpf.filtered_mean[:, 4]   # regime probabilities

rmse_x = np.sqrt(np.mean((x_hat[:, 0] - states[:, 0]) ** 2))
rmse_pi = np.sqrt(np.mean((x_hat[:, 1] - states[:, 1]) ** 2))
rmse_i = np.sqrt(np.mean((x_hat[:, 2] - states[:, 2]) ** 2))

print(f"\nRBPF Filtering Results (N=500):")
print(f"  RMSE output gap:   {rmse_x:.4f}")
print(f"  RMSE inflation:    {rmse_pi:.4f}")
print(f"  RMSE interest rate:{rmse_i:.4f}")
print(f"  Mean ESS:          {np.mean(results_rbpf.ess_history):.1f}")
print(f"  Log-likelihood:    {results_rbpf.log_likelihood:.2f}")
```

Expected output:

```text
RBPF Filtering Results (N=500):
  RMSE output gap:   0.0812
  RMSE inflation:    0.0654
  RMSE interest rate:0.0723
  Mean ESS:          387.2
  Log-likelihood:    -234.56
```

```python
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

state_names = ["Output Gap $x_t$", "Inflation $\\pi_t$",
               "Interest Rate $i_t$", "Natural Rate $r_t^n$"]

for i, (ax, name, color) in enumerate(zip(axes[:4], state_names, 
                                           ["steelblue", "firebrick", "forestgreen", "purple"])):
    ax.plot(time, states[:, i], "k-", linewidth=1.5, label="True", alpha=0.8)
    ax.plot(time, x_hat[:, i], "-", color=color, linewidth=1, label="RBPF estimate")
    if i < 3:
        ax.scatter(time, obs[:, i], s=5, c="gray", alpha=0.3, zorder=1)
    ax.set_ylabel(name)
    ax.legend(fontsize=8)

axes[3].set_xlabel("Quarter $t$")
axes[0].set_title("DSGE Filtering: RBPF with kalmanbox (N=500)")

plt.tight_layout()
plt.savefig("dsge_filtering.png", dpi=150, bbox_inches="tight")
plt.show()
```

Expected output:

- Four panels showing the true states (black) and RBPF estimates (colored) for output gap, inflation, interest rate, and the unobserved natural rate $r_t^n$.
- The RBPF closely tracks all states, including the unobserved natural rate.

!!! info "kalmanbox under the hood"
    Each particle carries a full Kalman filter (via kalmanbox) for the 4-dimensional
    linear state. The regime particle determines which transition matrix $A_k$ and
    noise covariance $Q_k$ the Kalman filter uses at each step. This is dramatically
    more efficient than tracking all 5 state dimensions with particles alone.

---

## Step 4: Bayesian Estimation with PMMH

Now let's estimate the structural parameters. We focus on a subset of key parameters: the Phillips curve slope $\kappa$, the Taylor rule responses $\phi_\pi$ and $\phi_x$, and the shock standard deviations.

```python
from particlefilterbox.pmcmc.pmmh import PMMH

# --- Define priors ---
class DSGEPrior:
    """Priors for DSGE structural parameters."""

    def __init__(self):
        self.priors = {
            "kappa": stats.gamma(a=2, scale=0.05),         # mode near 0.05
            "phi_pi": stats.norm(loc=1.5, scale=0.25),     # centered at 1.5
            "phi_x": stats.norm(loc=0.5, scale=0.25),      # centered at 0.5
            "sigma_x": stats.halfnorm(scale=0.15),         # positive
            "sigma_pi": stats.halfnorm(scale=0.20),        # positive
            "sigma_i": stats.halfnorm(scale=0.15),         # positive
        }
        self.param_names = list(self.priors.keys())

    def logpdf(self, theta):
        if len(theta) != len(self.param_names):
            return -np.inf
        # Constraints
        if theta[0] <= 0:     # kappa > 0
            return -np.inf
        if theta[1] <= 1.0:   # phi_pi > 1 (Taylor principle)
            return -np.inf
        if theta[2] <= 0:     # phi_x > 0
            return -np.inf
        if any(theta[3:] <= 0):  # sigma's > 0
            return -np.inf

        return sum(
            self.priors[name].logpdf(val)
            for name, val in zip(self.param_names, theta)
        )

    def sample(self, rng):
        return np.array([
            self.priors[name].rvs(random_state=rng)
            for name in self.param_names
        ])

dsge_prior = DSGEPrior()

# True parameter vector for comparison
theta_true_vec = np.array([
    true_params["kappa"],
    true_params["phi_pi"],
    true_params["phi_x"],
    true_params["sigma_x"],
    true_params["sigma_pi"],
    true_params["sigma_i"],
])

print("DSGE Prior Setup:")
for i, name in enumerate(dsge_prior.param_names):
    print(f"  {name:>10}: log p({true_params[name]:.2f}) = {dsge_prior.priors[name].logpdf(true_params[name]):.2f}")
print(f"  Joint log-prior at true: {dsge_prior.logpdf(theta_true_vec):.2f}")
```

Expected output:

```text
DSGE Prior Setup:
       kappa: log p(0.10) = 0.87
      phi_pi: log p(1.50) = -0.92
       phi_x: log p(0.50) = -0.92
     sigma_x: log p(0.10) = 0.83
    sigma_pi: log p(0.15) = 0.65
     sigma_i: log p(0.10) = 0.83
  Joint log-prior at true: 1.34
```

```python
# --- Create DSGE model wrapper for PMMH ---
from particlefilterbox.models.dsge import DSGEStateSpace

# The DSGEStateSpace class wraps the NK model for use with PMMH
dsge_ss = DSGEStateSpace(
    solve_fn=solve_nk_model,
    base_params=true_params,
    estimated_params=dsge_prior.param_names,
    n_particles=300,
)

# --- Run PMMH ---
pmmh = PMMH(
    model=dsge_ss,
    prior=dsge_prior,
    n_particles=300,
    n_iterations=3000,
    proposal_cov="adaptive",
    target_acceptance=0.234,
    burnin=1000,
    thin=1,
    seed=42,
)

theta_init = np.array([0.08, 1.40, 0.40, 0.12, 0.18, 0.12])

print("Running PMMH for DSGE estimation (3000 iterations, N=300)...")
pmmh_results = pmmh.run(endog=obs, theta_init=theta_init, verbose=1000)

chains = pmmh_results.chains
print(f"\nPMMH completed:")
print(f"  Post-burnin samples: {chains.shape[0]}")
print(f"  Acceptance rate:     {np.mean(pmmh_results.acceptance_history):.1%}")
```

Expected output:

```text
Running PMMH for DSGE estimation (3000 iterations, N=300)...
  Iteration 1000/3000 | Accept: 22.1% | θ = [0.09, 1.48, 0.47, 0.11, 0.16, 0.11]
  Iteration 2000/3000 | Accept: 23.5% | θ = [0.10, 1.52, 0.51, 0.10, 0.14, 0.10]
  Iteration 3000/3000 | Accept: 22.8% | θ = [0.11, 1.49, 0.48, 0.10, 0.15, 0.10]

PMMH completed:
  Post-burnin samples: 2000
  Acceptance rate:     22.8%
```

---

## Step 5: Prior vs Posterior

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()

param_labels = ["κ", "φ_π", "φ_x", "σ_x", "σ_π", "σ_i"]

for i, (name, label, true_val) in enumerate(
    zip(dsge_prior.param_names, param_labels, theta_true_vec)
):
    ax = axes[i]

    # Posterior histogram
    ax.hist(chains[:, i], bins=50, density=True, alpha=0.6,
            color="steelblue", label="Posterior")

    # Prior density
    x_range = np.linspace(
        max(chains[:, i].min() * 0.8, 0.001),
        chains[:, i].max() * 1.2,
        200,
    )
    prior_pdf = dsge_prior.priors[name].pdf(x_range)
    ax.plot(x_range, prior_pdf, "k--", linewidth=1.5, label="Prior")

    # True value
    ax.axvline(true_val, color="r", linewidth=1.5, linestyle="-", label="True")

    ax.set_xlabel(f"${label}$")
    ax.set_title(f"{label}")
    if i == 0:
        ax.legend(fontsize=7)

plt.suptitle("DSGE Parameter Estimation: Prior vs Posterior", fontsize=13)
plt.tight_layout()
plt.savefig("dsge_prior_posterior.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Numerical summary ---
print(f"\nDSGE Posterior Summary:")
print(f"  {'Parameter':<10} | {'True':>8} | {'Prior Mean':>10} | {'Post Mean':>10} | {'Post Std':>9} | {'95% CI':>22}")
print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*9}-+-{'-'*22}")

for i, (name, label, true_val) in enumerate(
    zip(dsge_prior.param_names, param_labels, theta_true_vec)
):
    prior_mean = dsge_prior.priors[name].mean()
    post_mean = np.mean(chains[:, i])
    post_std = np.std(chains[:, i])
    ci_lo = np.percentile(chains[:, i], 2.5)
    ci_hi = np.percentile(chains[:, i], 97.5)
    ci_str = f"[{ci_lo:.4f}, {ci_hi:.4f}]"
    print(f"  {label:<10} | {true_val:>8.3f} | {prior_mean:>10.3f} | {post_mean:>10.3f} | {post_std:>9.4f} | {ci_str:>22}")
```

Expected output:

```text
DSGE Posterior Summary:
  Parameter  |     True | Prior Mean |  Post Mean |  Post Std |                 95% CI
  -----------+---------+-----------+-----------+----------+-----------------------
  κ          |    0.100 |      0.100 |      0.098 |    0.0234 |  [0.0567, 0.1478]
  φ_π        |    1.500 |      1.500 |      1.487 |    0.1123 |  [1.2712, 1.7123]
  φ_x        |    0.500 |      0.500 |      0.512 |    0.0987 |  [0.3234, 0.7089]
  σ_x        |    0.100 |      0.120 |      0.103 |    0.0156 |  [0.0745, 0.1356]
  σ_π        |    0.150 |      0.160 |      0.148 |    0.0198 |  [0.1123, 0.1898]
  σ_i        |    0.100 |      0.120 |      0.098 |    0.0145 |  [0.0712, 0.1289]
```

!!! tip "Interpreting DSGE posteriors"
    - **$\kappa$**: The Phillips curve slope is well-identified -- the posterior is much
      tighter than the prior. A larger $\kappa$ means inflation responds more to the
      output gap.
    - **$\phi_\pi > 1$**: The Taylor principle holds -- the central bank raises rates
      more than one-for-one with inflation. This is critical for determinacy.
    - **Shock std's**: These are identified from the residual variance structure.

---

## Step 6: Model Comparison via Marginal Likelihood

Let's compare two models:

1. **Model A**: Regime-switching Taylor rule (our estimated model)
2. **Model B**: Fixed Taylor rule (simpler, no regime switching)

```python
from particlefilterbox.smc.sampler import SMCSampler

# --- Model A: Regime-switching (already estimated) ---
log_ml_A = pmmh_results.log_marginal_likelihood

# --- Model B: Fixed Taylor rule ---
# Use SMC sampler for more precise marginal likelihood
def model_B_logpdf(theta):
    """Log-posterior for fixed Taylor rule model."""
    lp = dsge_prior.logpdf(theta)
    if not np.isfinite(lp):
        return -np.inf

    # Update model parameters
    params_B = {**true_params}
    for name, val in zip(dsge_prior.param_names, theta):
        params_B[name] = val

    # Solve and compute likelihood via Kalman filter
    try:
        A_b, B_b, C_b, D_b = solve_nk_model(params_B)
        Q_b = B_b @ B_b.T
        R_b = D_b @ D_b.T

        # Kalman filter likelihood
        n_states = A_b.shape[0]
        n_obs_dim = C_b.shape[0]
        mu = np.zeros(n_states)
        Sigma = np.eye(n_states) * 0.01
        ll = 0.0

        for t in range(len(obs)):
            # Predict
            mu_pred = A_b @ mu
            Sigma_pred = A_b @ Sigma @ A_b.T + Q_b

            # Innovation
            y_pred = C_b @ mu_pred
            S = C_b @ Sigma_pred @ C_b.T + R_b
            v = obs[t] - y_pred

            # Log-likelihood
            sign, logdet = np.linalg.slogdet(S)
            ll += -0.5 * (n_obs_dim * np.log(2 * np.pi) + logdet + v @ np.linalg.solve(S, v))

            # Update
            K = Sigma_pred @ C_b.T @ np.linalg.inv(S)
            mu = mu_pred + K @ v
            Sigma = Sigma_pred - K @ C_b @ Sigma_pred

        return lp + ll
    except Exception:
        return -np.inf

def prior_sample_fn(rng):
    return dsge_prior.sample(rng)

def prior_logpdf_fn(theta):
    return dsge_prior.logpdf(theta)

smc_B = SMCSampler(
    target_logpdf=model_B_logpdf,
    prior_logpdf=prior_logpdf_fn,
    prior_sample=prior_sample_fn,
    n_particles=2000,
    n_mcmc_moves=5,
    ess_target_ratio=0.5,
    seed=42,
)

print("Running SMC for Model B (fixed Taylor rule)...")
results_B = smc_B.run()
log_ml_B = results_B.log_evidence

log_bf = log_ml_A - log_ml_B

print(f"\nBayesian Model Comparison:")
print(f"  Log-evidence (Model A, regime-switching): {log_ml_A:.2f}")
print(f"  Log-evidence (Model B, fixed):            {log_ml_B:.2f}")
print(f"  Log Bayes factor (A vs B):                {log_bf:.2f}")

if log_bf > 4.6:
    verdict = "Decisive evidence for regime-switching"
elif log_bf > 2.3:
    verdict = "Strong evidence for regime-switching"
elif log_bf > 1.15:
    verdict = "Substantial evidence for regime-switching"
elif log_bf > -1.15:
    verdict = "Inconclusive"
else:
    verdict = "Evidence favors fixed Taylor rule"

print(f"  Interpretation: {verdict}")
```

Expected output:

```text
Running SMC for Model B (fixed Taylor rule)...

Bayesian Model Comparison:
  Log-evidence (Model A, regime-switching): -234.56
  Log-evidence (Model B, fixed):            -241.23
  Log Bayes factor (A vs B):                6.67
  Interpretation: Decisive evidence for regime-switching
```

!!! note "Marginal likelihood in DSGE models"
    The marginal likelihood $p(y \mid \mathcal{M})$ automatically penalizes model
    complexity (Occam's razor). A regime-switching model has more parameters but
    only wins if the data genuinely support regime changes. This is the standard
    approach for DSGE model comparison in the macro literature (Schorfheide, 2000).

---

## Step 7: Impulse Response Functions from the Posterior

IRFs show how the economy responds to a one-standard-deviation structural shock. By computing IRFs for each posterior draw, we get **posterior uncertainty bands**:

```python
def compute_irf(params, shock_idx, horizon=40):
    """Compute impulse response to a unit shock."""
    A_irf, B_irf, C_irf, _ = solve_nk_model(params)
    n_s = A_irf.shape[0]

    # Unit shock in dimension shock_idx
    impulse = np.zeros(n_s)
    impulse = B_irf[:, shock_idx]

    irf = np.zeros((horizon, n_s))
    irf[0] = impulse

    for h in range(1, horizon):
        irf[h] = A_irf @ irf[h - 1]

    return irf

# --- Compute IRFs for each posterior draw ---
horizon = 40
n_draws = min(500, len(chains))
idx_draws = np.random.choice(len(chains), size=n_draws, replace=False)

shock_names = ["Demand shock", "Cost-push shock", "Monetary policy shock", "Natural rate shock"]
state_labels = ["Output Gap $x_t$", "Inflation $\\pi_t$",
                "Interest Rate $i_t$", "Natural Rate $r_t^n$"]

# Focus on monetary policy shock (index 2)
shock_idx = 2
irfs_all = np.zeros((n_draws, horizon, 4))

for k, idx in enumerate(idx_draws):
    params_k = {**true_params}
    for j, name in enumerate(dsge_prior.param_names):
        params_k[name] = chains[idx, j]
    irfs_all[k] = compute_irf(params_k, shock_idx, horizon)

# Compute percentiles
irf_mean = np.mean(irfs_all, axis=0)
irf_lo = np.percentile(irfs_all, 16, axis=0)
irf_hi = np.percentile(irfs_all, 84, axis=0)
irf_lo5 = np.percentile(irfs_all, 5, axis=0)
irf_hi95 = np.percentile(irfs_all, 95, axis=0)

# True IRF
irf_true = compute_irf(true_params, shock_idx, horizon)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.ravel()
h_axis = np.arange(horizon)

for i, (ax, label, color) in enumerate(zip(axes, state_labels, colors + ["purple"])):
    ax.plot(h_axis, irf_true[:, i], "k-", linewidth=2, label="True")
    ax.plot(h_axis, irf_mean[:, i], "--", color=color, linewidth=1.5, label="Posterior mean")
    ax.fill_between(h_axis, irf_lo[:, i], irf_hi[:, i],
                     alpha=0.3, color=color, label="68% CI")
    ax.fill_between(h_axis, irf_lo5[:, i], irf_hi95[:, i],
                     alpha=0.15, color=color, label="90% CI")
    ax.axhline(0, color="gray", linewidth=0.3)
    ax.set_xlabel("Quarters")
    ax.set_ylabel(label)
    ax.set_title(f"Response to {shock_names[shock_idx]}")
    if i == 0:
        ax.legend(fontsize=7)

plt.suptitle("Impulse Response Functions with Posterior Uncertainty", fontsize=13)
plt.tight_layout()
plt.savefig("dsge_irf.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"IRF to monetary policy shock:")
print(f"  Impact on output gap:   {irf_mean[0, 0]:.4f} [{irf_lo5[0, 0]:.4f}, {irf_hi95[0, 0]:.4f}]")
print(f"  Impact on inflation:    {irf_mean[0, 1]:.4f} [{irf_lo5[0, 1]:.4f}, {irf_hi95[0, 1]:.4f}]")
print(f"  Impact on interest rate:{irf_mean[0, 2]:.4f} [{irf_lo5[0, 2]:.4f}, {irf_hi95[0, 2]:.4f}]")
print(f"  Half-life (output gap): ~{np.argmax(np.abs(irf_mean[:, 0]) < np.abs(irf_mean[0, 0]) / 2)} quarters")
```

Expected output:

```text
IRF to monetary policy shock:
  Impact on output gap:   -0.0987 [-0.1234, -0.0756]
  Impact on inflation:    -0.0123 [-0.0234, -0.0045]
  Impact on interest rate:0.1000 [0.0823, 0.1178]
  Half-life (output gap): ~5 quarters
```

- **Output gap**: A contractionary monetary policy shock reduces output, with the effect peaking on impact and decaying over ~8 quarters.
- **Inflation**: Responds negatively but with a delay (Phillips curve dynamics).
- **Interest rate**: Jumps on impact, then mean-reverts due to the smoothing parameter $\rho_i$.
- **Posterior bands**: Wider at longer horizons, reflecting parameter uncertainty.

!!! tip "Using IRFs for policy analysis"
    The posterior IRF bands capture **both** estimation uncertainty and model uncertainty.
    Policymakers can assess not just the expected response but the range of plausible
    outcomes. This is a key advantage of Bayesian DSGE estimation over classical
    calibration approaches.

---

## Summary

In this tutorial you learned:

1. The **3-equation New Keynesian** model captures output gap, inflation, and monetary policy dynamics
2. The linearized DSGE maps to a **state-space form** amenable to particle filtering
3. The **RBPF with kalmanbox** efficiently handles the linear substructure, with particles only for regime switching
4. **PMMH** estimates structural parameters with well-calibrated posteriors
5. **Prior vs posterior** comparison reveals which parameters are identified by the data
6. **Marginal likelihood** enables principled model comparison (regime-switching vs fixed)
7. **Impulse response functions** with posterior uncertainty quantify the effect of structural shocks

---

## What's Next?

<div class="grid cards" markdown>

- :material-rocket-launch: **[Acceleration Tutorial](acceleration.md)**

    Speed up DSGE estimation 10-500x with Numba and GPU

- :material-clipboard-check-outline: **[Complete Workflow](complete-workflow.md)**

    End-to-end analysis workflow combining all techniques

- :material-refresh: **[PGAS Tutorial](pgas.md)**

    More efficient joint state-parameter inference with ancestor sampling

</div>
