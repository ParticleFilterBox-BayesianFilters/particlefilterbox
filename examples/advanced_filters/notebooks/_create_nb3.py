"""Create and execute 03_unscented_regularized.ipynb."""
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

cells = []

# Cell 1: Title
cells.append(nbf.v4.new_markdown_cell("""# Unscented PF and Regularized PF for Stochastic Volatility

This notebook demonstrates two advanced particle filter variants applied to
the Stochastic Volatility (SV) model:

1. **Unscented Particle Filter (UPF)** — van der Merwe et al. (2001): Uses the
   Unscented Kalman Filter (UKF) as a proposal distribution. Instead of sampling
   blindly from the transition prior, each particle uses sigma points to construct
   a proposal that incorporates the current observation. This yields more efficient
   proposals and better weight distributions.

2. **Regularized Particle Filter (RPF)** — Musso et al. (2001): After resampling,
   applies kernel density jittering to particles. Instead of having many duplicate
   particles after resampling, each particle is perturbed by a kernel (e.g.,
   Gaussian with Silverman bandwidth), maintaining particle diversity."""))

# Cell 2: Math
cells.append(nbf.v4.new_markdown_cell(r"""## Mathematical Description

### Stochastic Volatility Model
$$h_t = \mu + \phi(h_{t-1} - \mu) + \sigma_\eta \eta_t, \quad \eta_t \sim N(0,1)$$
$$y_t = \exp(h_t/2) \varepsilon_t, \quad \varepsilon_t \sim N(0,1)$$

### Unscented PF (UPF)

For each particle $i$, the UPF:
1. Generates **sigma points** around the current state estimate $(m^{(i)}, P^{(i)})$
2. Propagates sigma points through the **transition function** $f(x)$
3. Propagates through the **observation function** $h(x)$
4. Computes UKF posterior: $m_{UKF}^{(i)}, P_{UKF}^{(i)}$
5. Samples: $x_t^{(i)} \sim N(m_{UKF}^{(i)}, P_{UKF}^{(i)})$
6. Importance weight: $w_t^{(i)} = \frac{p(y_t|x_t^{(i)}) \cdot p(x_t^{(i)}|x_{t-1}^{(i)})}{q(x_t^{(i)}; m_{UKF}, P_{UKF})}$

### Regularized PF (RPF)

Standard Bootstrap PF with modified resampling:
1. Compute weights and resample as usual
2. After resampling, **jitter** each particle: $\tilde{x}^{(i)} = x^{(i)} + h \cdot K_i$
   - $h$: bandwidth (Silverman's rule: $h = \left(\frac{4}{N(d+2)}\right)^{1/(d+4)} \cdot \hat\sigma$)
   - $K_i$: kernel sample (Gaussian or Epanechnikov)"""))

# Cell 3: Imports and data
cells.append(nbf.v4.new_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..')))

from particlefilterbox.core.config import PFConfig
from particlefilterbox.core.model import ParticleFilterModel
from particlefilterbox.filters.bootstrap import BootstrapPF
from particlefilterbox.filters.unscented import UnscentedPF
from particlefilterbox.filters.regularized import RegularizedPF

# Load simulated SV data
data_dir = os.path.join(os.getcwd(), '..', '..', 'bootstrap_sir', 'data')
df = pd.read_csv(os.path.join(data_dir, 'simulated_sv.csv'))
h_true = df['h_true'].values
y_obs = df['y_obs'].values
T = len(y_obs)

print(f"Loaded {T} observations from SV model")
print(f"h_true range: [{h_true.min():.3f}, {h_true.max():.3f}]")"""))

# Cell 4: SV model with UPF-compatible interface
cells.append(nbf.v4.new_code_cell("""class SVModelFull(ParticleFilterModel):
    \"\"\"SV model with full interface for UPF/RPF/BPF.

    Provides transition_function, observation_function, Q, R_obs
    for UnscentedPF, plus standard interface for Bootstrap/Regularized.
    \"\"\"
    k_states = 1
    k_obs = 1

    def __init__(self, mu: float = -1.0, phi: float = 0.97, sigma: float = 0.15):
        self.mu = mu
        self.phi = phi
        self.sigma = sigma
        self.sigma_x = sigma  # alias for UPF
        self.sigma_y = 1.0    # observation noise scale (not directly used)

    @property
    def params(self) -> dict[str, float]:
        return {"mu": self.mu, "phi": self.phi, "sigma": self.sigma}

    def initial_distribution(self, n_particles: int, rng: np.random.Generator) -> np.ndarray:
        var_stat = self.sigma**2 / (1.0 - self.phi**2)
        return rng.normal(self.mu, np.sqrt(var_stat), size=(n_particles, 1))

    def transition(self, particles: np.ndarray, t: int, rng: np.random.Generator) -> np.ndarray:
        eta = rng.standard_normal(size=particles.shape)
        return self.mu + self.phi * (particles - self.mu) + self.sigma * eta

    def transition_function(self, x: np.ndarray, t: int) -> np.ndarray:
        \"\"\"Deterministic transition (for UPF sigma points).\"\"\"
        return self.mu + self.phi * (np.atleast_1d(x) - self.mu)

    def transition_mean(self, particles: np.ndarray, t: int) -> np.ndarray:
        return self.mu + self.phi * (particles - self.mu)

    def observation_function(self, x: np.ndarray, t: int) -> np.ndarray:
        \"\"\"Deterministic observation mapping h -> E[y|h] = 0 (mean of y given h).
        For UPF we map to zero since y_t ~ N(0, exp(h_t)).\"\"\"
        return np.zeros(1)

    def Q(self, t: int) -> np.ndarray:
        \"\"\"Process noise covariance.\"\"\"
        return np.array([[self.sigma**2]])

    def R_obs(self, t: int) -> np.ndarray:
        \"\"\"Observation noise covariance.
        For SV, this is approximate — true obs is non-Gaussian.
        We use exp(mu) as a reasonable scale.\"\"\"
        return np.array([[np.exp(self.mu)]])

    def log_observation_likelihood(self, particles: np.ndarray, y_t: np.ndarray, t: int) -> np.ndarray:
        h = particles[:, 0]
        vol = np.exp(h / 2.0)
        return -0.5 * np.log(2 * np.pi) - np.log(vol) - 0.5 * (float(y_t[0]) / vol) ** 2

model = SVModelFull(mu=-1.0, phi=0.97, sigma=0.15)
print(f"SV Model: {model.params}")"""))

# Cell 5: Run all three filters
cells.append(nbf.v4.new_code_cell("""N = 500

# Unscented PF
config_upf = PFConfig(n_particles=N, resampling="systematic", seed=42)
upf = UnscentedPF(model=model, config=config_upf, alpha=1.0, beta=2.0, kappa=0.0)

t0 = time.time()
res_upf = upf.filter(y_obs)
time_upf = time.time() - t0
h_upf = res_upf.filtered_means[:, 0]
rmse_upf = np.sqrt(np.mean((h_upf - h_true) ** 2))

# Regularized PF
config_rpf = PFConfig(n_particles=N, resampling="systematic", seed=42)
rpf = RegularizedPF(model=model, config=config_rpf, bandwidth='silverman', kernel='gaussian')

t0 = time.time()
res_rpf = rpf.filter(y_obs)
time_rpf = time.time() - t0
h_rpf = res_rpf.filtered_means[:, 0]
rmse_rpf = np.sqrt(np.mean((h_rpf - h_true) ** 2))

# Bootstrap PF (baseline)
config_bpf = PFConfig(n_particles=N, resampling="systematic", seed=42)
bpf = BootstrapPF(model=model, config=config_bpf)

t0 = time.time()
res_bpf = bpf.filter(y_obs)
time_bpf = time.time() - t0
h_bpf = res_bpf.filtered_means[:, 0]
rmse_bpf = np.sqrt(np.mean((h_bpf - h_true) ** 2))

print(f"Unscented PF:   RMSE={rmse_upf:.4f}, ESS_mean={res_upf.ess_history.mean():.1f}, time={time_upf:.2f}s")
print(f"Regularized PF: RMSE={rmse_rpf:.4f}, ESS_mean={res_rpf.ess_history.mean():.1f}, time={time_rpf:.2f}s")
print(f"Bootstrap PF:   RMSE={rmse_bpf:.4f}, ESS_mean={res_bpf.ess_history.mean():.1f}, time={time_bpf:.2f}s")"""))

# Cell 6: Plot filtered log-volatility
cells.append(nbf.v4.new_code_cell("""# Plot 1: Filtered log-volatility (3 filters vs true)
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(h_true, color='black', linewidth=0.8, alpha=0.5, label='True $h_t$')
ax.plot(h_upf, color='tab:red', linewidth=0.8, alpha=0.8,
        label=f'Unscented PF (RMSE={rmse_upf:.4f})')
ax.plot(h_rpf, color='tab:green', linewidth=0.8, alpha=0.8,
        label=f'Regularized PF (RMSE={rmse_rpf:.4f})')
ax.plot(h_bpf, color='tab:blue', linewidth=0.8, alpha=0.7, linestyle='--',
        label=f'Bootstrap PF (RMSE={rmse_bpf:.4f})')

ax.set_xlabel('Time')
ax.set_ylabel('Log-volatility $h_t$')
ax.set_title(f'Filtered Log-Volatility: UPF vs RPF vs BPF (N={N})')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""))

# Cell 7: Comparison table
cells.append(nbf.v4.new_code_cell("""# Table: RMSE, ESS, time
summary = pd.DataFrame({
    'Filter': ['Unscented PF', 'Regularized PF', 'Bootstrap PF'],
    'N': [N, N, N],
    'RMSE': [rmse_upf, rmse_rpf, rmse_bpf],
    'ESS_mean': [res_upf.ess_history.mean(), res_rpf.ess_history.mean(), res_bpf.ess_history.mean()],
    'ESS_min': [res_upf.ess_history.min(), res_rpf.ess_history.min(), res_bpf.ess_history.min()],
    'Log-likelihood': [res_upf.log_likelihood, res_rpf.log_likelihood, res_bpf.log_likelihood],
    'Time (s)': [time_upf, time_rpf, time_bpf],
})
summary = summary.round(4)

print("=" * 90)
print("COMPARISON TABLE: Unscented PF vs Regularized PF vs Bootstrap PF")
print("=" * 90)
print(summary.to_string(index=False))
print("=" * 90)"""))

# Cell 8: ESS comparison
cells.append(nbf.v4.new_code_cell("""# Plot 2: ESS over time
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(res_upf.ess_history, color='tab:red', linewidth=0.5, alpha=0.7,
        label=f'UPF (mean={res_upf.ess_history.mean():.1f})')
ax.plot(res_rpf.ess_history, color='tab:green', linewidth=0.5, alpha=0.7,
        label=f'RPF (mean={res_rpf.ess_history.mean():.1f})')
ax.plot(res_bpf.ess_history, color='tab:blue', linewidth=0.5, alpha=0.7,
        label=f'BPF (mean={res_bpf.ess_history.mean():.1f})')
ax.axhline(y=N * 0.5, color='gray', linestyle='--', alpha=0.5, label='Resample threshold')
ax.set_xlabel('Time')
ax.set_ylabel('ESS')
ax.set_title(f'Effective Sample Size Over Time (N={N})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""))

# Cell 9: Posterior distribution at t=250, 500, 750
cells.append(nbf.v4.new_code_cell("""# Plot 3: Posterior particle distribution at t=250, 500, 750
# Re-run filters with store_particles to get particle clouds at specific times
# Since we can't easily store all particles, we'll re-run at specific checkpoints

checkpoints = [250, 500, 750]
checkpoint_particles = {cp: {} for cp in checkpoints}

# Run step-by-step for each filter to capture particles at checkpoints
for filter_name, filter_cls, filter_kwargs in [
    ('UPF', UnscentedPF, {'alpha': 1.0, 'beta': 2.0, 'kappa': 0.0}),
    ('RPF', RegularizedPF, {'bandwidth': 'silverman', 'kernel': 'gaussian'}),
    ('BPF', BootstrapPF, {}),
]:
    config = PFConfig(n_particles=N, resampling="systematic", seed=42, store_particles=True)
    pf = filter_cls(model=model, config=config, **filter_kwargs)
    res = pf.filter(y_obs)

    # If final cloud is available, we can at least show the final distribution
    # For intermediate times, we'll use the filtered means and add noise to approximate
    # the posterior spread
    for cp in checkpoints:
        # Use filtered mean and covariance to approximate posterior
        mean = res.filtered_means[cp, 0]
        var = res.filtered_covs[cp, 0, 0] if res.filtered_covs[cp, 0, 0] > 0 else 0.01
        # Generate approximate posterior samples for visualization
        rng = np.random.default_rng(42)
        checkpoint_particles[cp][filter_name] = rng.normal(mean, np.sqrt(var), size=N)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, cp in enumerate(checkpoints):
    ax = axes[idx]
    for fname, color in [('UPF', 'tab:red'), ('RPF', 'tab:green'), ('BPF', 'tab:blue')]:
        parts = checkpoint_particles[cp][fname]
        ax.hist(parts, bins=40, color=color, alpha=0.4, density=True, label=fname)
    ax.axvline(h_true[cp], color='black', linewidth=2, linestyle='--', label=f'True h={h_true[cp]:.2f}')
    ax.set_xlabel('$h_t$')
    ax.set_ylabel('Density')
    ax.set_title(f't = {cp}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Approximate Posterior Distribution of Particles', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()"""))

# Cell 10: Conclusion
cells.append(nbf.v4.new_markdown_cell("""## Conclusion: Trade-offs Between Methods

### Unscented PF (UPF)
- **Pros**: Uses observation-informed proposals via UKF, potentially lower RMSE
  for the same N. Better weight distribution.
- **Cons**: Significantly slower due to per-particle UKF computations (sigma points,
  matrix operations). The SV model's non-Gaussian observation makes the UKF
  approximation less ideal than for truly Gaussian models.
- **Best for**: Models where the observation is informative and the state dimension
  is small (UKF cost scales with $O(k^3)$ per particle).

### Regularized PF (RPF)
- **Pros**: Prevents sample impoverishment after resampling. Simple to implement
  on top of any existing PF. Kernel jittering maintains particle diversity.
- **Cons**: Bandwidth selection is critical — too large and you blur the posterior,
  too small and it has no effect. The Silverman rule is a reasonable default.
- **Best for**: Low-dimensional problems with resampling degeneracy.
  Less effective in high dimensions where kernel density estimation breaks down.

### Bootstrap PF (BPF)
- **Pros**: Simplest algorithm, fastest per-step computation, no tuning parameters.
- **Cons**: Can suffer from weight degeneracy when the likelihood is informative.
  Requires more particles to achieve the same accuracy as UPF/RPF.
- **Best for**: Quick prototyping, benchmarking, and when computational budget
  per step is limited.

### Summary
| Method | RMSE | Speed | Complexity | Best Use Case |
|--------|------|-------|------------|---------------|
| UPF | Often lowest | Slowest | High | Small state, informative obs |
| RPF | Similar to BPF | Fast | Low | Resampling degeneracy |
| BPF | Baseline | Fastest | Lowest | General purpose |

### References
- van der Merwe, R. et al. (2001). *The Unscented Particle Filter*. NIPS 13.
- Musso, C. et al. (2001). *Improving regularised particle filters*. SMC Methods in Practice."""))

nb.cells = cells

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '03_unscented_regularized.ipynb')
with open(out_path, 'w') as f:
    nbf.write(nb, f)
print(f"Wrote {out_path} with {len(cells)} cells")
