"""Generate datasets for application examples."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_jump_diffusion(
    n: int = 1000,
    mu: float = 0.0005,
    sigma: float = 0.01,
    lam: float = 0.05,
    mu_j: float = -0.02,
    sigma_j: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate Merton jump-diffusion returns.

    r_t = mu + sigma*eps_t + J_t*Z_t
    J_t ~ Bernoulli(lam), Z_t ~ N(mu_j, sigma_j^2)
    """
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    jumps = rng.binomial(1, lam, n)
    jump_sizes = rng.normal(mu_j, sigma_j, n)
    returns = mu + sigma * eps + jumps * jump_sizes

    return pd.DataFrame({
        "t": np.arange(n),
        "returns": np.round(returns, 8),
        "jump_true": jumps,
        "jump_size_true": np.round(jumps * jump_sizes, 8),
    })


def generate_sir_epidemic(
    n: int = 200,
    N_pop: int = 10000,
    beta: float = 0.3,
    gamma: float = 0.1,
    obs_rate: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate SIR epidemic data with partial observation.

    S_{t+1} = S_t - beta*S_t*I_t/N
    I_{t+1} = I_t + beta*S_t*I_t/N - gamma*I_t
    R_{t+1} = R_t + gamma*I_t
    Observed: y_t ~ Poisson(obs_rate * I_t)
    """
    rng = np.random.default_rng(seed)
    S = np.zeros(n)
    I_true = np.zeros(n)
    R = np.zeros(n)
    y_obs = np.zeros(n, dtype=int)

    S[0] = N_pop - 10
    I_true[0] = 10
    R[0] = 0

    for t in range(1, n):
        new_inf = beta * S[t-1] * I_true[t-1] / N_pop
        new_rec = gamma * I_true[t-1]
        # Add stochasticity
        new_inf = rng.poisson(max(new_inf, 0))
        new_rec = rng.poisson(max(new_rec, 0))
        new_inf = min(new_inf, S[t-1])
        new_rec = min(new_rec, I_true[t-1])

        S[t] = S[t-1] - new_inf
        I_true[t] = I_true[t-1] + new_inf - new_rec
        R[t] = R[t-1] + new_rec

    for t in range(n):
        y_obs[t] = rng.poisson(max(obs_rate * I_true[t], 0.1))

    return pd.DataFrame({
        "t": np.arange(n),
        "S_true": S.astype(int),
        "I_true": I_true.astype(int),
        "R_true": R.astype(int),
        "y_obs": y_obs,
    })


def generate_treasury_yields(
    n: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic treasury yields for DSGE example."""
    rng = np.random.default_rng(seed)

    # Simple AR(1) for output gap, inflation, interest rate
    x = np.zeros(n)  # output gap
    pi_t = np.zeros(n)  # inflation
    r = np.zeros(n)  # interest rate

    for t in range(1, n):
        x[t] = 0.9 * x[t-1] + 0.1 * rng.standard_normal()
        pi_t[t] = 0.5 * pi_t[t-1] + 0.3 * x[t] + 0.1 * rng.standard_normal()
        r[t] = 0.8 * r[t-1] + 0.5 * pi_t[t] + 0.2 * x[t] + 0.05 * rng.standard_normal()

    # Observed with noise
    y_x = x + 0.2 * rng.standard_normal(n)
    y_pi = pi_t + 0.15 * rng.standard_normal(n)
    y_r = r + 0.1 * rng.standard_normal(n)

    return pd.DataFrame({
        "t": np.arange(n),
        "output_gap_obs": np.round(y_x, 6),
        "inflation_obs": np.round(y_pi, 6),
        "interest_rate_obs": np.round(y_r, 6),
        "output_gap_true": np.round(x, 6),
        "inflation_true": np.round(pi_t, 6),
        "interest_rate_true": np.round(r, 6),
    })


if __name__ == "__main__":
    import os

    data_dir = os.path.dirname(os.path.abspath(__file__))

    df_jd = generate_jump_diffusion()
    df_jd.to_csv(os.path.join(data_dir, "simulated_jump_diff.csv"), index=False)
    print(f"Jump-Diffusion: {len(df_jd)} obs, jumps={df_jd['jump_true'].sum()}")

    df_sir = generate_sir_epidemic()
    df_sir.to_csv(os.path.join(data_dir, "simulated_sir.csv"), index=False)
    print(f"SIR Epidemic: {len(df_sir)} obs, peak I={df_sir['I_true'].max()}")

    df_ty = generate_treasury_yields()
    df_ty.to_csv(os.path.join(data_dir, "treasury_yields.csv"), index=False)
    print(f"Treasury Yields: {len(df_ty)} obs")
