"""Simulate command for particlefilterbox CLI.

Simulates data from a specified model.

Usage
-----
    $ pfbox simulate --model sv --n-obs 500 --seed 42
    $ pfbox simulate --model sv --n-obs 1000 --output sim.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from particlefilterbox._logging import get_logger

logger = get_logger("cli.simulate")


def simulate_command(
    model: str = typer.Option(
        "sv",
        "--model",
        "-m",
        help="Model name: sv, local_level, linear_gaussian.",
    ),
    n_obs: int = typer.Option(
        500,
        "--n-obs",
        "-n",
        help="Number of observations to simulate.",
        min=10,
    ),
    params: str | None = typer.Option(
        None,
        "--params",
        help="Model parameters as JSON string.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output CSV file path.",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        "-s",
        help="Random seed for reproducibility.",
    ),
    plot: bool = typer.Option(
        False,
        "--plot",
        "-p",
        help="Plot simulated data.",
    ),
) -> None:
    """Simulate data from a state-space model.

    Generates synthetic observations and latent states from the specified
    model with given or default parameters.
    """
    import numpy as np

    typer.echo(f"Simulating from model: {model}")
    typer.echo(f"Observations: {n_obs}")

    rng = np.random.default_rng(seed)

    # Parse parameters
    param_dict: dict[str, float] = {}
    if params is not None:
        try:
            param_dict = json.loads(params)
            typer.echo(f"Parameters: {param_dict}")
        except json.JSONDecodeError as e:
            typer.echo(f"Invalid JSON parameters: {e}", err=True)
            raise typer.Exit(code=1) from None

    # Simulate
    try:
        model_instance = _get_model(model, param_dict)
        states, observations = _simulate(model_instance, n_obs, rng)
    except Exception as e:
        typer.echo(f"Error during simulation: {e}", err=True)
        raise typer.Exit(code=1) from None

    typer.echo(f"Simulated {n_obs} observations")
    typer.echo(f"States shape: {states.shape}")
    typer.echo(f"Observations shape: {observations.shape}")

    # Summary statistics
    obs_flat = observations.flatten()
    typer.echo(f"Obs mean: {np.mean(obs_flat):.4f}, std: {np.std(obs_flat):.4f}")

    # Save output
    if output is not None:
        import pandas as pd

        output.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(observations)
        n_cols = observations.shape[1] if observations.ndim > 1 else 1
        df.columns = [f"y_{i}" for i in range(n_cols)]
        df.to_csv(output, index=False)
        typer.echo(f"Data saved to {output}")

    # Plot
    if plot:
        try:
            import matplotlib.pyplot as plt

            from particlefilterbox.visualization import set_theme

            set_theme("nodesecon")

            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            axes[0].plot(states[:, 0] if states.ndim > 1 else states, linewidth=0.8)
            axes[0].set_title("Latent States")
            axes[0].set_ylabel("State")

            axes[1].plot(obs_flat, linewidth=0.8, alpha=0.7)
            axes[1].set_title("Observations")
            axes[1].set_xlabel("Time")
            axes[1].set_ylabel("Observation")

            plt.tight_layout()
            plt.show()
        except ImportError:
            typer.echo("matplotlib not installed. Skipping plot.", err=True)

    typer.echo("Done.")


def _get_model(name: str, params: dict[str, float]) -> Any:
    """Get model instance with optional parameters."""
    try:
        if name == "sv":
            from particlefilterbox.models.sv import SVModel

            return SVModel(**params) if params else SVModel()
        elif name == "local_level":
            from particlefilterbox.models.local_level import LocalLevelModel

            return LocalLevelModel(**params) if params else LocalLevelModel()
        elif name == "linear_gaussian":
            from particlefilterbox.models.linear_gaussian import LinearGaussianModel

            return LinearGaussianModel(**params) if params else LinearGaussianModel()
        else:
            msg = f"Unknown model: {name}"
            raise ValueError(msg)
    except ImportError as e:
        typer.echo(f"Error importing model '{name}': {e}", err=True)
        raise typer.Exit(code=1) from None


def _simulate(model: Any, n_obs: int, rng: Any) -> tuple[Any, Any]:
    """Simulate from a model."""
    import numpy as np

    simulate_fn = getattr(model, "simulate", None)
    if simulate_fn is not None:
        result = simulate_fn(n_obs=n_obs, rng=rng)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        states = getattr(result, "states", None)
        obs = getattr(result, "observations", None)
        if states is not None and obs is not None:
            return np.asarray(states), np.asarray(obs)

    # Fallback: manual simulation
    state_list = []
    obs_list = []
    x = np.zeros(1)

    for t in range(n_obs):
        if hasattr(model, "transition"):
            x = model.transition(x, t, rng)
        else:
            x = x + rng.standard_normal(x.shape) * 0.1
        if hasattr(model, "observation"):
            y = model.observation(x, t, rng)
        else:
            y = x + rng.standard_normal(x.shape) * 0.5
        state_list.append(x.copy())
        obs_list.append(y.copy())

    return np.array(state_list), np.array(obs_list)
