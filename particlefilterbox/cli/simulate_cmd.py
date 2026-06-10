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
from particlefilterbox.cli._models import SUPPORTED_MODELS, build_model

logger = get_logger("cli.simulate")


def simulate_command(
    model: str = typer.Option(
        "sv",
        "--model",
        "-m",
        help=f"Model name. Supported: {', '.join(SUPPORTED_MODELS)}.",
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

    # Parse parameters
    param_dict: dict[str, float] = {}
    if params is not None:
        try:
            param_dict = json.loads(params)
            typer.echo(f"Parameters: {param_dict}")
        except json.JSONDecodeError as e:
            typer.echo(f"Invalid JSON parameters: {e}", err=True)
            raise typer.Exit(code=1) from None

    # Build model and simulate.
    try:
        model_instance = build_model(model, param_dict)
        states, observations = _simulate(model_instance, n_obs, seed)
    except Exception as e:  # noqa: BLE001 - surface any failure to the user
        typer.echo(f"Error during simulation: {e}", err=True)
        raise typer.Exit(code=1) from e

    typer.echo(f"Simulated {n_obs} observations")
    typer.echo(f"States shape: {states.shape}")
    typer.echo(f"Observations shape: {observations.shape}")

    # Summary statistics
    obs_flat = observations.flatten()
    typer.echo(f"Obs mean: {np.mean(obs_flat):.4f}, std: {np.std(obs_flat):.4f}")

    # Save output
    if output is not None:
        try:
            import pandas as pd

            output.parent.mkdir(parents=True, exist_ok=True)
            obs2d = observations if observations.ndim > 1 else observations.reshape(-1, 1)
            df = pd.DataFrame(obs2d)
            df.columns = [f"y_{i}" for i in range(obs2d.shape[1])]
            df.to_csv(output, index=False)
            typer.echo(f"Data saved to {output}")
        except Exception as e:  # noqa: BLE001
            typer.echo(f"Error saving output: {e}", err=True)
            raise typer.Exit(code=1) from e

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


def _simulate(model: Any, n_obs: int, seed: int | None) -> tuple[Any, Any]:
    """Simulate ``(states, observations)`` from a model.

    The library models expose ``simulate(T, seed) -> {'observations', 'states'}``.
    """
    import numpy as np

    result = model.simulate(n_obs, seed=seed)
    states = np.asarray(result["states"])
    observations = np.asarray(result["observations"])
    return states, observations
