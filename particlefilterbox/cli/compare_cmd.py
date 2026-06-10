"""Compare command for particlefilterbox CLI.

Compares multiple models on the same dataset using log-evidence.

Usage
-----
    $ pfbox compare data.csv --models sv --n-particles 2000
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from particlefilterbox._logging import get_logger
from particlefilterbox.cli._models import (
    SUPPORTED_MODELS,
    build_filter,
    build_filter_model,
)

logger = get_logger("cli.compare")


def compare_command(
    data: Path = typer.Argument(
        ...,
        help="Path to CSV data file.",
        exists=True,
        readable=True,
    ),
    models: str = typer.Option(
        "sv",
        "--models",
        "-m",
        help=(
            "Comma-separated model names to compare. "
            f"Supported: {', '.join(SUPPORTED_MODELS)}."
        ),
    ),
    method: str = typer.Option(
        "bootstrap",
        "--method",
        help="Filter method used for every model.",
    ),
    n_particles: int = typer.Option(
        1000,
        "--n-particles",
        "-n",
        help="Number of particles for each filter run.",
        min=10,
    ),
    n_runs: int = typer.Option(
        5,
        "--n-runs",
        help="Number of repeated runs for variance estimation.",
        min=1,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path for comparison results.",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        "-s",
        help="Random seed.",
    ),
) -> None:
    """Compare multiple models using log-evidence.

    Runs particle filters for each model, computes log-likelihood
    estimates with variance from repeated runs, and ranks models.
    """
    import numpy as np
    import pandas as pd

    typer.echo(f"Loading data from {data}...")
    try:
        df = pd.read_csv(data)
        observations = df.values.astype(float)
    except Exception as e:  # noqa: BLE001
        typer.echo(f"Error loading data: {e}", err=True)
        raise typer.Exit(code=1) from e

    model_names = [m.strip() for m in models.split(",") if m.strip()]
    typer.echo(f"Comparing models: {', '.join(model_names)}")
    typer.echo(f"Particles: {n_particles}, Runs: {n_runs}")

    results_dict: dict[str, dict[str, float]] = {}

    for model_name in model_names:
        typer.echo(f"\nRunning {model_name}...")
        log_likes: list[float] = []

        for run in range(n_runs):
            try:
                model_instance = build_filter_model(model_name)
                run_seed = seed + run if seed is not None else None
                pf = build_filter(method, model_instance, n_particles, run_seed)
                result = pf.filter(observations)
                ll = getattr(result, "log_likelihood", None)
                if ll is not None:
                    log_likes.append(float(ll))
            except Exception as e:  # noqa: BLE001
                typer.echo(f"  Run {run + 1} failed: {e}", err=True)

        if log_likes:
            results_dict[model_name] = {
                "mean_loglike": float(np.mean(log_likes)),
                "std_loglike": float(np.std(log_likes)),
                "n_runs": len(log_likes),
            }
            typer.echo(
                f"  Log-likelihood: {np.mean(log_likes):.4f} "
                f"(+/- {np.std(log_likes):.4f})"
            )
        else:
            typer.echo(f"  No successful runs for {model_name}", err=True)

    # If nothing succeeded, this is a genuine failure.
    if not results_dict:
        typer.echo("No models produced results.", err=True)
        raise typer.Exit(code=1)

    # Print comparison table.
    typer.echo("\n" + "=" * 60)
    typer.echo("Model Comparison")
    typer.echo("=" * 60)
    typer.echo(f"{'Model':<20} {'Mean LL':>12} {'Std LL':>12} {'Runs':>6}")
    typer.echo("-" * 50)

    sorted_models = sorted(
        results_dict.items(),
        key=lambda x: x[1]["mean_loglike"],
        reverse=True,
    )
    for model_name, stats in sorted_models:
        typer.echo(
            f"{model_name:<20} {stats['mean_loglike']:>12.4f} "
            f"{stats['std_loglike']:>12.4f} {int(stats['n_runs']):>6}"
        )
    typer.echo("=" * 60)

    # Save output.
    if output is not None:
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(results_dict, indent=2))
            typer.echo(f"\nResults saved to {output}")
        except Exception as e:  # noqa: BLE001
            typer.echo(f"Error saving output: {e}", err=True)
            raise typer.Exit(code=1) from e

    typer.echo("Done.")
