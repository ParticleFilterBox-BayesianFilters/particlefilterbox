"""Filter command for particlefilterbox CLI.

Runs a particle filter on data and outputs filtered state estimates.

Usage
-----
    $ pfbox filter data.csv --model sv --n-particles 1000 --method bootstrap
    $ pfbox filter data.csv --model sv --n-particles 2000 --output results.json --plot
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from particlefilterbox._logging import get_logger
from particlefilterbox.cli._models import (
    SUPPORTED_METHODS,
    SUPPORTED_MODELS,
    build_filter,
    build_filter_model,
)

logger = get_logger("cli.filter")


def filter_command(
    data: Path = typer.Argument(
        ...,
        help="Path to CSV data file.",
        exists=True,
        readable=True,
    ),
    model: str = typer.Option(
        "sv",
        "--model",
        "-m",
        help=f"Model name. Supported: {', '.join(SUPPORTED_MODELS)}.",
    ),
    n_particles: int = typer.Option(
        1000,
        "--n-particles",
        "-n",
        help="Number of particles.",
        min=10,
    ),
    method: str = typer.Option(
        "bootstrap",
        "--method",
        help=f"Filter method. Supported: {', '.join(SUPPORTED_METHODS)}.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (JSON or CSV).",
    ),
    plot: bool = typer.Option(
        False,
        "--plot",
        "-p",
        help="Generate and display plots.",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        "-s",
        help="Random seed for reproducibility.",
    ),
) -> None:
    """Run a particle filter on the given data.

    Loads data from CSV, configures the specified model and filter method,
    runs the filter, and optionally saves results and generates plots.
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

    typer.echo(f"Model: {model}")
    typer.echo(f"Method: {method}")
    typer.echo(f"Particles: {n_particles}")

    # Configure model + filter and run.
    try:
        model_instance = build_filter_model(model)
        filter_instance = build_filter(method, model_instance, n_particles, seed)
        typer.echo("Running particle filter...")
        results = filter_instance.filter(observations)
    except Exception as e:  # noqa: BLE001
        typer.echo(f"Error during filtering: {e}", err=True)
        raise typer.Exit(code=1) from e

    # Report summary.
    log_likelihood = getattr(results, "log_likelihood", None)
    if log_likelihood is not None:
        typer.echo(f"Log-likelihood: {log_likelihood:.4f}")

    ess_history = getattr(results, "ess_history", None)
    if ess_history is not None:
        ess_arr = np.asarray(ess_history)
        typer.echo(f"Mean ESS: {ess_arr.mean():.1f}")

    n_resampled = getattr(results, "resampled", None)
    if n_resampled is not None:
        typer.echo(f"Resampling steps: {int(np.sum(n_resampled))}")

    # Save output.
    if output is not None:
        try:
            _save_results(results, output)
            typer.echo(f"Results saved to {output}")
        except Exception as e:  # noqa: BLE001
            typer.echo(f"Error saving output: {e}", err=True)
            raise typer.Exit(code=1) from e

    # Generate plots.
    if plot:
        _generate_filter_plots(results, model)

    typer.echo("Done.")


def _save_results(results: Any, path: Path) -> None:
    """Save filter results to file."""
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)

    filtered_means = getattr(results, "filtered_means", None)

    if path.suffix == ".json":
        output_dict: dict[str, Any] = {}
        log_likelihood = getattr(results, "log_likelihood", None)
        if log_likelihood is not None:
            output_dict["log_likelihood"] = float(log_likelihood)
        ess_history = getattr(results, "ess_history", None)
        if ess_history is not None:
            output_dict["ess_history"] = [float(e) for e in np.asarray(ess_history)]
        if filtered_means is not None:
            output_dict["filtered_means"] = np.asarray(filtered_means).tolist()
        path.write_text(json.dumps(output_dict, indent=2))
    elif path.suffix == ".csv":
        import pandas as pd

        if filtered_means is not None:
            df = pd.DataFrame(np.asarray(filtered_means))
            df.columns = [f"state_{i}" for i in range(df.shape[1])]
            df.to_csv(path, index=False)
    else:
        msg = f"Unsupported output format: {path.suffix}. Use .json or .csv."
        raise ValueError(msg)


def _generate_filter_plots(results: Any, model_name: str) -> None:
    """Generate filter diagnostic plots."""
    try:
        import matplotlib.pyplot as plt

        from particlefilterbox.visualization import (
            plot_ess_timeline,
            plot_filtered_state,
            set_theme,
        )

        set_theme("nodesecon")
        plot_filtered_state(results)
        plt.show()

        plot_ess_timeline(results)
        plt.show()
    except ImportError:
        typer.echo("matplotlib not installed. Skipping plots.", err=True)
