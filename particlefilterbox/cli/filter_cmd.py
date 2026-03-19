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
        help="Model name: sv, local_level, linear_gaussian, dsge.",
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
        help="Filter method: bootstrap, apf, ekpf, ukpf.",
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
    df = pd.read_csv(data)
    observations = df.values

    typer.echo(f"Model: {model}")
    typer.echo(f"Method: {method}")
    typer.echo(f"Particles: {n_particles}")

    rng = np.random.default_rng(seed)

    # Configure model
    model_instance = _get_model(model)
    filter_instance = _get_filter(method, model_instance, n_particles, rng)

    typer.echo("Running particle filter...")
    results = filter_instance.filter(observations)

    # Report summary
    log_likelihood = getattr(results, "log_likelihood", None)
    if log_likelihood is not None:
        typer.echo(f"Log-likelihood: {log_likelihood:.4f}")

    ess = getattr(results, "ess", None)
    if ess is not None:
        ess_arr = np.asarray(ess)
        typer.echo(f"Mean ESS: {ess_arr.mean():.1f}")

    # Save output
    if output is not None:
        _save_results(results, output)
        typer.echo(f"Results saved to {output}")

    # Generate plots
    if plot:
        _generate_filter_plots(results, model)

    typer.echo("Done.")


def _get_model(name: str) -> Any:
    """Get model instance by name."""
    try:
        if name == "sv":
            from particlefilterbox.models.sv import SVModel

            return SVModel()
        elif name == "local_level":
            from particlefilterbox.models.local_level import LocalLevelModel

            return LocalLevelModel()
        elif name == "linear_gaussian":
            from particlefilterbox.models.linear_gaussian import LinearGaussianModel

            return LinearGaussianModel()
        else:
            typer.echo(f"Unknown model: {name}. Using default SV model.", err=True)
            from particlefilterbox.models.sv import SVModel

            return SVModel()
    except ImportError as e:
        typer.echo(f"Error importing model '{name}': {e}", err=True)
        raise typer.Exit(code=1) from None


def _get_filter(
    method: str,
    model: Any,
    n_particles: int,
    rng: Any,
) -> Any:
    """Get filter instance by method name."""
    try:
        if method == "bootstrap":
            from particlefilterbox.filters.bootstrap import BootstrapFilter

            return BootstrapFilter(model=model, n_particles=n_particles, rng=rng)
        elif method == "apf":
            from particlefilterbox.filters.apf import AuxiliaryParticleFilter

            return AuxiliaryParticleFilter(model=model, n_particles=n_particles, rng=rng)
        else:
            typer.echo(f"Unknown method: {method}. Using bootstrap.", err=True)
            from particlefilterbox.filters.bootstrap import BootstrapFilter

            return BootstrapFilter(model=model, n_particles=n_particles, rng=rng)
    except ImportError as e:
        typer.echo(f"Error importing filter '{method}': {e}", err=True)
        raise typer.Exit(code=1) from None


def _save_results(results: Any, path: Path) -> None:
    """Save filter results to file."""
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".json":
        output_dict: dict[str, Any] = {}
        log_likelihood = getattr(results, "log_likelihood", None)
        if log_likelihood is not None:
            output_dict["log_likelihood"] = float(log_likelihood)
        ess = getattr(results, "ess", None)
        if ess is not None:
            output_dict["ess"] = [float(e) for e in np.asarray(ess)]
        filtered_mean = getattr(results, "filtered_mean", None)
        if filtered_mean is not None:
            output_dict["filtered_mean"] = np.asarray(filtered_mean).tolist()
        path.write_text(json.dumps(output_dict, indent=2))
    elif path.suffix == ".csv":
        import pandas as pd

        filtered_mean = getattr(results, "filtered_mean", None)
        if filtered_mean is not None:
            df = pd.DataFrame(np.asarray(filtered_mean))
            df.to_csv(path, index=False)
    else:
        typer.echo(f"Unsupported output format: {path.suffix}", err=True)


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
        fig, ax = plot_filtered_state(results)
        plt.show()

        fig, ax = plot_ess_timeline(results)
        plt.show()
    except ImportError:
        typer.echo("matplotlib not installed. Skipping plots.", err=True)
