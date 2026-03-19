"""Estimate command for particlefilterbox CLI.

Estimates model parameters using PMCMC methods.

Usage
-----
    $ pfbox estimate data.csv --model sv --method pmmh --n-iterations 5000
    $ pfbox estimate data.csv --model sv --n-particles 500 --n-iterations 10000
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from particlefilterbox._logging import get_logger

logger = get_logger("cli.estimate")


def estimate_command(
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
        help="Model name: sv, local_level, dsge.",
    ),
    method: str = typer.Option(
        "pmmh",
        "--method",
        help="Estimation method: pmmh, particle_gibbs, smc2.",
    ),
    n_particles: int = typer.Option(
        500,
        "--n-particles",
        "-n",
        help="Number of particles for the filter.",
        min=10,
    ),
    n_iterations: int = typer.Option(
        5000,
        "--n-iterations",
        "-i",
        help="Number of MCMC iterations.",
        min=100,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path for results.",
    ),
    plot: bool = typer.Option(
        False,
        "--plot",
        "-p",
        help="Generate diagnostic plots.",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        "-s",
        help="Random seed.",
    ),
) -> None:
    """Estimate model parameters using PMCMC.

    Loads data, configures PMCMC estimation, runs the specified number
    of iterations, and reports posterior summaries.
    """
    import numpy as np
    import pandas as pd

    typer.echo(f"Loading data from {data}...")
    df = pd.read_csv(data)
    observations = df.values

    typer.echo(f"Model: {model}")
    typer.echo(f"Method: {method}")
    typer.echo(f"Particles: {n_particles}")
    typer.echo(f"Iterations: {n_iterations}")

    rng = np.random.default_rng(seed)

    typer.echo("Running parameter estimation...")
    typer.echo("(This may take a while...)")

    # Build and run estimator
    try:
        estimator = _get_estimator(
            method=method,
            model_name=model,
            n_particles=n_particles,
            n_iterations=n_iterations,
            rng=rng,
        )
        results = estimator.run(observations)
    except Exception as e:
        typer.echo(f"Error during estimation: {e}", err=True)
        raise typer.Exit(code=1) from None

    # Report posterior summary
    chain = getattr(results, "chain", None)
    param_names = getattr(results, "param_names", None)

    if chain is not None:
        chain_arr = np.asarray(chain)
        burn_in = chain_arr.shape[0] // 4
        post = chain_arr[burn_in:]

        if param_names is None:
            param_names = [f"param_{i}" for i in range(post.shape[1])]

        typer.echo("\nPosterior Summary (after 25% burn-in):")
        typer.echo(f"{'Parameter':<15} {'Mean':>10} {'Std':>10} {'2.5%':>10} {'97.5%':>10}")
        typer.echo("-" * 55)
        for j, name in enumerate(param_names):
            samples = post[:, j]
            typer.echo(
                f"{name:<15} {np.mean(samples):>10.4f} {np.std(samples):>10.4f} "
                f"{np.percentile(samples, 2.5):>10.4f} "
                f"{np.percentile(samples, 97.5):>10.4f}"
            )

    acceptance_rate = getattr(results, "acceptance_rate", None)
    if acceptance_rate is not None:
        typer.echo(f"\nAcceptance rate: {acceptance_rate:.4f}")

    # Save output
    if output is not None:
        _save_estimation_results(results, output)
        typer.echo(f"Results saved to {output}")

    # Generate plots
    if plot:
        _generate_estimation_plots(results)

    typer.echo("Done.")


def _get_estimator(
    method: str,
    model_name: str,
    n_particles: int,
    n_iterations: int,
    rng: Any,
) -> Any:
    """Get PMCMC estimator by method name."""
    try:
        if method == "pmmh":
            from particlefilterbox.pmcmc.pmmh import PMMH

            model = _get_model_for_estimation(model_name)
            return PMMH(
                model=model,
                n_particles=n_particles,
                n_iterations=n_iterations,
                rng=rng,
            )
        else:
            typer.echo(f"Unknown method: {method}. Using PMMH.", err=True)
            from particlefilterbox.pmcmc.pmmh import PMMH

            model = _get_model_for_estimation(model_name)
            return PMMH(
                model=model,
                n_particles=n_particles,
                n_iterations=n_iterations,
                rng=rng,
            )
    except ImportError as e:
        typer.echo(f"Error importing estimator: {e}", err=True)
        raise typer.Exit(code=1) from None


def _get_model_for_estimation(name: str) -> Any:
    """Get model instance for estimation."""
    try:
        if name == "sv":
            from particlefilterbox.models.sv import SVModel

            return SVModel()
        else:
            typer.echo(f"Unknown model: {name}. Using SV.", err=True)
            from particlefilterbox.models.sv import SVModel

            return SVModel()
    except ImportError as e:
        typer.echo(f"Error importing model: {e}", err=True)
        raise typer.Exit(code=1) from None


def _save_estimation_results(results: Any, path: Path) -> None:
    """Save estimation results."""
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".json":
        output_dict: dict[str, Any] = {}
        chain = getattr(results, "chain", None)
        if chain is not None:
            output_dict["chain_shape"] = list(np.asarray(chain).shape)
        param_names = getattr(results, "param_names", None)
        if param_names is not None:
            output_dict["param_names"] = list(param_names)
        acceptance_rate = getattr(results, "acceptance_rate", None)
        if acceptance_rate is not None:
            output_dict["acceptance_rate"] = float(acceptance_rate)
        path.write_text(json.dumps(output_dict, indent=2))
    else:
        typer.echo(f"Unsupported format: {path.suffix}. Using JSON.", err=True)


def _generate_estimation_plots(results: Any) -> None:
    """Generate estimation diagnostic plots."""
    try:
        import matplotlib.pyplot as plt

        from particlefilterbox.visualization import plot_posterior, plot_trace, set_theme

        set_theme("nodesecon")

        param_names = getattr(results, "param_names", None)
        if param_names:
            for name in param_names:
                fig, ax = plot_trace(results, param=name)
                plt.show()
                fig, ax = plot_posterior(results, param=name)
                plt.show()
    except ImportError:
        typer.echo("matplotlib not installed. Skipping plots.", err=True)
