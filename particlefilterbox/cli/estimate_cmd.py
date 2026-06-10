"""Estimate command for particlefilterbox CLI.

Estimates model parameters using Particle Marginal Metropolis-Hastings (PMMH).

Usage
-----
    $ pfbox estimate data.csv --model sv --n-particles 200 --n-iterations 5000
    $ pfbox estimate data.csv --model sv --output chains.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from particlefilterbox._logging import get_logger
from particlefilterbox.cli._models import (
    SUPPORTED_ESTIMATORS,
    SUPPORTED_MODELS,
    build_pmmh,
)

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
        help=f"Model name. Supported: {', '.join(SUPPORTED_MODELS)}.",
    ),
    method: str = typer.Option(
        "pmmh",
        "--method",
        help=f"Estimation method. Supported: {', '.join(SUPPORTED_ESTIMATORS)}.",
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
        help="Output file path for the posterior chains (.json or .csv).",
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
    """Estimate model parameters using PMCMC (PMMH).

    Loads data from CSV, builds the model adapter and prior, runs the PMMH
    sampler, and prints a posterior summary (per-parameter mean and std after
    burn-in, plus the acceptance rate). Optionally saves the chains.
    """
    import numpy as np
    import pandas as pd

    if method not in SUPPORTED_ESTIMATORS:
        typer.echo(
            f"Unknown method '{method}'. Supported methods: "
            f"{', '.join(SUPPORTED_ESTIMATORS)}.",
            err=True,
        )
        raise typer.Exit(code=1)

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
    typer.echo(f"Iterations: {n_iterations}")

    # Build the PMMH sampler and run.
    try:
        pmmh, adapter, _prior = build_pmmh(
            model_name=model,
            n_particles=n_particles,
            n_iterations=n_iterations,
            seed=seed,
        )
    except Exception as e:  # noqa: BLE001
        typer.echo(f"Error configuring estimator: {e}", err=True)
        raise typer.Exit(code=1) from e

    typer.echo("Running PMMH...")
    try:
        results = pmmh.run(observations)
    except Exception as e:  # noqa: BLE001
        typer.echo(f"Error during estimation: {e}", err=True)
        raise typer.Exit(code=1) from e

    # Report posterior summary.
    param_names = results.param_names or adapter.param_names
    means = results.posterior_mean()
    stds = results.posterior_std()
    accept_rate = results.acceptance_rate()

    typer.echo("")
    typer.echo("Posterior summary (post burn-in):")
    typer.echo(f"  {'parameter':<12}{'mean':>14}{'std':>14}")
    for name, mean, std in zip(
        param_names, np.atleast_1d(means), np.atleast_1d(stds), strict=True
    ):
        typer.echo(f"  {name:<12}{float(mean):>14.6f}{float(std):>14.6f}")
    typer.echo(f"Acceptance rate: {accept_rate:.4f}")
    typer.echo(f"Burn-in: {results.burnin}")
    typer.echo(f"Effective samples: {results.n_effective_samples}")

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
        _generate_estimate_plots(results)

    typer.echo("Done.")


def _save_results(results: Any, path: Path) -> None:
    """Save posterior chains to file (.json or .csv)."""
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)

    param_names = results.param_names
    chains = np.asarray(results.chains)

    if path.suffix == ".json":
        means = np.atleast_1d(results.posterior_mean())
        stds = np.atleast_1d(results.posterior_std())
        output_dict: dict[str, Any] = {
            "param_names": list(param_names),
            "posterior_mean": [float(m) for m in means],
            "posterior_std": [float(s) for s in stds],
            "acceptance_rate": float(results.acceptance_rate()),
            "burnin": int(results.burnin),
            "thin": int(results.thin),
            "n_iterations": int(results.n_iterations),
            "chains": chains.tolist(),
        }
        path.write_text(json.dumps(output_dict, indent=2))
    elif path.suffix == ".csv":
        results.to_dataframe().to_csv(path, index=False)
    else:
        msg = f"Unsupported output format: {path.suffix}. Use .json or .csv."
        raise ValueError(msg)


def _generate_estimate_plots(results: Any) -> None:
    """Generate trace plots for the posterior chains."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        param_names = results.param_names
        chains = np.asarray(results.chains)
        k = chains.shape[1] if chains.ndim > 1 else 1
        fig, axes = plt.subplots(k, 1, figsize=(8, 2.5 * k), squeeze=False)
        for j in range(k):
            col = chains[:, j] if chains.ndim > 1 else chains
            axes[j, 0].plot(col, lw=0.7)
            axes[j, 0].set_ylabel(param_names[j])
        axes[-1, 0].set_xlabel("iteration")
        fig.suptitle("PMMH trace plots")
        plt.tight_layout()
        plt.show()
    except ImportError:
        typer.echo("matplotlib not installed. Skipping plots.", err=True)
