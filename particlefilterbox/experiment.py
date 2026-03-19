"""PFExperiment: Reproducible experiment framework for particlefilterbox.

Provides a high-level interface for running, comparing, and reporting
particle filter experiments across multiple models, datasets, and metrics.

Examples
--------
>>> from particlefilterbox.experiment import PFExperiment
>>> exp = PFExperiment()
>>> exp.add_model('sv', SVModel())
>>> exp.add_dataset('sp500', load_dataset('sp500_returns'))
>>> exp.add_metric('loglike', lambda r: r.log_likelihood)
>>> results = exp.run()
>>> print(results.compare())
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from particlefilterbox._logging import get_logger

logger = get_logger("experiment")


@dataclass
class ExperimentResult:
    """Result of a single experiment run.

    Attributes
    ----------
    model_name : str
        Name of the model used.
    dataset_name : str
        Name of the dataset used.
    metrics : dict[str, float]
        Computed metric values.
    filter_results : Any
        Raw filter results object.
    elapsed_time : float
        Time in seconds for the run.
    """

    model_name: str
    dataset_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    filter_results: Any = None
    elapsed_time: float = 0.0


class PFExperiment:
    """Reproducible experiment framework.

    Manages models, datasets, and metrics for systematic comparison
    of particle filtering approaches.

    Parameters
    ----------
    n_particles : int
        Default number of particles. Default is 1000.
    seed : int
        Random seed for reproducibility. Default is 42.
    name : str
        Experiment name. Default is 'experiment'.

    Examples
    --------
    >>> exp = PFExperiment(n_particles=500, seed=42)
    >>> exp.add_model('sv', sv_model)
    >>> exp.add_dataset('sp500', sp500_df)
    >>> exp.add_metric('loglike', lambda r: r.log_likelihood)
    >>> comparison = exp.run()
    >>> print(comparison)
    """

    def __init__(
        self,
        n_particles: int = 1000,
        seed: int = 42,
        name: str = "experiment",
    ) -> None:
        self.n_particles = n_particles
        self.seed = seed
        self.name = name

        self._models: dict[str, Any] = {}
        self._datasets: dict[str, Any] = {}
        self._metrics: dict[str, Callable[..., float]] = {}
        self._results: list[ExperimentResult] = []

    def add_model(self, name: str, model: Any) -> None:
        """Add a model to the experiment.

        Parameters
        ----------
        name : str
            Display name for the model.
        model : Any
            Model instance (must have standard interface).
        """
        self._models[name] = model
        logger.info(f"Added model: {name}")

    def add_dataset(
        self,
        name: str,
        data: Any,
        obs_column: str | None = None,
    ) -> None:
        """Add a dataset to the experiment.

        Parameters
        ----------
        name : str
            Display name for the dataset.
        data : pd.DataFrame or NDArray
            Data to use.
        obs_column : str or None
            Column name for observations (if DataFrame).
        """
        if isinstance(data, pd.DataFrame) and obs_column is not None:
            self._datasets[name] = data[obs_column].values
        elif isinstance(data, pd.DataFrame):
            # Try common column names
            for col in ["returns", "observation", "y", "log_return"]:
                if col in data.columns:
                    self._datasets[name] = data[col].values
                    break
            else:
                # Use first numeric column
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    self._datasets[name] = data[numeric_cols[0]].values
                else:
                    self._datasets[name] = data.values
        else:
            self._datasets[name] = np.asarray(data)

        logger.info(f"Added dataset: {name}")

    def add_metric(
        self,
        name: str,
        metric_fn: Callable[..., float],
    ) -> None:
        """Add a metric to evaluate.

        Parameters
        ----------
        name : str
            Display name for the metric.
        metric_fn : callable
            Function that takes filter results and returns a float.
        """
        self._metrics[name] = metric_fn
        logger.info(f"Added metric: {name}")

    def run(self, verbose: bool = True) -> pd.DataFrame:
        """Run all experiments.

        Executes the particle filter for each (model, dataset) combination
        and computes all metrics.

        Parameters
        ----------
        verbose : bool
            Print progress. Default is True.

        Returns
        -------
        pd.DataFrame
            Comparison table with models as rows and metrics as columns.
        """
        import time

        self._results = []

        for model_name, model in self._models.items():
            for dataset_name, data in self._datasets.items():
                if verbose:
                    logger.info(f"Running {model_name} on {dataset_name}...")

                rng = np.random.default_rng(self.seed)
                t0 = time.time()

                try:
                    # Try to create and run filter
                    filter_results = self._run_filter(model, data, rng)
                    elapsed = time.time() - t0

                    # Compute metrics
                    metrics: dict[str, float] = {}
                    for metric_name, metric_fn in self._metrics.items():
                        try:
                            metrics[metric_name] = float(metric_fn(filter_results))
                        except Exception as e:
                            logger.warning(f"Metric '{metric_name}' failed: {e}")
                            metrics[metric_name] = float("nan")

                    result = ExperimentResult(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        metrics=metrics,
                        filter_results=filter_results,
                        elapsed_time=elapsed,
                    )
                    self._results.append(result)

                    if verbose:
                        logger.info(
                            f"  {model_name}/{dataset_name}: "
                            + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                            + f" ({elapsed:.2f}s)"
                        )

                except Exception as e:
                    logger.error(f"  {model_name}/{dataset_name} FAILED: {e}")
                    self._results.append(
                        ExperimentResult(
                            model_name=model_name,
                            dataset_name=dataset_name,
                            elapsed_time=time.time() - t0,
                        )
                    )

        return self.compare()

    def _run_filter(self, model: Any, data: Any, rng: Any) -> Any:
        """Run particle filter on model and data."""
        try:
            from particlefilterbox.filters.bootstrap import BootstrapFilter

            pf = BootstrapFilter(
                model=model,
                n_particles=self.n_particles,
                rng=rng,
            )
            return pf.filter(data)
        except ImportError as err:
            msg = "BootstrapFilter not available"
            raise ImportError(msg) from err

    def compare(self) -> pd.DataFrame:
        """Generate comparison DataFrame.

        Returns
        -------
        pd.DataFrame
            Table with columns: model, dataset, metric1, metric2, ..., time.
        """
        rows = []
        for result in self._results:
            row: dict[str, Any] = {
                "model": result.model_name,
                "dataset": result.dataset_name,
            }
            row.update(result.metrics)
            row["time_s"] = result.elapsed_time
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> None:
        """Save experiment results to JSON.

        Parameters
        ----------
        path : str or Path
            Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "name": self.name,
            "n_particles": self.n_particles,
            "seed": self.seed,
            "results": [
                {
                    "model": r.model_name,
                    "dataset": r.dataset_name,
                    "metrics": r.metrics,
                    "elapsed_time": r.elapsed_time,
                }
                for r in self._results
            ],
        }

        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Experiment saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> PFExperiment:
        """Load experiment results from JSON.

        Parameters
        ----------
        path : str or Path
            Input file path.

        Returns
        -------
        PFExperiment
            Experiment with loaded results (models/datasets not restored).
        """
        path = Path(path)
        data = json.loads(path.read_text())

        exp = cls(
            n_particles=data.get("n_particles", 1000),
            seed=data.get("seed", 42),
            name=data.get("name", "loaded"),
        )

        for r in data.get("results", []):
            exp._results.append(
                ExperimentResult(
                    model_name=r["model"],
                    dataset_name=r["dataset"],
                    metrics=r.get("metrics", {}),
                    elapsed_time=r.get("elapsed_time", 0.0),
                )
            )

        return exp

    def report(self) -> Any:
        """Generate a report from experiment results.

        Returns
        -------
        BaseReport
            Report object with experiment summary.
        """
        from particlefilterbox.reports.base import BaseReport

        report = BaseReport(title=f"Experiment: {self.name}")

        # Summary section
        report.add_section(
            title="Experiment Summary",
            content=(
                f"Experiment '{self.name}' with {len(self._models)} models, "
                f"{len(self._datasets)} datasets, {len(self._metrics)} metrics. "
                f"N particles: {self.n_particles}, seed: {self.seed}."
            ),
        )

        # Comparison table
        df = self.compare()
        if not df.empty:
            report.add_section(
                title="Results Comparison",
                tables=[
                    {
                        "headers": list(df.columns),
                        "rows": df.values.tolist(),
                        "caption": "Model Comparison",
                    }
                ],
            )

        return report
