"""Model comparison via Bayes factors.

Compares multiple models using log-evidence estimated by particle filters.
Interprets Bayes factors using the Kass & Raftery (1995) scale.

Reference:
    Kass, R.E. & Raftery, A.E. (1995). Bayes Factors. Journal of the
    American Statistical Association, 90(430), 773-795.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class ModelResult:
    """Result for a single model.

    Attributes:
        name: Model name.
        log_evidence: Estimated log marginal likelihood.
        n_particles: Number of particles used.
    """

    name: str
    log_evidence: float
    n_particles: int


class ModelComparison:
    """Compare multiple models using Bayes factors.

    Runs particle filters for each registered model to estimate
    log marginal likelihoods, then computes Bayes factors and rankings.

    Parameters:
        n_particles: Number of particles for each filter (default 1000).
        n_repeats: Number of repeated runs for averaging (default 1).

    Examples:
        >>> mc = ModelComparison(n_particles=500)
        >>> mc.add_model("model_A", model_a, filter_factory_a)
        >>> mc.add_model("model_B", model_b, filter_factory_b)
        >>> mc.run(endog=observations)
        >>> print(mc.ranking())
        >>> print(mc.bayes_factor("model_A", "model_B"))
    """

    def __init__(
        self,
        n_particles: int = 1000,
        n_repeats: int = 1,
    ) -> None:
        self.n_particles = n_particles
        self.n_repeats = n_repeats

        self._models: dict[str, dict[str, Any]] = {}
        self._results: dict[str, ModelResult] = {}

    def add_model(
        self,
        name: str,
        model: Any,
        filter_factory: Any | None = None,
    ) -> None:
        """Register a model for comparison.

        Parameters:
            name: Unique name for the model.
            model: Model object (with simulate/filter capabilities).
            filter_factory: Factory to create particle filter for this model.
                If None, assumes model has a built-in filter method.
        """
        self._models[name] = {
            "model": model,
            "filter_factory": filter_factory,
        }

    def run(
        self,
        endog: NDArray[np.float64],
        seed: int = 42,
    ) -> dict[str, ModelResult]:
        """Run particle filters for all models and compute log-evidence.

        Parameters:
            endog: Observed data.
            seed: Random seed.

        Returns:
            Dictionary mapping model name to ModelResult.
        """
        rng = np.random.default_rng(seed)
        self._results.clear()

        for name, info in self._models.items():
            model = info["model"]
            factory = info["filter_factory"]

            log_evidences: list[float] = []
            for _ in range(self.n_repeats):
                rng.integers(0, 2**31)  # advance RNG state

                if factory is not None:
                    pf = factory.create(model, self.n_particles)
                else:
                    pf = model.create_filter(n_particles=self.n_particles)

                result = pf.filter(endog)
                log_lik = result.log_likelihood
                log_evidences.append(float(log_lik))

            mean_log_evidence = float(np.mean(log_evidences))
            self._results[name] = ModelResult(
                name=name,
                log_evidence=mean_log_evidence,
                n_particles=self.n_particles,
            )

        return dict(self._results)

    def log_evidence(self, name: str | None = None) -> float | dict[str, float]:
        """Get log evidence for a model or all models.

        Parameters:
            name: Model name. If None, returns all.

        Returns:
            Log evidence value or dictionary of all values.
        """
        if not self._results:
            raise RuntimeError("Must call run() first.")
        if name is not None:
            if name not in self._results:
                raise KeyError(f"Model '{name}' not found.")
            return self._results[name].log_evidence
        return {n: r.log_evidence for n, r in self._results.items()}

    def bayes_factor(self, model1: str, model2: str) -> dict[str, Any]:
        """Compute log Bayes factor between two models.

        Parameters:
            model1: Name of first model.
            model2: Name of second model.

        Returns:
            Dictionary with log_bf, interpretation, and favored model.
        """
        if not self._results:
            raise RuntimeError("Must call run() first.")

        le1 = self._results[model1].log_evidence
        le2 = self._results[model2].log_evidence
        log_bf = le1 - le2

        abs_log_bf = abs(log_bf)
        if abs_log_bf < 1:
            interpretation = "No evidence"
        elif abs_log_bf < 3:
            interpretation = "Positive evidence"
        elif abs_log_bf < 5:
            interpretation = "Strong evidence"
        else:
            interpretation = "Very strong evidence"

        favored = model1 if log_bf > 0 else model2

        return {
            "log_bayes_factor": log_bf,
            "interpretation": interpretation,
            "favored_model": favored,
            "model1_log_evidence": le1,
            "model2_log_evidence": le2,
        }

    def ranking(self) -> list[tuple[str, float]]:
        """Rank models by log evidence (highest first).

        Returns:
            List of (model_name, log_evidence) tuples sorted descending.
        """
        if not self._results:
            raise RuntimeError("Must call run() first.")
        ranked = sorted(
            self._results.items(),
            key=lambda x: x[1].log_evidence,
            reverse=True,
        )
        return [(name, r.log_evidence) for name, r in ranked]

    def summary(self) -> dict[str, Any]:
        """Generate comparison summary.

        Returns:
            Dictionary with rankings and pairwise Bayes factors.
        """
        if not self._results:
            raise RuntimeError("Must call run() first.")

        names = list(self._results.keys())
        pairwise: dict[str, dict[str, Any]] = {}
        for i, n1 in enumerate(names):
            for n2 in names[i + 1 :]:
                key = f"{n1}_vs_{n2}"
                pairwise[key] = self.bayes_factor(n1, n2)

        return {
            "ranking": self.ranking(),
            "pairwise_bayes_factors": pairwise,
            "n_models": len(self._results),
            "n_particles": self.n_particles,
        }
