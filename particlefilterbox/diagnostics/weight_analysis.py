"""Weight distribution analysis for particle filters.

Provides entropy, coefficient of variation, maximum weight, and Gini index
analysis for diagnosing weight degeneracy in particle filters.

Reference:
    Carpenter, J., Clifford, P. & Fearnhead, P. (1999). Improved particle
    filter for nonlinear problems. IEE Proceedings-Radar, Sonar and
    Navigation, 146(1), 2-7.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class WeightAnalysis:
    """Detailed analysis of particle weight distributions.

    Tracks entropy, coefficient of variation, maximum weight, and Gini
    index over time to diagnose weight degeneracy.

    Examples:
        >>> wa = WeightAnalysis()
        >>> weights = np.ones(100) / 100  # uniform
        >>> wa.update_from_weights(weights)
        >>> info = wa.summary()
        >>> assert abs(info["entropy_current"] - np.log(100)) < 1e-10
    """

    def __init__(self) -> None:
        self._entropy_history: list[float] = []
        self._max_weight_history: list[float] = []
        self._cv_history: list[float] = []
        self._gini_history: list[float] = []
        self._n_particles_history: list[int] = []

    @staticmethod
    def entropy(weights: NDArray[np.float64]) -> float:
        """Compute Shannon entropy of weight distribution.

        H = -sum(w_i * log(w_i))

        For uniform weights: H = log(N).
        For degenerate weights (single particle): H = 0.

        Parameters:
            weights: Normalized weights (must sum to 1).

        Returns:
            Shannon entropy.
        """
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()  # ensure normalization
        # Avoid log(0) by masking zero weights
        mask = w > 0
        return float(-np.sum(w[mask] * np.log(w[mask])))

    @staticmethod
    def gini_index(weights: NDArray[np.float64]) -> float:
        """Compute Gini index of weight distribution.

        Gini = 0 for perfectly uniform weights.
        Gini ~ 1 for completely degenerate weights.

        Parameters:
            weights: Normalized weights (must sum to 1).

        Returns:
            Gini index in [0, 1).
        """
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        n = len(w)
        w_sorted = np.sort(w)
        indices = np.arange(1, n + 1, dtype=np.float64)
        gini = (2.0 * np.sum(indices * w_sorted) / (n * np.sum(w_sorted))) - (n + 1.0) / n
        return float(gini)

    @staticmethod
    def coefficient_of_variation(weights: NDArray[np.float64]) -> float:
        """Compute coefficient of variation of weights.

        CV = std(w) / mean(w).
        For uniform weights: CV = 0.

        Parameters:
            weights: Normalized weights.

        Returns:
            Coefficient of variation.
        """
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        mean_w: float = w.mean().item()
        if mean_w == 0:
            return float("inf")
        std_w: float = w.std().item()
        return std_w / mean_w

    def update_from_weights(
        self,
        weights: NDArray[np.float64],
    ) -> None:
        """Update analysis with new weights.

        Parameters:
            weights: Normalized particle weights.
        """
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()

        self._entropy_history.append(self.entropy(w))
        self._max_weight_history.append(float(np.max(w)))
        self._cv_history.append(self.coefficient_of_variation(w))
        self._gini_history.append(self.gini_index(w))
        self._n_particles_history.append(len(w))

    def update(self, cloud: Any) -> None:
        """Update analysis from a ParticleCloud object.

        Parameters:
            cloud: Object with a `weights` attribute (normalized).
        """
        weights = np.asarray(cloud.weights, dtype=np.float64)
        self.update_from_weights(weights)

    def entropy_history(self) -> NDArray[np.float64]:
        """Return entropy history as array."""
        return np.array(self._entropy_history, dtype=np.float64)

    def gini_history(self) -> NDArray[np.float64]:
        """Return Gini index history as array."""
        return np.array(self._gini_history, dtype=np.float64)

    def summary(self) -> dict[str, Any]:
        """Generate summary of weight analysis.

        Returns:
            Dictionary with current and historical weight statistics.
        """
        result: dict[str, Any] = {
            "n_time_steps": len(self._entropy_history),
        }

        if self._entropy_history:
            result["entropy_current"] = self._entropy_history[-1]
            result["entropy_mean"] = float(np.mean(self._entropy_history))
            result["entropy_min"] = float(np.min(self._entropy_history))

            result["max_weight_current"] = self._max_weight_history[-1]
            result["max_weight_mean"] = float(np.mean(self._max_weight_history))
            result["max_weight_max"] = float(np.max(self._max_weight_history))

            result["cv_current"] = self._cv_history[-1]
            result["cv_mean"] = float(np.mean(self._cv_history))

            result["gini_current"] = self._gini_history[-1]
            result["gini_mean"] = float(np.mean(self._gini_history))
            result["gini_max"] = float(np.max(self._gini_history))

            n = self._n_particles_history[-1]
            result["max_entropy"] = float(np.log(n))
            result["entropy_ratio"] = self._entropy_history[-1] / float(np.log(n))

        return result

    def reset(self) -> None:
        """Reset all analysis state."""
        self._entropy_history.clear()
        self._max_weight_history.clear()
        self._cv_history.clear()
        self._gini_history.clear()
        self._n_particles_history.clear()
