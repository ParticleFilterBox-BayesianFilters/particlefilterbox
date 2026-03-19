"""Degeneracy detection via ancestral tree analysis.

Detects particle filter degeneracy by analyzing the genealogy of particles
(unique ancestors and coalescence times).

Reference:
    Del Moral, P. (2004). Feynman-Kac Formulae: Genealogical and Interacting
    Particle Systems with Applications. Springer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class DegeneracyDetector:
    """Detect particle degeneracy via ancestral tree analysis.

    Analyzes the genealogical tree of particles to detect path degeneracy.
    When all particles at time t share a common ancestor at time t-lag (for
    small lag), the filter is degenerate.

    Requires that the particle filter was run with `store_ancestors=True`
    so that ancestor indices are available.

    Examples:
        >>> dd = DegeneracyDetector()
        >>> dd.load_ancestors(ancestor_indices)  # T x N array
        >>> ct = dd.coalescence_time(t=50)
        >>> print(f"Coalescence time: {ct}")
    """

    def __init__(self) -> None:
        self._ancestors: NDArray[np.int64] | None = None
        self._n_particles: int = 0
        self._n_time_steps: int = 0

    def load_ancestors(self, ancestors: NDArray[np.int64]) -> None:
        """Load ancestor indices from a completed filter run.

        Parameters:
            ancestors: Array of shape (T, N) where ancestors[t, i] is the
                index of particle i's parent at time t-1.

        Raises:
            ValueError: If ancestors has wrong shape.
        """
        arr = np.asarray(ancestors, dtype=np.int64)
        if arr.ndim != 2:
            raise ValueError(f"ancestors must be 2D, got {arr.ndim}D")
        self._ancestors = arr
        self._n_time_steps, self._n_particles = arr.shape

    def load_from_result(self, result: Any) -> None:
        """Load ancestors from a filter result object.

        Parameters:
            result: Filter result with `ancestors` attribute (T x N array).

        Raises:
            AttributeError: If result has no ancestors attribute.
            RuntimeError: If ancestors were not stored.
        """
        if not hasattr(result, "ancestors"):
            raise AttributeError(
                "Filter result has no 'ancestors' attribute. "
                "Run the filter with store_ancestors=True."
            )
        ancestors = result.ancestors
        if ancestors is None:
            raise RuntimeError("Ancestors are None. Run the filter with store_ancestors=True.")
        self.load_ancestors(ancestors)

    def _check_loaded(self) -> None:
        """Check that ancestors have been loaded."""
        if self._ancestors is None:
            raise RuntimeError(
                "No ancestors loaded. Call load_ancestors() or load_from_result() first."
            )

    def unique_ancestors(self, t: int, lag: int) -> int:
        """Count unique ancestors at time t-lag for particles alive at time t.

        Traces back the genealogy from time t to time t-lag and counts
        how many distinct ancestors exist.

        Parameters:
            t: Current time step.
            lag: Number of steps to trace back.

        Returns:
            Number of unique ancestors.

        Raises:
            ValueError: If t or lag is out of range.
        """
        self._check_loaded()
        assert self._ancestors is not None  # for type checker

        if t < 0 or t >= self._n_time_steps:
            raise ValueError(f"t={t} out of range [0, {self._n_time_steps})")
        if lag < 0:
            raise ValueError(f"lag must be >= 0, got {lag}")
        if lag == 0:
            return self._n_particles

        # Start with all particle indices at time t
        indices = np.arange(self._n_particles, dtype=np.int64)

        # Trace back
        current_t = t
        for _ in range(lag):
            if current_t <= 0:
                break
            indices = self._ancestors[current_t, indices]
            current_t -= 1

        return int(len(np.unique(indices)))

    def coalescence_time(self, t: int) -> int | None:
        """Find the coalescence time at time step t.

        The coalescence time is the smallest lag such that
        unique_ancestors(t, lag) == 1 (all particles share one ancestor).

        Parameters:
            t: Time step to analyze.

        Returns:
            Coalescence lag, or None if no coalescence within available history.
        """
        self._check_loaded()

        max_lag = min(t, self._n_time_steps - 1)
        for lag in range(1, max_lag + 1):
            if self.unique_ancestors(t, lag) == 1:
                return lag
        return None

    def mean_coalescence_time(self, start_t: int | None = None) -> float:
        """Compute mean coalescence time across time steps.

        Parameters:
            start_t: First time step to include (default: half of T).

        Returns:
            Mean coalescence time, or inf if some steps don't coalesce.
        """
        self._check_loaded()

        if start_t is None:
            start_t = self._n_time_steps // 2

        coal_times: list[float] = []
        for t in range(start_t, self._n_time_steps):
            ct = self.coalescence_time(t)
            if ct is not None:
                coal_times.append(float(ct))
            else:
                coal_times.append(float("inf"))

        return float(np.mean(coal_times)) if coal_times else float("inf")

    def is_degenerate(self, threshold: float = 0.1, lag: int = 10) -> bool:
        """Check if the filter is degenerate.

        Degenerate means that the fraction of unique ancestors at a given
        lag is below the threshold on average.

        Parameters:
            threshold: Fraction of N below which degeneracy is declared.
            lag: Number of steps to look back.

        Returns:
            True if the filter is degenerate.
        """
        self._check_loaded()

        start_t = max(lag, self._n_time_steps // 2)
        fractions: list[float] = []
        for t in range(start_t, self._n_time_steps):
            ua = self.unique_ancestors(t, lag)
            fractions.append(ua / self._n_particles)

        if not fractions:
            return False
        return float(np.mean(fractions)) < threshold

    def ancestral_tree_data(self) -> dict[str, NDArray[np.float64]]:
        """Get data for plotting the ancestral tree.

        Returns:
            Dictionary with 'unique_ancestors' (T x max_lag) array and
            'lags' array.
        """
        self._check_loaded()
        assert self._ancestors is not None

        max_lag = min(20, self._n_time_steps)
        lags = np.arange(1, max_lag + 1, dtype=np.int64)
        start_t = self._n_time_steps // 2

        n_steps = self._n_time_steps - start_t
        ua_matrix = np.zeros((n_steps, max_lag), dtype=np.float64)

        for i, t in enumerate(range(start_t, self._n_time_steps)):
            for j, lag in enumerate(lags):
                if int(lag) <= t:
                    ua_matrix[i, j] = self.unique_ancestors(t, int(lag)) / self._n_particles

        return {
            "unique_ancestor_fractions": ua_matrix,
            "lags": lags.astype(np.float64),
            "time_steps": np.arange(start_t, self._n_time_steps, dtype=np.float64),
        }

    def summary(self) -> dict[str, Any]:
        """Generate summary of degeneracy analysis.

        Returns:
            Dictionary with degeneracy statistics.
        """
        self._check_loaded()

        mean_ct = self.mean_coalescence_time()
        is_deg = self.is_degenerate()

        return {
            "n_particles": self._n_particles,
            "n_time_steps": self._n_time_steps,
            "mean_coalescence_time": mean_ct,
            "is_degenerate": is_deg,
        }
