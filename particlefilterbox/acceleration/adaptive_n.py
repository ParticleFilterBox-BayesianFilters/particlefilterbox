"""Adaptive particle count for particle filters.

Dynamically adjusts the number of particles based on ESS,
growing when the filter struggles and shrinking when it performs well.

Reference:
    Fox, D. (2003). Adapting the sample size in particle filters through
    KLD-sampling. International Journal of Robotics Research, 22(12), 985-1003.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class AdaptiveN:
    """Adaptive particle count based on ESS.

    Increases particle count when ESS is low (filter struggling) and
    decreases when ESS is high (filter healthy) to balance accuracy
    and computational cost.

    Parameters:
        n_min: Minimum number of particles.
        n_max: Maximum number of particles.
        growth_factor: Multiplicative factor for increasing N.
        shrink_factor: Multiplicative factor for decreasing N.
        ess_threshold_low: ESS/N ratio below which N is increased.
        ess_threshold_high: ESS/N ratio above which N is decreased.

    Examples:
        >>> adaptive = AdaptiveN(n_min=100, n_max=10000)
        >>> new_n = adaptive.adapt(current_n=500, ess=50, n_particles=500)
        >>> print(f"New N: {new_n}")
    """

    def __init__(
        self,
        n_min: int = 100,
        n_max: int = 10000,
        growth_factor: float = 2.0,
        shrink_factor: float = 0.5,
        ess_threshold_low: float = 0.2,
        ess_threshold_high: float = 0.8,
    ) -> None:
        if n_min < 1:
            raise ValueError(f"n_min must be >= 1, got {n_min}")
        if n_max < n_min:
            raise ValueError(f"n_max ({n_max}) must be >= n_min ({n_min})")
        if growth_factor <= 1.0:
            raise ValueError(f"growth_factor must be > 1, got {growth_factor}")
        if shrink_factor <= 0.0 or shrink_factor >= 1.0:
            raise ValueError(f"shrink_factor must be in (0, 1), got {shrink_factor}")

        self.n_min = n_min
        self.n_max = n_max
        self.growth_factor = growth_factor
        self.shrink_factor = shrink_factor
        self.ess_threshold_low = ess_threshold_low
        self.ess_threshold_high = ess_threshold_high

        self._n_history: list[int] = []
        self._ess_ratio_history: list[float] = []

    def adapt(
        self,
        current_n: int,
        ess: float,
        n_particles: int | None = None,
    ) -> int:
        """Compute new particle count based on ESS.

        Parameters:
            current_n: Current number of particles.
            ess: Current ESS value.
            n_particles: Total particles (for ratio). Defaults to current_n.

        Returns:
            New number of particles (clamped to [n_min, n_max]).
        """
        if n_particles is None:
            n_particles = current_n

        ess_ratio = ess / n_particles if n_particles > 0 else 0.0
        self._ess_ratio_history.append(ess_ratio)

        if ess_ratio < self.ess_threshold_low:
            # ESS too low -> grow
            new_n = int(current_n * self.growth_factor)
        elif ess_ratio > self.ess_threshold_high:
            # ESS high -> shrink
            new_n = int(current_n * self.shrink_factor)
        else:
            new_n = current_n

        # Clamp to bounds
        new_n = max(self.n_min, min(self.n_max, new_n))
        self._n_history.append(new_n)

        return new_n

    def add_particles(
        self,
        particles: NDArray[np.float64],
        weights: NDArray[np.float64],
        n_new: int,
        jitter_scale: float = 0.01,
        rng: np.random.Generator | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Add new particles by jittering existing high-weight particles.

        Resamples from existing particles weighted by their weights,
        then adds Gaussian jitter to create new diverse particles.

        Parameters:
            particles: Current particles, shape (N, D) or (N,).
            weights: Current normalized weights, shape (N,).
            n_new: Number of particles to add.
            jitter_scale: Standard deviation of Gaussian jitter.
            rng: Random number generator.

        Returns:
            Tuple of (new_particles, new_weights) with N + n_new particles.
        """
        if rng is None:
            rng = np.random.default_rng()

        n_current = len(weights)
        is_1d = particles.ndim == 1

        # Sample parent indices from current particles
        parent_indices = rng.choice(n_current, size=n_new, p=weights)

        if is_1d:
            new_particles = particles[parent_indices] + rng.normal(0, jitter_scale, size=n_new)
            combined_particles = np.concatenate([particles, new_particles])
        else:
            d = particles.shape[1]
            new_particles = particles[parent_indices] + rng.normal(0, jitter_scale, size=(n_new, d))
            combined_particles = np.vstack([particles, new_particles])

        # New particles get uniform weight, then re-normalize
        combined_weights = np.concatenate(
            [
                weights * (n_current / (n_current + n_new)),
                np.ones(n_new) / (n_current + n_new),
            ]
        )
        combined_weights = combined_weights / combined_weights.sum()

        return combined_particles, combined_weights

    def prune_particles(
        self,
        particles: NDArray[np.float64],
        weights: NDArray[np.float64],
        n_keep: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Remove particles with lowest weights.

        Keeps the top n_keep particles by weight and re-normalizes.

        Parameters:
            particles: Current particles, shape (N, D) or (N,).
            weights: Current normalized weights, shape (N,).
            n_keep: Number of particles to keep.

        Returns:
            Tuple of (pruned_particles, pruned_weights).
        """
        n_current = len(weights)
        if n_keep >= n_current:
            return particles.copy(), weights.copy()

        # Keep particles with highest weights
        top_indices = np.argsort(weights)[-n_keep:]
        top_indices = np.sort(top_indices)  # maintain order

        pruned_particles = particles[top_indices]
        pruned_weights = weights[top_indices]
        pruned_weights = pruned_weights / pruned_weights.sum()

        return pruned_particles, pruned_weights

    @property
    def n_history(self) -> list[int]:
        """History of particle counts."""
        return list(self._n_history)

    @property
    def ess_ratio_history(self) -> list[float]:
        """History of ESS/N ratios."""
        return list(self._ess_ratio_history)

    def reset(self) -> None:
        """Reset history."""
        self._n_history.clear()
        self._ess_ratio_history.clear()
