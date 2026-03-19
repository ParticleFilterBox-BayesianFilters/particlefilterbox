"""Regularized Particle Filter (Musso et al, 2001).

Applies kernel jittering after resampling to prevent sample impoverishment.
Instead of using resampled particles directly, each particle is perturbed
by a kernel to maintain diversity.

References
----------
Musso, C., Oudjane, N. & Le Gland, F. (2001). Improving regularised
particle filters. Sequential Monte Carlo Methods in Practice, 247-271.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from particlefilterbox._logging import get_logger
from particlefilterbox.filters.base import BaseParticleFilter

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from particlefilterbox.core.cloud import ParticleCloud
    from particlefilterbox.core.config import PFConfig
    from particlefilterbox.core.model import ParticleFilterModel

logger = get_logger(__name__)


class RegularizedPF(BaseParticleFilter):
    """Regularized Particle Filter with kernel jittering.

    After resampling, each particle is perturbed by a kernel to avoid
    sample impoverishment. Uses Silverman's rule for bandwidth selection
    or a user-specified bandwidth.

    Parameters
    ----------
    model : ParticleFilterModel
        State-space model.
    config : PFConfig
        Particle filter configuration.
    bandwidth : str | float
        Bandwidth selection method. Either ``'silverman'`` for automatic
        Silverman bandwidth, or a float for fixed bandwidth.
    kernel : str
        Kernel type: ``'gaussian'`` or ``'epanechnikov'``.

    Examples
    --------
    >>> rpf = RegularizedPF(model=model, config=config, bandwidth='silverman', kernel='gaussian')
    >>> result = rpf.filter(observations)
    """

    SUPPORTED_KERNELS = ("gaussian", "epanechnikov")

    def __init__(
        self,
        model: ParticleFilterModel,
        config: PFConfig,
        bandwidth: str | float = "silverman",
        kernel: Literal["gaussian", "epanechnikov"] = "gaussian",
    ) -> None:
        super().__init__(model=model, config=config)

        if kernel not in self.SUPPORTED_KERNELS:
            raise ValueError(f"Unsupported kernel '{kernel}'. Choose from {self.SUPPORTED_KERNELS}")

        self._bandwidth_param = bandwidth
        self._kernel = kernel

        logger.info(
            "RegularizedPF initialized (bandwidth=%s, kernel=%s)",
            bandwidth,
            kernel,
        )

    def _silverman_bandwidth(
        self,
        n_particles: int,
        k_states: int,
    ) -> float:
        """Compute Silverman's rule-of-thumb bandwidth.

        h = (4 / (N * (k + 2)))^(1 / (k + 4))

        Parameters
        ----------
        n_particles : int
            Number of particles.
        k_states : int
            State dimension.

        Returns
        -------
        h : float
            Bandwidth.
        """
        return float((4.0 / (n_particles * (k_states + 2))) ** (1.0 / (k_states + 4)))

    def _compute_bandwidth(
        self,
        particles: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> float:
        """Compute the bandwidth for kernel jittering.

        Parameters
        ----------
        particles : ndarray of shape (N, k_states)
            Current particles.
        weights : ndarray of shape (N,)
            Normalized weights.

        Returns
        -------
        h : float
            Bandwidth.
        """
        n_particles = particles.shape[0]
        k_states = particles.shape[1]

        if isinstance(self._bandwidth_param, str) and self._bandwidth_param == "silverman":
            h_base = self._silverman_bandwidth(n_particles, k_states)
            weighted_mean = np.average(particles, weights=weights, axis=0)
            diff = particles - weighted_mean
            weighted_var = np.average(diff**2, weights=weights, axis=0)
            weighted_std = np.sqrt(np.mean(weighted_var))
            return float(h_base * weighted_std)

        return float(self._bandwidth_param)

    def _jitter(
        self,
        particles: NDArray[np.float64],
        bandwidth: float,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Apply kernel jittering to particles.

        Parameters
        ----------
        particles : ndarray of shape (N, k_states)
            Resampled particles.
        bandwidth : float
            Bandwidth for the kernel.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        jittered : ndarray of shape (N, k_states)
            Jittered particles.
        """
        n_particles = particles.shape[0]
        k_states = particles.shape[1]

        if self._kernel == "gaussian":
            noise = rng.normal(0, bandwidth, size=(n_particles, k_states))
            return particles + noise

        if self._kernel == "epanechnikov":
            # Epanechnikov kernel: uniform on the unit ball, scaled by bandwidth
            noise = rng.normal(0, 1, size=(n_particles, k_states))
            u = rng.uniform(0, 1, size=n_particles)
            radius = u ** (1.0 / k_states)
            norms = np.linalg.norm(noise, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            directions = noise / norms
            perturbation = bandwidth * radius[:, np.newaxis] * directions
            # Epanechnikov scaling factor: sqrt(k+2) for unit ball
            perturbation *= np.sqrt(k_states + 2)
            return particles + perturbation

        raise ValueError(f"Unknown kernel: {self._kernel}")

    def _propagate(
        self,
        cloud: ParticleCloud,
        y_t: NDArray[np.float64],
        t: int,
        rng: np.random.Generator,
    ) -> ParticleCloud:
        """Propagate particles using the transition prior (same as bootstrap)."""
        from particlefilterbox.core.cloud import ParticleCloud as _Cloud

        new_particles = self.model.transition(cloud.particles, t, rng)

        new_cloud = _Cloud(cloud.n_particles, cloud.k_states)
        new_cloud.particles = new_particles
        new_cloud.set_log_weights(cloud.log_weights.copy())
        return new_cloud

    def _compute_weights(
        self,
        cloud: ParticleCloud,
        y_t: NDArray[np.float64],
        t: int,
    ) -> NDArray[np.float64]:
        """Compute incremental log-weights using observation likelihood."""
        return self.model.log_observation_likelihood(cloud.particles, y_t, t)

    def _maybe_resample(
        self,
        cloud: ParticleCloud,
        rng: np.random.Generator,
    ) -> tuple[ParticleCloud, bool]:
        """Resample with kernel jittering to maintain particle diversity.

        Overrides base class to add kernel jittering after resampling.
        """
        from particlefilterbox.resampling import systematic_resample

        ess = self._compute_ess(cloud)
        threshold = self.config.ess_threshold * self.n_particles

        if ess < threshold:
            weights = cloud.normalized_weights

            # Compute bandwidth before resampling (using weighted particles)
            bandwidth = self._compute_bandwidth(cloud.particles, weights)

            # Resample
            indices = systematic_resample(weights, rng=rng)
            cloud.resample(indices)

            # Apply kernel jittering
            cloud.particles = self._jitter(cloud.particles, bandwidth, rng)

            logger.debug("Resampled + jittered (h=%.4f, ESS=%.1f)", bandwidth, ess)
            return cloud, True

        return cloud, False
