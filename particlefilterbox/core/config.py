"""Particle filter configuration."""

from __future__ import annotations

from dataclasses import dataclass

_VALID_RESAMPLING = {
    "multinomial",
    "systematic",
    "stratified",
    "residual",
    "optimal_transport",
    "adaptive",
    "killing",
}


@dataclass
class PFConfig:
    """Configuration for a particle filter run.

    Parameters
    ----------
    n_particles : int
        Number of particles (default: 1000).
    resampling : str
        Resampling method name (default: 'systematic').
    ess_threshold : float
        Resample when ESS < threshold * N (default: 0.5).
    seed : int or None
        Random seed for reproducibility (default: None).
    store_particles : bool
        Store full particle history - uses more memory (default: False).
    store_ancestors : bool
        Store ancestor indices history (default: False).
    store_weights : bool
        Store weight history (default: False).
    log_level : str
        Logging level (default: 'WARNING').
    """

    n_particles: int = 1000
    resampling: str = "systematic"
    ess_threshold: float = 0.5
    seed: int | None = None
    store_particles: bool = False
    store_ancestors: bool = False
    store_weights: bool = False
    log_level: str = "WARNING"

    def validate(self) -> None:
        """Validate configuration consistency.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        """
        if self.n_particles <= 0:
            msg = f"n_particles must be positive, got {self.n_particles}"
            raise ValueError(msg)
        if not (0.0 < self.ess_threshold <= 1.0):
            msg = f"ess_threshold must be in (0, 1], got {self.ess_threshold}"
            raise ValueError(msg)
        if self.resampling not in _VALID_RESAMPLING:
            msg = f"Unknown resampling '{self.resampling}'. Valid: {sorted(_VALID_RESAMPLING)}"
            raise ValueError(msg)

    def effective_threshold(self) -> float:
        """Return the absolute ESS threshold: ess_threshold * n_particles."""
        return self.ess_threshold * self.n_particles
