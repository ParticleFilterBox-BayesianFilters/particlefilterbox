"""Particle filters for state estimation.

This module provides a comprehensive collection of particle filters,
from basic Bootstrap/SIR to advanced methods including Rao-Blackwellized
and Unscented Particle Filters.

Basic Filters (Fase 2)
----------------------
- BootstrapPF: Standard Bootstrap/SIR particle filter
- SIR: Sequential Importance Resampling with general proposal support

Advanced Filters (Fase 3)
-------------------------
- AuxiliaryPF: Auxiliary Particle Filter (Pitt & Shephard, 1999)
- LocallyOptimalPF: Locally Optimal PF with analytic proposal
- RegularizedPF: Regularized PF with kernel jittering (Musso et al, 2001)
- RaoBlackwellizedPF: Rao-Blackwellized PF for mixed models (uses kalmanbox)
- UnscentedPF: Unscented PF with UKF proposal (uses kalmanbox)
- EnsemblePF: Ensemble PF with Kalman-like updates
- GuidedPF: Guided PF with observation-informed proposals
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from particlefilterbox.filters.auxiliary import AuxiliaryPF
from particlefilterbox.filters.base import BaseParticleFilter
from particlefilterbox.filters.bootstrap import BootstrapPF
from particlefilterbox.filters.ensemble import EnsemblePF
from particlefilterbox.filters.guided import GuidedPF
from particlefilterbox.filters.locally_optimal import LocallyOptimalPF
from particlefilterbox.filters.regularized import RegularizedPF
from particlefilterbox.filters.sir import SIR

if TYPE_CHECKING:
    from particlefilterbox.filters.rao_blackwellized import RaoBlackwellizedPF
    from particlefilterbox.filters.unscented import UnscentedPF


def __getattr__(name: str) -> type:
    if name == "RaoBlackwellizedPF":
        from particlefilterbox.filters.rao_blackwellized import RaoBlackwellizedPF

        return RaoBlackwellizedPF
    if name == "UnscentedPF":
        from particlefilterbox.filters.unscented import UnscentedPF

        return UnscentedPF
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)

__all__ = [
    # Base
    "BaseParticleFilter",
    # Fase 2
    "BootstrapPF",
    "SIR",
    # Fase 3
    "AuxiliaryPF",
    "LocallyOptimalPF",
    "RegularizedPF",
    "RaoBlackwellizedPF",
    "UnscentedPF",
    "EnsemblePF",
    "GuidedPF",
]
