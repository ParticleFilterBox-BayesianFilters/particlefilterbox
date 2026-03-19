"""Sequential Monte Carlo (SMC) advanced methods.

This module provides SMC methods for parameter inference, model evidence
computation, and sampling from complex target distributions.

Methods:
    - SMCSampler: General-purpose SMC sampler with adaptive tempering
    - Tempering: Specialized SMC for Bayesian inference via likelihood tempering
    - SMCSquared: SMC^2 for state-space models (joint state/parameter inference)
    - IBIS: Iterated Batch Importance Sampling
    - WasteFreeSMC: Waste-free SMC that recycles MCMC rejects

MCMC Kernels:
    - RandomWalkMH: Random walk Metropolis-Hastings
    - AdaptiveMH: Adaptive Metropolis-Hastings
    - IndependentMH: Independent Metropolis-Hastings
    - MALAKernel: Metropolis-Adjusted Langevin Algorithm

Results:
    - SMCResults: Container for SMC output with posterior summaries

References:
    Del Moral, P., Doucet, A. & Jasra, A. (2006). Sequential Monte Carlo
    samplers. JRSS-B, 68(3), 411-436.
    Chopin, N. (2002). A sequential particle filter method for static models.
    Biometrika, 89(3), 539-552.
    Chopin, N., Jacob, P.E. & Papaspiliopoulos, O. (2013). SMC^2.
    JRSS-B, 75(3), 397-426.
    Dau, H.D. & Chopin, N. (2022). Waste-free Sequential Monte Carlo.
    JRSS-B, 84(1), 114-148.
"""

from __future__ import annotations

from particlefilterbox.smc.base import BaseSMC
from particlefilterbox.smc.ibis import IBIS
from particlefilterbox.smc.mcmc_moves import (
    AdaptiveMH,
    IndependentMH,
    MALAKernel,
    MCMCStepResult,
    RandomWalkMH,
    random_walk_mh,
    run_mcmc_chain,
)
from particlefilterbox.smc.results import SMCResults
from particlefilterbox.smc.sampler import SMCSampler
from particlefilterbox.smc.smc_squared import SMCSquared
from particlefilterbox.smc.tempering import Tempering
from particlefilterbox.smc.waste_free import WasteFreeSMC

__all__ = [
    # Base
    "BaseSMC",
    # Samplers
    "SMCSampler",
    "Tempering",
    "SMCSquared",
    "IBIS",
    "WasteFreeSMC",
    # MCMC kernels
    "RandomWalkMH",
    "AdaptiveMH",
    "IndependentMH",
    "MALAKernel",
    "MCMCStepResult",
    "random_walk_mh",
    "run_mcmc_chain",
    # Results
    "SMCResults",
]
