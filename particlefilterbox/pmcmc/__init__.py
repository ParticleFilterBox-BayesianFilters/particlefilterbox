"""Particle Markov Chain Monte Carlo (PMCMC) methods.

This module provides PMCMC methods for Bayesian inference in state-space models,
combining MCMC with particle filters for likelihood estimation.

Methods:
    - PMMH: Particle Marginal Metropolis-Hastings
    - ParticleGibbs: Particle Gibbs sampler with Conditional SMC
    - PGAS: Particle Gibbs with Ancestor Sampling
    - SMC2Online: Online/streaming SMC^2
    - ConditionalSMC: SMC conditioned on reference trajectory

Proposals:
    - GaussianRandomWalk: Fixed covariance Gaussian random walk
    - AdaptiveGaussian: Adaptive Gaussian with Roberts-Rosenthal scaling
    - LogNormalProposal: Log-normal proposal for positive parameters
    - TransformedProposal: Proposal in transformed space

Results:
    - PMCMCResults: Container with chains, diagnostics, and posterior summaries

References:
    Andrieu, C., Doucet, A. & Holenstein, R. (2010). Particle Markov chain
    Monte Carlo methods. JRSS-B, 72(3), 269-342.
    Lindsten, F., Jordan, M. I. & Schon, T. B. (2014). Particle Gibbs
    with ancestor sampling. JMLR, 15, 2145-2184.
    Andrieu, C. & Roberts, G. O. (2009). The pseudo-marginal approach for
    efficient Monte Carlo computations. Annals of Statistics, 37(2), 697-725.
    Chopin, N., Jacob, P. E. & Papaspiliopoulos, O. (2013). SMC^2: An
    efficient algorithm for sequential analysis of state space models.
    JRSS-B, 75(3), 397-426.
"""

from __future__ import annotations

from particlefilterbox.pmcmc.base import BasePMCMC
from particlefilterbox.pmcmc.conditional_smc import ConditionalSMC, CSMCResult
from particlefilterbox.pmcmc.particle_gibbs import ParticleGibbs
from particlefilterbox.pmcmc.pgas import PGAS
from particlefilterbox.pmcmc.pmmh import PMMH
from particlefilterbox.pmcmc.proposals import (
    AdaptiveGaussian,
    BaseProposal,
    GaussianRandomWalk,
    LogNormalProposal,
    TransformedProposal,
)
from particlefilterbox.pmcmc.results import PMCMCResults
from particlefilterbox.pmcmc.smc2_online import SMC2Online, SMC2OnlineState

__all__ = [
    # Base
    "BasePMCMC",
    # Methods
    "PMMH",
    "ParticleGibbs",
    "PGAS",
    "ConditionalSMC",
    "CSMCResult",
    "SMC2Online",
    "SMC2OnlineState",
    # Proposals
    "BaseProposal",
    "GaussianRandomWalk",
    "AdaptiveGaussian",
    "LogNormalProposal",
    "TransformedProposal",
    # Results
    "PMCMCResults",
]
