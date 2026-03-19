"""
particlefilterbox.models - Modelos pre-construidos para particle filtering.

Modelos disponiveis:
- StochasticVolatility: 4 variantes (basic, leverage, jumps, factor)
- JumpDiffusion: 3 variantes (merton, kou, bates)
- DSGE: modelos de equilibrio geral
- CountStateSpace: modelos de contagem (poisson, binomial, sir)
- NonlinearRegime: modelos com mudanca de regime
- BoundedStates: estados com restricoes
- Mixture: modelos de mistura
- ContinuousTime: modelos em tempo continuo (cir, vasicek, heston)
"""

from particlefilterbox.models.bounded_states import BoundedStates
from particlefilterbox.models.continuous_time import ContinuousTime
from particlefilterbox.models.count_state_space import CountStateSpace
from particlefilterbox.models.dsge import DSGE
from particlefilterbox.models.jump_diffusion import JumpDiffusion
from particlefilterbox.models.mixture import Mixture
from particlefilterbox.models.nonlinear_regime import NonlinearRegime
from particlefilterbox.models.stochastic_volatility import StochasticVolatility

__all__ = [
    "BoundedStates",
    "ContinuousTime",
    "CountStateSpace",
    "DSGE",
    "JumpDiffusion",
    "Mixture",
    "NonlinearRegime",
    "StochasticVolatility",
]
