"""Particle smoothers for state estimation using all data.

Smoothers estimate p(x_t | y_{1:T}) using both past and future observations,
providing improved estimates compared to filtering p(x_t | y_{1:t}).

Key property: Var[x_t | y_{1:T}] <= Var[x_t | y_{1:t}]

Available smoothers:
- BaseParticleSmoother: Abstract base class for all smoothers
- FFBSm: Forward Filtering Backward Smoothing (exact, O(T*N^2))
- FFBSi: Forward Filtering Backward Simulation (simulation, O(T*N*M))
- TwoFilterSmoother: Two-filter smoothing (forward + backward)
- FixedLagSmoother: Fixed-lag online smoothing with ancestor tracing

References
----------
- Godsill, S.J., Doucet, A. & West, M. (2004). Monte Carlo smoothing for
  nonlinear time series. JASA, 99(465), 156-168.
- Briers, M., Doucet, A. & Maskell, S. (2010). Smoothing algorithms for
  state-space models. Annals of the Institute of Statistical Mathematics.
- Kitagawa, G. (1996). Monte Carlo filter and smoother for non-Gaussian
  nonlinear state space models.
- Doucet, A. & Johansen, A.M. (2009). A tutorial on particle filtering and
  smoothing: Fifteen years later.
- Lindsten, F. & Schon, T.B. (2013). Backward simulation methods for Monte
  Carlo statistical inference.
"""

from particlefilterbox.smoothers.base import BaseParticleSmoother
from particlefilterbox.smoothers.ffbsi import FFBSi
from particlefilterbox.smoothers.ffbsm import FFBSm
from particlefilterbox.smoothers.fixed_lag import FixedLagSmoother
from particlefilterbox.smoothers.two_filter import TwoFilterSmoother

__all__ = [
    "BaseParticleSmoother",
    "FFBSm",
    "FFBSi",
    "TwoFilterSmoother",
    "FixedLagSmoother",
]
