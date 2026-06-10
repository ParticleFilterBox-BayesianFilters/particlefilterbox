"""Shared model/filter helpers for the particlefilterbox CLI.

The CLI exposes a small, curated set of models and filters. The underlying
library models (e.g. :class:`StochasticVolatility`) do not all implement the
:class:`~particlefilterbox.core.model.ParticleFilterModel` protocol that the
filters consume, so this module provides thin adapters that bridge the two.

Currently supported CLI models
------------------------------
``sv`` -> :class:`particlefilterbox.models.stochastic_volatility.StochasticVolatility`
         (``basic`` variant, scalar latent log-volatility).

Currently supported CLI filter methods
---------------------------------------
``bootstrap`` -> :class:`particlefilterbox.filters.BootstrapPF`
``apf``       -> :class:`particlefilterbox.filters.AuxiliaryPF`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Names accepted by --model across the CLI commands.
SUPPORTED_MODELS = ("sv",)

# Names accepted by --method across the CLI commands.
SUPPORTED_METHODS = ("bootstrap", "apf")

# Estimation methods accepted by ``estimate --method``.
SUPPORTED_ESTIMATORS = ("pmmh",)


class _SVFilterAdapter:
    """Adapt :class:`StochasticVolatility` to the particle-filter protocol.

    The filters (Bootstrap, APF, ...) call ``initial_distribution(n, rng)``,
    ``transition(particles, t, rng)`` and ``log_observation_likelihood(
    particles, y_t, t)``. The :class:`StochasticVolatility` model exposes
    ``initial_state``, ``transition(state, rng)`` (no ``t``) and
    ``log_observation_density(y, state)``. This adapter reconciles the two
    signatures so the model can be filtered directly.
    """

    def __init__(self, sv: Any) -> None:
        self._sv = sv
        self.k_states = sv.k_states
        self.k_obs = sv.k_obs

    @property
    def param_names(self) -> list[str]:
        return list(self._sv.param_names)

    @property
    def params(self) -> dict[str, float]:
        return self._sv.params

    def initial_distribution(
        self, n_particles: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        states = self._sv.initial_state(n_particles, rng)
        return np.atleast_2d(states) if states.ndim == 1 else states

    def transition(
        self,
        particles: NDArray[np.float64],
        t: int,  # noqa: ARG002 - SV transition is time-homogeneous
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        return self._sv.transition(particles, rng)

    def log_observation_likelihood(
        self,
        particles: NDArray[np.float64],
        y_t: NDArray[np.float64],
        t: int,  # noqa: ARG002 - SV observation density is time-homogeneous
    ) -> NDArray[np.float64]:
        y = np.asarray(y_t).ravel()
        y_scalar = y[0] if y.size >= 1 else float(y_t)
        return self._sv.log_observation_density(y_scalar, particles)


def build_model(name: str, params: dict[str, float] | None = None) -> Any:
    """Return the raw library model for a CLI ``--model`` name.

    Parameters
    ----------
    name : str
        CLI model name (see :data:`SUPPORTED_MODELS`).
    params : dict[str, float] | None
        Optional parameter overrides passed to the model constructor.

    Returns
    -------
    Any
        The constructed library model instance.

    Raises
    ------
    ValueError
        If ``name`` is not a supported CLI model.
    """
    if name == "sv":
        from particlefilterbox.models.stochastic_volatility import StochasticVolatility

        return StochasticVolatility(
            variant="basic", params=params if params else None
        )

    msg = (
        f"Unknown model '{name}'. Supported models: "
        f"{', '.join(SUPPORTED_MODELS)}."
    )
    raise ValueError(msg)


def build_filter_model(name: str, params: dict[str, float] | None = None) -> Any:
    """Return a filter-protocol-compatible model for a CLI ``--model`` name.

    Wraps the raw library model in an adapter when its native interface does
    not match the particle-filter protocol.
    """
    raw = build_model(name, params)
    if name == "sv":
        return _SVFilterAdapter(raw)
    return raw


def build_filter(method: str, model: Any, n_particles: int, seed: int | None) -> Any:
    """Construct a particle filter for a CLI ``--method`` name.

    Parameters
    ----------
    method : str
        CLI filter method (see :data:`SUPPORTED_METHODS`).
    model : Any
        A filter-protocol-compatible model (see :func:`build_filter_model`).
    n_particles : int
        Number of particles.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    Any
        The constructed filter instance.

    Raises
    ------
    ValueError
        If ``method`` is not a supported CLI filter.
    """
    from particlefilterbox.core.config import PFConfig

    config = PFConfig(n_particles=n_particles, seed=seed)
    config.validate()

    if method == "bootstrap":
        from particlefilterbox.filters import BootstrapPF

        return BootstrapPF(model=model, config=config)
    if method == "apf":
        from particlefilterbox.filters import AuxiliaryPF

        return AuxiliaryPF(model=model, config=config)

    msg = (
        f"Unknown method '{method}'. Supported methods: "
        f"{', '.join(SUPPORTED_METHODS)}."
    )
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# PMMH (parameter estimation) adapters
# ---------------------------------------------------------------------------


@dataclass
class _PFLogLikResult:
    """Minimal particle-filter result exposing ``log_likelihood``.

    PMMH (via :class:`~particlefilterbox.pmcmc.base.BasePMCMC`) only reads the
    ``.log_likelihood`` attribute of the object returned by ``model.filter``.
    """

    log_likelihood: float


class _SVPMMHAdapter:
    """Adapt :class:`StochasticVolatility` to the PMMH model interface.

    PMMH requires the model to expose ``param_names`` (a list), a parameter
    vector via ``get_params()``/``set_params(theta)`` (in ``param_names``
    order), and ``filter(endog, n_particles, rng)`` returning an object with a
    ``.log_likelihood`` attribute. The SV model stores parameters in a dict and
    exposes the bootstrap-filter building blocks (``initial_state``,
    ``transition``, ``log_observation_density``); this adapter bridges the two.

    Invalid parameter values (``|phi| >= 1`` or ``sigma <= 0``) yield a
    ``-inf`` log-likelihood so PMMH rejects the proposal.
    """

    def __init__(self, sv: Any) -> None:
        self._sv = sv
        self.param_names = list(sv.param_names)

    def get_params(self) -> NDArray[np.float64]:
        """Return the current parameter vector in ``param_names`` order."""
        return np.array(
            [float(self._sv.params[name]) for name in self.param_names],
            dtype=np.float64,
        )

    def set_params(self, theta: NDArray[np.float64]) -> None:
        """Write ``theta`` back into the model's parameter dict."""
        theta = np.atleast_1d(np.asarray(theta, dtype=np.float64))
        for name, value in zip(self.param_names, theta, strict=True):
            self._sv.params[name] = float(value)

    @staticmethod
    def _as_obs_series(endog: NDArray[np.float64]) -> NDArray[np.float64]:
        """Coerce ``endog`` of shape ``(T,)`` or ``(T, 1)`` to scalars ``(T,)``."""
        arr = np.asarray(endog, dtype=np.float64)
        if arr.ndim == 1:
            return arr
        return arr[:, 0]

    def filter(
        self,
        endog: NDArray[np.float64],
        n_particles: int = 200,
        rng: np.random.Generator | None = None,
    ) -> _PFLogLikResult:
        """Run a bootstrap particle filter and return the log-likelihood.

        Mirrors the numerically-stable accumulation used in the reference SV
        workflow: per step ``log_lik += max_lw + log(mean(exp(log_w - max_lw)))``
        with multinomial resampling each step.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Guard obviously invalid parameters before touching the filter.
        phi = float(self._sv.params.get("phi", 0.0))
        sigma = float(self._sv.params.get("sigma", 1.0))
        if not np.isfinite(phi) or not np.isfinite(sigma):
            return _PFLogLikResult(log_likelihood=-np.inf)
        if abs(phi) >= 1.0 or sigma <= 0.0:
            return _PFLogLikResult(log_likelihood=-np.inf)

        y = self._as_obs_series(endog)
        n_steps = y.shape[0]

        try:
            particles = self._sv.initial_state(n_particles, rng)
            log_lik = 0.0
            for t in range(n_steps):
                if t > 0:
                    particles = self._sv.transition(particles, rng)
                log_w = self._sv.log_observation_density(float(y[t]), particles)
                max_lw = float(np.max(log_w))
                if not np.isfinite(max_lw):
                    return _PFLogLikResult(log_likelihood=-np.inf)
                w = np.exp(log_w - max_lw)
                mean_w = float(np.mean(w))
                if mean_w <= 0.0 or not np.isfinite(mean_w):
                    return _PFLogLikResult(log_likelihood=-np.inf)
                log_lik += max_lw + np.log(mean_w)
                w_sum = float(np.sum(w))
                w = w / w_sum
                indices = rng.choice(n_particles, size=n_particles, p=w)
                particles = particles[indices]
        except (ValueError, FloatingPointError):
            return _PFLogLikResult(log_likelihood=-np.inf)

        if not np.isfinite(log_lik):
            return _PFLogLikResult(log_likelihood=-np.inf)
        return _PFLogLikResult(log_likelihood=float(log_lik))


class _DictPrior:
    """Independent prior built from a model's ``default_prior()`` dict.

    Each parameter spec maps to a scipy.stats distribution:

    * ``normal``        -> ``norm(loc, scale)``
    * ``beta``          -> ``beta(a, b)``
    * ``inverse_gamma`` -> ``invgamma(a, scale=b)``

    The parameter order is fixed by ``param_names``. ``logpdf`` returns the sum
    of per-parameter log-densities (``-inf`` if any value is outside its
    support / non-finite); ``sample`` draws each parameter independently.
    """

    def __init__(
        self,
        prior_spec: dict[str, dict[str, Any]],
        param_names: list[str],
    ) -> None:
        from scipy import stats

        self.param_names = list(param_names)
        self._dists: list[Any] = []
        for name in self.param_names:
            if name not in prior_spec:
                msg = f"No prior specified for parameter '{name}'."
                raise ValueError(msg)
            spec = prior_spec[name]
            dist_name = spec.get("distribution")
            if dist_name == "normal":
                self._dists.append(stats.norm(loc=spec["loc"], scale=spec["scale"]))
            elif dist_name == "beta":
                self._dists.append(stats.beta(spec["a"], spec["b"]))
            elif dist_name == "inverse_gamma":
                self._dists.append(stats.invgamma(spec["a"], scale=spec["b"]))
            else:
                msg = (
                    f"Unsupported prior distribution '{dist_name}' for "
                    f"parameter '{name}'. Supported: normal, beta, "
                    f"inverse_gamma."
                )
                raise ValueError(msg)

    def logpdf(self, theta: NDArray[np.float64]) -> float:
        """Return the summed independent log-prior density at ``theta``."""
        theta = np.atleast_1d(np.asarray(theta, dtype=np.float64))
        total = 0.0
        for value, dist in zip(theta, self._dists, strict=True):
            lp = float(dist.logpdf(value))
            if not np.isfinite(lp):
                return -np.inf
            total += lp
        return total

    def sample(self, rng: np.random.Generator) -> NDArray[np.float64]:
        """Draw an independent sample from the prior."""
        return np.array(
            [float(dist.rvs(random_state=rng)) for dist in self._dists],
            dtype=np.float64,
        )


def build_pmmh(
    model_name: str,
    n_particles: int,
    n_iterations: int,
    seed: int | None,
    params: dict[str, float] | None = None,
) -> tuple[Any, _SVPMMHAdapter, _DictPrior]:
    """Build a configured PMMH sampler for a CLI ``--model`` name.

    Parameters
    ----------
    model_name : str
        CLI model name (see :data:`SUPPORTED_MODELS`).
    n_particles : int
        Number of particles for the bootstrap filter inside PMMH.
    n_iterations : int
        Number of MCMC iterations.
    seed : int | None
        Random seed for reproducibility.
    params : dict[str, float] | None
        Optional initial parameter overrides for the model.

    Returns
    -------
    tuple
        ``(pmmh, adapter, prior)`` where ``pmmh`` is a ready-to-run
        :class:`~particlefilterbox.pmcmc.pmmh.PMMH` instance.

    Raises
    ------
    ValueError
        If ``model_name`` is not a supported CLI model.
    """
    from particlefilterbox.pmcmc.pmmh import PMMH

    if model_name != "sv":
        msg = (
            f"Unknown model '{model_name}'. Supported models for estimation: "
            f"{', '.join(SUPPORTED_MODELS)}."
        )
        raise ValueError(msg)

    sv = build_model(model_name, params)
    adapter = _SVPMMHAdapter(sv)
    prior = _DictPrior(sv.default_prior(), adapter.param_names)

    pmmh = PMMH(
        model=adapter,
        prior=prior,
        n_particles=n_particles,
        n_iterations=n_iterations,
        seed=seed,
    )
    return pmmh, adapter, prior
