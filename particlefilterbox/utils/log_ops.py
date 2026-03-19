"""Log-scale operations for numerical stability in particle filtering."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def log_sum_exp(log_w: NDArray[np.float64]) -> float:
    """Compute log(sum(exp(log_w))) in a numerically stable way.

    Uses the identity: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))

    Parameters
    ----------
    log_w : ndarray
        Log-weights array.

    Returns
    -------
    float
        log(sum(exp(log_w)))
    """
    max_log_w = np.max(log_w)
    if np.isinf(max_log_w):
        return float(max_log_w)
    return float(max_log_w + np.log(np.sum(np.exp(log_w - max_log_w))))


def log_mean_exp(log_w: NDArray[np.float64]) -> float:
    """Compute log(mean(exp(log_w))) in a numerically stable way.

    Parameters
    ----------
    log_w : ndarray
        Log-weights array.

    Returns
    -------
    float
        log(mean(exp(log_w)))
    """
    return log_sum_exp(log_w) - np.log(len(log_w))


def normalize_log_weights(log_w: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize log-weights to obtain probability weights that sum to 1.

    Parameters
    ----------
    log_w : ndarray
        Unnormalized log-weights.

    Returns
    -------
    ndarray
        Normalized weights (sum to 1).
    """
    lse = log_sum_exp(log_w)
    return np.exp(log_w - lse)


def ess_from_log_weights(log_w: NDArray[np.float64]) -> float:
    """Compute Effective Sample Size from log-weights.

    ESS = 1 / sum(w_i^2) where w_i are normalized weights.

    Parameters
    ----------
    log_w : ndarray
        Unnormalized log-weights.

    Returns
    -------
    float
        Effective sample size, in [1, N].
    """
    w = normalize_log_weights(log_w)
    return ess_from_weights(w)


def ess_from_weights(w: NDArray[np.float64]) -> float:
    """Compute Effective Sample Size from normalized weights.

    ESS = 1 / sum(w_i^2)

    Parameters
    ----------
    w : ndarray
        Normalized weights (must sum to 1).

    Returns
    -------
    float
        Effective sample size, in [1, N].
    """
    return float(1.0 / np.sum(w**2))
