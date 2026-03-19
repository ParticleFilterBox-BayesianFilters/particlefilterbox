"""GPU backend for particle filter operations.

Supports CuPy (CUDA) and JAX (XLA) backends for GPU-accelerated
particle filtering. Handles missing backends gracefully.

Reference:
    Lam, S.K., Pitrou, A. & Seibert, S. (2015). Numba: A LLVM-based
    Python JIT compiler.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class GPUBackend:
    """Abstract GPU backend for particle filter operations.

    Provides a unified interface for GPU-accelerated array operations
    using either CuPy or JAX as the underlying library.

    Parameters:
        backend: Which GPU library to use ('cupy' or 'jax').

    Raises:
        ImportError: If the requested backend is not installed.

    Examples:
        >>> gpu = GPUBackend(backend='cupy')
        >>> particles_gpu = gpu.to_device(particles)
        >>> result = gpu.propagate(particles_gpu, noise)
    """

    def __init__(self, backend: str = "cupy") -> None:
        self.backend_name = backend
        self._xp: Any = None  # array module (cupy or jax.numpy)
        self._backend_module: Any = None

        if backend == "cupy":
            try:
                import cupy as cp  # type: ignore[import-not-found]

                self._xp = cp
                self._backend_module = cp
            except ImportError:
                raise ImportError(
                    "CuPy is not installed. Install with: pip install cupy-cuda12x "
                    "(replace 12x with your CUDA version)"
                ) from None
        elif backend == "jax":
            try:
                import jax  # type: ignore[import-not-found]
                import jax.numpy as jnp  # type: ignore[import-not-found]

                self._xp = jnp
                self._backend_module = jax
            except ImportError:
                raise ImportError(
                    "JAX is not installed. Install with: pip install jax jaxlib"
                ) from None
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'cupy' or 'jax'.")

    @property
    def xp(self) -> Any:
        """Array module (cupy or jax.numpy)."""
        return self._xp

    def to_device(self, array: NDArray[np.float64]) -> Any:
        """Transfer a NumPy array to GPU.

        Parameters:
            array: NumPy array.

        Returns:
            GPU array.
        """
        if self.backend_name == "cupy":
            return self._xp.asarray(array)
        elif self.backend_name == "jax":
            import jax  # type: ignore[import-not-found]

            return jax.device_put(np.asarray(array))  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType,reportReturnType]
        return array

    def to_host(self, array: Any) -> NDArray[np.float64]:
        """Transfer a GPU array to CPU as NumPy.

        Parameters:
            array: GPU array.

        Returns:
            NumPy array.
        """
        if self.backend_name == "cupy":
            return self._xp.asnumpy(array).astype(np.float64)
        elif self.backend_name == "jax":
            return np.asarray(array, dtype=np.float64)
        return np.asarray(array, dtype=np.float64)

    def propagate(
        self,
        particles: Any,
        noise: Any,
    ) -> Any:
        """Propagate particles by adding noise.

        Parameters:
            particles: Particle array on GPU, shape (N, D).
            noise: Noise array on GPU, shape (N, D).

        Returns:
            Updated particles on GPU.
        """
        return particles + noise

    def compute_weights(
        self,
        particles: Any,
        observation: Any,
        sigma: float = 1.0,
    ) -> Any:
        """Compute log-weights (Gaussian observation model).

        Parameters:
            particles: Particle array on GPU, shape (N,) or (N, D).
            observation: Observation scalar or array.
            sigma: Observation noise standard deviation.

        Returns:
            Log-weights on GPU, shape (N,).
        """
        xp = self._xp
        diff = particles - observation
        if diff.ndim > 1:
            return -0.5 * xp.sum(diff**2, axis=1) / (sigma**2)
        return -0.5 * diff**2 / (sigma**2)

    def log_sum_exp(self, log_w: Any) -> Any:
        """Log-sum-exp on GPU.

        Parameters:
            log_w: Log-weights on GPU.

        Returns:
            Scalar log-sum-exp value.
        """
        xp = self._xp
        max_w = xp.max(log_w)
        return max_w + xp.log(xp.sum(xp.exp(log_w - max_w)))

    def normalize_log_weights(self, log_w: Any) -> Any:
        """Normalize log-weights on GPU.

        Parameters:
            log_w: Unnormalized log-weights.

        Returns:
            Normalized weights summing to 1.
        """
        lse = self.log_sum_exp(log_w)
        xp = self._xp
        return xp.exp(log_w - lse)

    def summary(self) -> dict[str, Any]:
        """GPU backend information.

        Returns:
            Dictionary with backend info.
        """
        info: dict[str, Any] = {
            "backend": self.backend_name,
            "available": True,
        }
        if self.backend_name == "cupy":
            try:
                import cupy as cp  # type: ignore[import-not-found]

                info["device_name"] = str(cp.cuda.Device().name)  # type: ignore[attr-defined]
                info["gpu_memory"] = int(cp.cuda.Device().mem_info[1])  # type: ignore[attr-defined]
            except Exception:
                pass
        elif self.backend_name == "jax":
            try:
                import jax  # type: ignore[import-not-found]

                devices: list[Any] = list(jax.devices())  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                info["devices"] = [str(d) for d in devices]
            except Exception:
                pass
        return info
