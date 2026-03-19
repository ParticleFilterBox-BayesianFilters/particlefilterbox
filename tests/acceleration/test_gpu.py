"""Tests for GPU backend."""

import numpy as np
import pytest


class TestGPUBackend:
    """Tests for GPUBackend."""

    def test_gpu_skip_if_unavailable_cupy(self) -> None:
        """CuPy backend should raise ImportError if not installed."""
        try:
            import cupy  # noqa: F401

            pytest.skip("CuPy is available, skip unavailability test")
        except ImportError:
            from particlefilterbox.acceleration.gpu import GPUBackend

            with pytest.raises(ImportError, match="CuPy"):
                GPUBackend(backend="cupy")

    def test_gpu_skip_if_unavailable_jax(self) -> None:
        """JAX backend should raise ImportError if not installed."""
        try:
            import jax  # noqa: F401

            pytest.skip("JAX is available, skip unavailability test")
        except ImportError:
            from particlefilterbox.acceleration.gpu import GPUBackend

            with pytest.raises(ImportError, match="JAX"):
                GPUBackend(backend="jax")

    def test_unknown_backend(self) -> None:
        """Unknown backend should raise ValueError."""
        from particlefilterbox.acceleration.gpu import GPUBackend

        with pytest.raises(ValueError, match="Unknown backend"):
            GPUBackend(backend="unknown")

    def test_cupy_operations(self) -> None:
        """Test CuPy operations if available."""
        try:
            import cupy  # noqa: F401
        except ImportError:
            pytest.skip("CuPy not installed")

        from particlefilterbox.acceleration.gpu import GPUBackend

        gpu = GPUBackend(backend="cupy")

        # Test propagate
        particles = gpu.to_device(np.ones((100, 2)))
        noise = gpu.to_device(np.zeros((100, 2)))
        result = gpu.propagate(particles, noise)
        host_result = gpu.to_host(result)
        np.testing.assert_allclose(host_result, np.ones((100, 2)))

        # Test log_sum_exp
        log_w = gpu.to_device(np.zeros(100))
        lse = gpu.log_sum_exp(log_w)
        assert abs(float(gpu.to_host(np.array([lse]))[0]) - np.log(100)) < 1e-6

    def test_jax_operations(self) -> None:
        """Test JAX operations if available."""
        try:
            import jax  # noqa: F401
        except ImportError:
            pytest.skip("JAX not installed")

        from particlefilterbox.acceleration.gpu import GPUBackend

        gpu = GPUBackend(backend="jax")

        particles = gpu.to_device(np.ones((100, 2)))
        noise = gpu.to_device(np.zeros((100, 2)))
        result = gpu.propagate(particles, noise)
        host_result = gpu.to_host(result)
        np.testing.assert_allclose(host_result, np.ones((100, 2)))
