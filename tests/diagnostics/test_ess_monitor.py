"""Tests for ESS monitoring."""

import numpy as np
import pytest

from particlefilterbox.diagnostics.ess_monitor import AlertLevel, ESSMonitor


class TestESSMonitor:
    """Tests for ESSMonitor."""

    def test_uniform_ess_equals_n(self) -> None:
        """Uniform weights should give ESS = N."""
        n = 500
        weights = np.ones(n) / n
        ess = ESSMonitor.compute_ess(weights)
        assert abs(ess - n) < 1e-8, f"ESS should be {n}, got {ess}"

    def test_degenerate_ess_equals_one(self) -> None:
        """Degenerate weights (all on one particle) should give ESS = 1."""
        n = 500
        weights = np.zeros(n)
        weights[0] = 1.0
        ess = ESSMonitor.compute_ess(weights)
        assert abs(ess - 1.0) < 1e-8, f"ESS should be 1, got {ess}"

    def test_degenerate_alert_critical(self) -> None:
        """Degenerate weights should trigger CRITICAL alert."""
        monitor = ESSMonitor()
        weights = np.zeros(100)
        weights[0] = 1.0
        level = monitor.update_from_weights(weights, time_step=0)
        assert level == AlertLevel.CRITICAL
        assert not monitor.is_healthy()
        assert len(monitor.alerts) == 1
        assert monitor.alerts[0].level == AlertLevel.CRITICAL

    def test_warning_alert(self) -> None:
        """Low ESS should trigger WARNING alert."""
        monitor = ESSMonitor(warning_ratio=0.5)
        n = 100
        # Create weights with ESS ~ 20 (ratio 0.2 < 0.5)
        weights = np.ones(n)
        weights[0] = 50.0  # dominant particle
        weights = weights / weights.sum()
        level = monitor.update_from_weights(weights, time_step=0)
        assert level == AlertLevel.WARNING

    def test_healthy_uniform(self) -> None:
        """Uniform weights should not trigger any alerts."""
        monitor = ESSMonitor()
        weights = np.ones(100) / 100
        level = monitor.update_from_weights(weights, time_step=0)
        assert level == AlertLevel.OK
        assert monitor.is_healthy()
        assert len(monitor.alerts) == 0

    def test_ess_min_mean(self) -> None:
        """Test ess_min and ess_mean properties."""
        monitor = ESSMonitor()
        n = 100
        # Step 1: uniform
        w1 = np.ones(n) / n
        monitor.update_from_weights(w1, time_step=0)
        # Step 2: somewhat degenerate
        w2 = np.ones(n)
        w2[0] = 20.0
        w2 = w2 / w2.sum()
        monitor.update_from_weights(w2, time_step=1)

        assert monitor.ess_min < n
        assert monitor.ess_mean < n

    def test_ess_ratio_history(self) -> None:
        """Test ess_ratio_history property."""
        monitor = ESSMonitor()
        n = 100
        w = np.ones(n) / n
        monitor.update_from_weights(w, time_step=0)
        ratios = monitor.ess_ratio_history
        assert len(ratios) == 1
        assert abs(ratios[0] - 1.0) < 1e-8

    def test_summary(self) -> None:
        """Test summary output."""
        monitor = ESSMonitor()
        w = np.ones(100) / 100
        monitor.update_from_weights(w, time_step=0)
        s = monitor.summary()
        assert s["n_time_steps"] == 1
        assert s["is_healthy"] is True
        assert "ess_min" in s
        assert "ess_mean" in s

    def test_reset(self) -> None:
        """Test reset clears all state."""
        monitor = ESSMonitor()
        w = np.ones(100) / 100
        monitor.update_from_weights(w)
        monitor.reset()
        assert len(monitor.alerts) == 0
        assert np.isnan(monitor.ess_min)

    def test_auto_time_step(self) -> None:
        """Test auto-incrementing time step."""
        monitor = ESSMonitor()
        w = np.ones(100) / 100
        monitor.update_from_weights(w)
        monitor.update_from_weights(w)
        monitor.update_from_weights(w)
        assert monitor.summary()["n_time_steps"] == 3
