"""Tests for PFConfig."""

from __future__ import annotations

import pytest

from particlefilterbox.core.config import PFConfig


class TestPFConfig:
    def test_default_values(self) -> None:
        config = PFConfig()
        assert config.n_particles == 1000
        assert config.resampling == "systematic"
        assert config.ess_threshold == 0.5

    def test_validation_negative_particles(self) -> None:
        config = PFConfig(n_particles=-1)
        with pytest.raises(ValueError, match="n_particles must be positive"):
            config.validate()

    def test_validation_zero_particles(self) -> None:
        config = PFConfig(n_particles=0)
        with pytest.raises(ValueError, match="n_particles must be positive"):
            config.validate()

    def test_validation_threshold_too_low(self) -> None:
        config = PFConfig(ess_threshold=0.0)
        with pytest.raises(ValueError, match="ess_threshold must be in"):
            config.validate()

    def test_validation_threshold_too_high(self) -> None:
        config = PFConfig(ess_threshold=1.5)
        with pytest.raises(ValueError, match="ess_threshold must be in"):
            config.validate()

    def test_validation_invalid_resampling(self) -> None:
        config = PFConfig(resampling="invalid")
        with pytest.raises(ValueError, match="Unknown resampling"):
            config.validate()

    def test_validation_passes(self) -> None:
        config = PFConfig()
        config.validate()  # Should not raise

    def test_effective_threshold(self) -> None:
        config = PFConfig(n_particles=2000, ess_threshold=0.5)
        assert config.effective_threshold() == 1000.0
