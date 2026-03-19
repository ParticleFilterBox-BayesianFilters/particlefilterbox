"""Tests for model comparison."""

import numpy as np
import pytest

from particlefilterbox.diagnostics.model_comparison import ModelComparison


class MockFilterResult:
    """Mock filter result with log_likelihood."""

    def __init__(self, log_lik: float) -> None:
        self.log_likelihood = log_lik


class MockFilter:
    """Mock filter that returns predetermined log-likelihood."""

    def __init__(self, log_lik: float) -> None:
        self._log_lik = log_lik

    def filter(self, endog: np.ndarray) -> MockFilterResult:  # type: ignore[type-arg]
        return MockFilterResult(self._log_lik)


class MockFilterFactory:
    """Factory that creates filters with predetermined log-likelihoods."""

    def __init__(self, log_lik: float) -> None:
        self._log_lik = log_lik

    def create(self, model: object, n_particles: int) -> MockFilter:
        return MockFilter(self._log_lik)


class TestModelComparison:
    """Tests for ModelComparison."""

    def test_bayes_factor_correct_model(self) -> None:
        """Model with higher log-evidence should be favored."""
        mc = ModelComparison(n_particles=100)
        mc.add_model("good", object(), MockFilterFactory(log_lik=-100.0))
        mc.add_model("bad", object(), MockFilterFactory(log_lik=-200.0))
        mc.run(endog=np.random.randn(50))

        bf = mc.bayes_factor("good", "bad")
        assert bf["log_bayes_factor"] > 0
        assert bf["favored_model"] == "good"
        assert bf["interpretation"] == "Very strong evidence"

    def test_ranking(self) -> None:
        """Ranking should order by log evidence."""
        mc = ModelComparison(n_particles=100)
        mc.add_model("A", object(), MockFilterFactory(log_lik=-150.0))
        mc.add_model("B", object(), MockFilterFactory(log_lik=-100.0))
        mc.add_model("C", object(), MockFilterFactory(log_lik=-200.0))
        mc.run(endog=np.random.randn(50))

        ranking = mc.ranking()
        assert ranking[0][0] == "B"
        assert ranking[1][0] == "A"
        assert ranking[2][0] == "C"

    def test_log_evidence(self) -> None:
        """Test log_evidence retrieval."""
        mc = ModelComparison(n_particles=100)
        mc.add_model("A", object(), MockFilterFactory(log_lik=-100.0))
        mc.run(endog=np.random.randn(50))

        le = mc.log_evidence("A")
        assert abs(le - (-100.0)) < 1e-10

    def test_log_evidence_all(self) -> None:
        """Test log_evidence for all models."""
        mc = ModelComparison(n_particles=100)
        mc.add_model("A", object(), MockFilterFactory(log_lik=-100.0))
        mc.add_model("B", object(), MockFilterFactory(log_lik=-200.0))
        mc.run(endog=np.random.randn(50))

        le_all = mc.log_evidence()
        assert isinstance(le_all, dict)
        assert "A" in le_all
        assert "B" in le_all

    def test_summary(self) -> None:
        """Test summary output."""
        mc = ModelComparison(n_particles=100)
        mc.add_model("A", object(), MockFilterFactory(log_lik=-100.0))
        mc.add_model("B", object(), MockFilterFactory(log_lik=-200.0))
        mc.run(endog=np.random.randn(50))

        s = mc.summary()
        assert "ranking" in s
        assert "pairwise_bayes_factors" in s
        assert s["n_models"] == 2

    def test_run_required(self) -> None:
        """Operations before run() should raise."""
        mc = ModelComparison()
        with pytest.raises(RuntimeError, match="Must call run"):
            mc.ranking()

    def test_interpretation_levels(self) -> None:
        """Test all Kass-Raftery interpretation levels."""
        mc = ModelComparison(n_particles=100)

        # No evidence: |log BF| < 1
        mc.add_model("A", object(), MockFilterFactory(log_lik=-100.0))
        mc.add_model("B", object(), MockFilterFactory(log_lik=-100.5))
        mc.run(endog=np.random.randn(50))
        bf = mc.bayes_factor("A", "B")
        assert bf["interpretation"] == "No evidence"
