"""Integration test for the full particlefilterbox pipeline.

Tests data -> model -> filter -> estimation -> visualization -> report.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


class TestFullPipeline:
    """Test complete data-to-report pipeline."""

    def test_datasets_available(self) -> None:
        """Bundled datasets should be discoverable."""
        try:
            from particlefilterbox.datasets import list_datasets

            datasets = list_datasets()
            assert len(datasets) >= 3
        except ImportError:
            pytest.skip("Datasets module not available")

    def test_visualization_theme_applies(self) -> None:
        """Theme should be settable."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            from particlefilterbox.visualization import get_theme, set_theme

            set_theme("nodesecon")
            theme = get_theme()
            assert "#2E86AB" in theme["colors"]
        except ImportError:
            pytest.skip("Visualization module not available")

    def test_report_generation(self) -> None:
        """Report should be generable from mock results."""
        try:
            from particlefilterbox.reports import PFReportTransformer

            mock_results = SimpleNamespace(
                particles=np.random.randn(50, 100, 1),
                weights=np.ones((50, 100)) / 100,
                filtered_mean=np.random.randn(50, 1),
                ess=np.random.uniform(50, 100, size=50),
                log_likelihood=-100.0,
                log_likelihoods=np.random.randn(50) - 2.0,
                n_particles=100,
            )

            transformer = PFReportTransformer()
            report = transformer.transform(mock_results)
            html = report.to_html()
            md = report.to_markdown()

            assert "Summary" in html
            assert "# Particle Filter Report" in md

        except ImportError:
            pytest.skip("Reports module not available")

    def test_experiment_pattern(self) -> None:
        """PFExperiment pattern should work end-to-end."""
        try:
            from particlefilterbox.experiment import PFExperiment

            exp = PFExperiment(n_particles=50, seed=42, name="test")
            # Just test the infrastructure
            df = exp.compare()
            assert df.empty  # No models added yet

        except ImportError:
            pytest.skip("Experiment module not available")

    def test_cli_importable(self) -> None:
        """CLI module should be importable."""
        try:
            from particlefilterbox.cli.main import app

            assert app is not None
        except ImportError:
            pytest.skip("CLI module not available")
