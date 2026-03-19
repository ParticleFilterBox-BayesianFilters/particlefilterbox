"""Tests for report generation system."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from particlefilterbox.reports.base import BaseReport
from particlefilterbox.reports.pf_report import PFReportTransformer
from particlefilterbox.reports.pmcmc_report import PMCMCReportTransformer
from particlefilterbox.reports.sv_report import SVReportTransformer
from particlefilterbox.reports.dsge_report import DSGEReportTransformer


@pytest.fixture
def mock_pf_results() -> SimpleNamespace:
    """Create mock particle filter results."""
    rng = np.random.default_rng(42)
    T, N, d = 100, 500, 2
    return SimpleNamespace(
        particles=rng.standard_normal((T, N, d)),
        weights=np.ones((T, N)) / N,
        filtered_mean=rng.standard_normal((T, d)),
        ess=rng.uniform(200, 500, size=T),
        log_likelihood=-150.0,
        log_likelihoods=rng.standard_normal(T) - 2.0,
        n_particles=N,
    )


@pytest.fixture
def mock_pmcmc_results() -> SimpleNamespace:
    """Create mock PMCMC results."""
    rng = np.random.default_rng(42)
    n_iter = 2000
    k_params = 3
    chain = rng.standard_normal((n_iter, k_params))
    chain[:, 0] += 0.5  # mu
    chain[:, 1] = 0.95 + chain[:, 1] * 0.02  # phi
    chain[:, 2] = np.abs(chain[:, 2] * 0.1)  # sigma
    return SimpleNamespace(
        chain=chain,
        param_names=["mu", "phi", "sigma_eta"],
        acceptance_rate=0.23,
        observations=rng.standard_normal(100),
    )


class TestBaseReport:
    """Tests for BaseReport."""

    def test_create_report(self) -> None:
        """BaseReport should be creatable with title."""
        report = BaseReport(title="Test Report")
        assert report.title == "Test Report"
        assert len(report.sections) == 0

    def test_add_section(self) -> None:
        """add_section should add a section to the report."""
        report = BaseReport(title="Test")
        report.add_section("Section 1", "Content here")
        assert len(report.sections) == 1
        assert report.sections[0].title == "Section 1"

    def test_to_html(self) -> None:
        """to_html should produce valid HTML string."""
        report = BaseReport(title="Test Report")
        report.add_section("Summary", "Test content")
        html = report.to_html()
        assert "<!DOCTYPE html>" in html
        assert "Test Report" in html
        assert "Summary" in html
        assert "Test content" in html

    def test_to_latex(self) -> None:
        """to_latex should produce valid LaTeX string."""
        report = BaseReport(title="Test Report")
        report.add_section("Summary", "Test content")
        latex = report.to_latex()
        assert r"\documentclass" in latex
        assert "Test Report" in latex
        assert r"\subsection{Summary}" in latex

    def test_to_markdown(self) -> None:
        """to_markdown should produce valid Markdown string."""
        report = BaseReport(title="Test Report")
        report.add_section("Summary", "Test content")
        md = report.to_markdown()
        assert "# Test Report" in md
        assert "## Summary" in md
        assert "Test content" in md

    def test_html_with_table(self) -> None:
        """to_html should render tables correctly."""
        report = BaseReport(title="Test")
        report.add_section(
            "Data",
            tables=[
                {
                    "headers": ["Name", "Value"],
                    "rows": [["a", "1"], ["b", "2"]],
                    "caption": "Test Table",
                }
            ],
        )
        html = report.to_html()
        assert "<table>" in html
        assert "<th>" in html
        assert "Test Table" in html

    def test_markdown_with_table(self) -> None:
        """to_markdown should render tables correctly."""
        report = BaseReport(title="Test")
        report.add_section(
            "Data",
            tables=[
                {
                    "headers": ["Name", "Value"],
                    "rows": [["a", "1"], ["b", "2"]],
                    "caption": "Test Table",
                }
            ],
        )
        md = report.to_markdown()
        assert "| Name | Value |" in md
        assert "| --- | --- |" in md
        assert "| a | 1 |" in md


class TestPFReportHTML:
    """Test PF report HTML generation."""

    def test_pf_report_html(self, mock_pf_results: Any) -> None:
        """PFReportTransformer should generate valid HTML report."""
        transformer = PFReportTransformer()
        report = transformer.transform(mock_pf_results)
        html = report.to_html()
        assert "Particle Filter Report" in html
        assert "Summary" in html
        assert "ESS" in html
        assert "Weight" in html
        assert "Log-Likelihood" in html

    def test_pf_report_has_sections(self, mock_pf_results: Any) -> None:
        """PFReportTransformer should generate 5 sections."""
        transformer = PFReportTransformer()
        report = transformer.transform(mock_pf_results)
        assert len(report.sections) == 5


class TestPMCMCReportHTML:
    """Test PMCMC report HTML generation."""

    def test_pmcmc_report_html(self, mock_pmcmc_results: Any) -> None:
        """PMCMCReportTransformer should generate valid HTML report."""
        transformer = PMCMCReportTransformer()
        report = transformer.transform(mock_pmcmc_results)
        html = report.to_html()
        assert "PMCMC" in html
        assert "Posterior" in html
        assert "mu" in html
        assert "phi" in html

    def test_pmcmc_report_has_sections(self, mock_pmcmc_results: Any) -> None:
        """PMCMCReportTransformer should generate 5 sections."""
        transformer = PMCMCReportTransformer()
        report = transformer.transform(mock_pmcmc_results)
        assert len(report.sections) == 5


class TestReportMarkdown:
    """Test report Markdown generation."""

    def test_pf_report_markdown(self, mock_pf_results: Any) -> None:
        """PF report should generate valid Markdown."""
        transformer = PFReportTransformer()
        report = transformer.transform(mock_pf_results)
        md = report.to_markdown()
        assert "# Particle Filter Report" in md
        assert "## Summary" in md

    def test_pmcmc_report_markdown(self, mock_pmcmc_results: Any) -> None:
        """PMCMC report should generate valid Markdown."""
        transformer = PMCMCReportTransformer()
        report = transformer.transform(mock_pmcmc_results)
        md = report.to_markdown()
        assert "# PMCMC" in md
        assert "## Posterior" in md


class TestSVReport:
    """Test SV report generation."""

    def test_sv_report_transform(self, mock_pmcmc_results: Any) -> None:
        """SVReportTransformer should generate a report."""
        transformer = SVReportTransformer()
        report = transformer.transform(mock_pmcmc_results)
        assert len(report.sections) >= 3
        html = report.to_html()
        assert "Stochastic Volatility" in html


class TestDSGEReport:
    """Test DSGE report generation."""

    def test_dsge_report_transform(self) -> None:
        """DSGEReportTransformer should generate a report."""
        results = SimpleNamespace(
            model_name="NK_3eq",
            n_states=3,
            n_obs=2,
            n_shocks=2,
            chain=np.random.default_rng(42).standard_normal((1000, 4)),
            param_names=["sigma", "kappa", "phi_pi", "phi_y"],
            log_likelihood=-200.0,
            log_evidence=-210.0,
        )
        transformer = DSGEReportTransformer()
        report = transformer.transform(results)
        assert len(report.sections) >= 3
        html = report.to_html()
        assert "DSGE" in html
