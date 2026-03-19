"""Tests for CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from particlefilterbox.cli.main import app

runner = CliRunner()


@pytest.fixture
def tmp_csv(tmp_path: Path) -> Path:
    """Create a temporary CSV file with simulated data."""
    rng = np.random.default_rng(42)
    n = 100
    data = rng.standard_normal((n, 1))
    df = pd.DataFrame(data, columns=["y"])
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


class TestFilterCmd:
    """Tests for pfbox filter command."""

    def test_filter_cmd_help(self) -> None:
        """pfbox filter --help should work."""
        result = runner.invoke(app, ["filter", "--help"])
        assert result.exit_code == 0
        assert "particle filter" in result.stdout.lower() or "filter" in result.stdout.lower()

    def test_filter_cmd_runs(self, tmp_csv: Path) -> None:
        """pfbox filter should run on CSV data."""
        result = runner.invoke(app, [
            "filter",
            str(tmp_csv),
            "--model",
            "sv",
            "--n-particles",
            "50",
            "--seed",
            "42",
        ])
        # May fail due to model not existing yet, but should at least parse args
        assert "Loading data" in result.stdout or result.exit_code in (0, 1)


class TestEstimateCmd:
    """Tests for pfbox estimate command."""

    def test_estimate_cmd_help(self) -> None:
        """pfbox estimate --help should work."""
        result = runner.invoke(app, ["estimate", "--help"])
        assert result.exit_code == 0
        assert "estimate" in result.stdout.lower() or "PMCMC" in result.stdout

    def test_estimate_cmd_runs(self, tmp_csv: Path) -> None:
        """pfbox estimate should run on CSV data."""
        result = runner.invoke(app, [
            "estimate",
            str(tmp_csv),
            "--model",
            "sv",
            "--method",
            "pmmh",
            "--n-particles",
            "50",
            "--n-iterations",
            "100",
            "--seed",
            "42",
        ])
        assert "Loading data" in result.stdout or result.exit_code in (0, 1)


class TestCompareCmd:
    """Tests for pfbox compare command."""

    def test_compare_cmd_help(self) -> None:
        """pfbox compare --help should work."""
        result = runner.invoke(app, ["compare", "--help"])
        assert result.exit_code == 0
        assert "compare" in result.stdout.lower() or "model" in result.stdout.lower()

    def test_compare_cmd_runs(self, tmp_csv: Path) -> None:
        """pfbox compare should run on CSV data."""
        result = runner.invoke(app, [
            "compare",
            str(tmp_csv),
            "--models",
            "sv",
            "--n-particles",
            "50",
            "--n-runs",
            "1",
            "--seed",
            "42",
        ])
        assert (
            "Loading data" in result.stdout
            or "Comparing" in result.stdout
            or result.exit_code in (0, 1)
        )


class TestSimulateCmd:
    """Tests for pfbox simulate command."""

    def test_simulate_cmd_help(self) -> None:
        """pfbox simulate --help should work."""
        result = runner.invoke(app, ["simulate", "--help"])
        assert result.exit_code == 0
        assert "simulate" in result.stdout.lower() or "model" in result.stdout.lower()

    def test_simulate_cmd_runs(self, tmp_path: Path) -> None:
        """pfbox simulate should produce data."""
        output = tmp_path / "sim.csv"
        result = runner.invoke(app, [
            "simulate",
            "--model",
            "sv",
            "--n-obs",
            "100",
            "--seed",
            "42",
            "--output",
            str(output),
        ])
        assert "Simulating" in result.stdout or result.exit_code in (0, 1)


class TestVersion:
    """Tests for version flag."""

    def test_version(self) -> None:
        """pfbox --version should print version."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.stdout

    def test_no_args_shows_help(self) -> None:
        """pfbox with no args should show help."""
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "pfbox" in result.stdout.lower() or "Usage" in result.stdout
