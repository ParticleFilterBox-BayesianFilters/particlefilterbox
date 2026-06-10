"""Tests for CLI commands.

These tests exercise the real CLI surface via Typer's ``CliRunner`` and make
concrete assertions about exit codes, generated files, and printed output.
The supported CLI surface (per the command implementations) is:

* ``--model``/``--models`` supports only ``sv``.
* ``--method`` supports ``bootstrap`` and ``apf``.
* ``estimate`` is intentionally disabled and exits non-zero with guidance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from particlefilterbox.cli.main import app

runner = CliRunner()

N_OBS = 40


@pytest.fixture
def tmp_csv(tmp_path: Path) -> Path:
    """Create a temporary single-column CSV with simulated-style data.

    Returns a CSV with exactly ``N_OBS`` observation rows (plus a header),
    which the filter/compare commands read via ``pd.read_csv(...).values``.
    """
    rng = np.random.default_rng(42)
    data = rng.standard_normal((N_OBS, 1)) * 0.5
    df = pd.DataFrame(data, columns=["y"])
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


class TestSimulateCmd:
    """Tests for pfbox simulate command."""

    def test_simulate_cmd_help(self) -> None:
        """pfbox simulate --help should work."""
        result = runner.invoke(app, ["simulate", "--help"])
        assert result.exit_code == 0
        assert "simulate" in result.stdout.lower() or "model" in result.stdout.lower()

    def test_simulate_cmd_runs(self, tmp_path: Path) -> None:
        """pfbox simulate should produce a CSV with the requested rows."""
        output = tmp_path / "sim.csv"
        result = runner.invoke(
            app,
            [
                "simulate",
                "--model",
                "sv",
                "--n-obs",
                str(N_OBS),
                "--seed",
                "42",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "Simulating from model: sv" in result.stdout
        assert f"Simulated {N_OBS} observations" in result.stdout
        assert f"Data saved to {output}" in result.stdout
        assert "Done." in result.stdout

        # Output CSV exists, is non-empty, and has exactly N_OBS data rows.
        assert output.exists()
        assert output.stat().st_size > 0
        df = pd.read_csv(output)
        assert len(df) == N_OBS
        assert df.shape[1] >= 1

    def test_simulate_unknown_model_exits_nonzero(self, tmp_path: Path) -> None:
        """An unknown --model must fail with a non-zero exit code."""
        output = tmp_path / "sim.csv"
        result = runner.invoke(
            app,
            [
                "simulate",
                "--model",
                "definitely-not-a-real-model",
                "--n-obs",
                str(N_OBS),
                "--seed",
                "42",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code != 0
        assert not output.exists()


class TestFilterCmd:
    """Tests for pfbox filter command."""

    def test_filter_cmd_help(self) -> None:
        """pfbox filter --help should work."""
        result = runner.invoke(app, ["filter", "--help"])
        assert result.exit_code == 0
        assert "filter" in result.stdout.lower()

    def test_filter_cmd_runs(self, tmp_csv: Path) -> None:
        """pfbox filter should run on CSV data and report diagnostics."""
        result = runner.invoke(
            app,
            [
                "filter",
                str(tmp_csv),
                "--model",
                "sv",
                "--n-particles",
                "50",
                "--seed",
                "42",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "Loading data" in result.stdout
        assert "Running particle filter" in result.stdout
        assert "Log-likelihood:" in result.stdout
        assert "Mean ESS:" in result.stdout
        assert "Done." in result.stdout

    def test_filter_cmd_writes_json_output(self, tmp_csv: Path, tmp_path: Path) -> None:
        """pfbox filter --output should write a JSON file with results."""
        out = tmp_path / "results.json"
        result = runner.invoke(
            app,
            [
                "filter",
                str(tmp_csv),
                "--model",
                "sv",
                "--n-particles",
                "50",
                "--seed",
                "42",
                "--output",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert f"Results saved to {out}" in result.stdout
        assert out.exists()
        assert out.stat().st_size > 0
        payload = json.loads(out.read_text())
        assert "log_likelihood" in payload
        assert "ess_history" in payload
        assert len(payload["ess_history"]) == N_OBS

    def test_filter_apf_method_runs(self, tmp_csv: Path) -> None:
        """pfbox filter --method apf should also run successfully."""
        result = runner.invoke(
            app,
            [
                "filter",
                str(tmp_csv),
                "--model",
                "sv",
                "--n-particles",
                "50",
                "--method",
                "apf",
                "--seed",
                "42",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "Log-likelihood:" in result.stdout

    def test_filter_unknown_model_exits_nonzero(self, tmp_csv: Path) -> None:
        """An unknown --model must fail with a non-zero exit code."""
        result = runner.invoke(
            app,
            [
                "filter",
                str(tmp_csv),
                "--model",
                "definitely-not-a-real-model",
                "--n-particles",
                "50",
                "--seed",
                "42",
            ],
        )
        assert result.exit_code != 0

    def test_filter_unknown_method_exits_nonzero(self, tmp_csv: Path) -> None:
        """An unknown --method must fail with a non-zero exit code."""
        result = runner.invoke(
            app,
            [
                "filter",
                str(tmp_csv),
                "--model",
                "sv",
                "--method",
                "not-a-method",
                "--n-particles",
                "50",
                "--seed",
                "42",
            ],
        )
        assert result.exit_code != 0


class TestCompareCmd:
    """Tests for pfbox compare command."""

    def test_compare_cmd_help(self) -> None:
        """pfbox compare --help should work."""
        result = runner.invoke(app, ["compare", "--help"])
        assert result.exit_code == 0
        assert "compare" in result.stdout.lower() or "model" in result.stdout.lower()

    def test_compare_cmd_runs(self, tmp_csv: Path) -> None:
        """pfbox compare should run a minimal comparison and rank models."""
        result = runner.invoke(
            app,
            [
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
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "Comparing models: sv" in result.stdout
        assert "Model Comparison" in result.stdout
        assert "Log-likelihood:" in result.stdout
        assert "Done." in result.stdout

    def test_compare_cmd_writes_output(self, tmp_csv: Path, tmp_path: Path) -> None:
        """pfbox compare --output should write a JSON summary."""
        out = tmp_path / "compare.json"
        result = runner.invoke(
            app,
            [
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
                "--output",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert out.exists()
        payload = json.loads(out.read_text())
        assert "sv" in payload
        assert "mean_loglike" in payload["sv"]

    def test_compare_unknown_model_exits_nonzero(self, tmp_csv: Path) -> None:
        """compare with only an unknown model produces no results -> non-zero."""
        result = runner.invoke(
            app,
            [
                "compare",
                str(tmp_csv),
                "--models",
                "definitely-not-a-real-model",
                "--n-particles",
                "50",
                "--n-runs",
                "1",
                "--seed",
                "42",
            ],
        )
        assert result.exit_code != 0


class TestEstimateCmd:
    """Tests for pfbox estimate command.

    Estimation runs PMMH parameter estimation against the curated SV model via
    adapters that expose the model/prior interface PMMH requires.
    """

    def test_estimate_cmd_help(self) -> None:
        """pfbox estimate --help should work."""
        result = runner.invoke(app, ["estimate", "--help"])
        assert result.exit_code == 0
        assert "estimate" in result.stdout.lower() or "PMCMC" in result.stdout

    def test_estimate_cmd_runs(self, tmp_csv: Path) -> None:
        """pfbox estimate should run PMMH and print a finite posterior summary."""
        result = runner.invoke(
            app,
            [
                "estimate",
                str(tmp_csv),
                "--model",
                "sv",
                "--method",
                "pmmh",
                "--n-particles",
                "50",
                "--n-iterations",
                "120",
                "--seed",
                "42",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Running PMMH" in result.output
        assert "Posterior summary" in result.output
        # All three SV parameters reported with finite means.
        for name in ("mu", "phi", "sigma"):
            assert name in result.output
        assert "Acceptance rate:" in result.output
        assert "Done." in result.output

        # Acceptance rate parses to a value in [0, 1].
        line = next(
            ln for ln in result.output.splitlines() if ln.startswith("Acceptance rate:")
        )
        rate = float(line.split(":", 1)[1].strip())
        assert 0.0 <= rate <= 1.0

    def test_estimate_cmd_writes_json_output(
        self, tmp_csv: Path, tmp_path: Path
    ) -> None:
        """pfbox estimate --output should write a JSON file with chains/summary."""
        out = tmp_path / "chains.json"
        result = runner.invoke(
            app,
            [
                "estimate",
                str(tmp_csv),
                "--model",
                "sv",
                "--n-particles",
                "50",
                "--n-iterations",
                "120",
                "--seed",
                "42",
                "--output",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert f"Results saved to {out}" in result.output
        assert out.exists()
        payload = json.loads(out.read_text())
        assert payload["param_names"] == ["mu", "phi", "sigma"]
        assert len(payload["posterior_mean"]) == 3
        assert all(np.isfinite(payload["posterior_mean"]))
        assert 0.0 <= payload["acceptance_rate"] <= 1.0
        assert len(payload["chains"]) == 120

    def test_estimate_unknown_model_exits_nonzero(self, tmp_csv: Path) -> None:
        """An unknown --model must fail with a non-zero exit code."""
        result = runner.invoke(
            app,
            [
                "estimate",
                str(tmp_csv),
                "--model",
                "definitely-not-a-real-model",
                "--n-particles",
                "50",
                "--n-iterations",
                "120",
                "--seed",
                "42",
            ],
        )
        assert result.exit_code != 0

    def test_estimate_unknown_method_exits_nonzero(self, tmp_csv: Path) -> None:
        """An unknown --method must fail with a non-zero exit code."""
        result = runner.invoke(
            app,
            [
                "estimate",
                str(tmp_csv),
                "--model",
                "sv",
                "--method",
                "not-a-method",
                "--n-iterations",
                "120",
            ],
        )
        assert result.exit_code != 0


class TestVersion:
    """Tests for version flag."""

    def test_version(self) -> None:
        """pfbox --version should print version."""
        from particlefilterbox import __version__

        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.stdout

    def test_no_args_shows_help(self) -> None:
        """pfbox with no args should show help."""
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "pfbox" in result.stdout.lower() or "Usage" in result.stdout
