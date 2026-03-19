"""Tests for dataset loading and PFExperiment."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from particlefilterbox.datasets.load import (
    DATASETS,
    get_dataset_info,
    list_datasets,
    load_dataset,
)
from particlefilterbox.experiment import PFExperiment


class TestAllDatasetsLoadable:
    """Test that all registered datasets can be loaded."""

    def test_list_datasets_not_empty(self) -> None:
        """list_datasets should return non-empty list."""
        datasets = list_datasets()
        assert len(datasets) >= 5

    def test_list_datasets_by_category(self) -> None:
        """list_datasets should filter by category."""
        finance = list_datasets(category="finance")
        assert len(finance) >= 2
        for d in finance:
            assert d["category"] == "finance"

    @pytest.mark.parametrize("name", list(DATASETS.keys()))
    def test_load_dataset(self, name: str) -> None:
        """Each registered dataset should be loadable."""
        try:
            df = load_dataset(name)
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
        except FileNotFoundError:
            pytest.skip(f"Dataset file not generated yet: {name}")

    def test_load_unknown_dataset_raises(self) -> None:
        """load_dataset with unknown name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nonexistent_dataset_xyz")

    def test_get_dataset_info(self) -> None:
        """get_dataset_info should return metadata."""
        info = get_dataset_info("sp500_returns")
        assert "description" in info
        assert "category" in info
        assert info["category"] == "finance"

    def test_sp500_returns_shape(self) -> None:
        """sp500_returns should have expected columns."""
        try:
            df = load_dataset("sp500_returns")
            assert "returns" in df.columns
            assert len(df) >= 500
        except FileNotFoundError:
            pytest.skip("Dataset not generated")

    def test_linear_gaussian_has_state(self) -> None:
        """linear_gaussian should have state and observation columns."""
        try:
            df = load_dataset("linear_gaussian")
            assert "state" in df.columns
            assert "observation" in df.columns
        except FileNotFoundError:
            pytest.skip("Dataset not generated")


class TestExperimentWorkflow:
    """Test PFExperiment workflow."""

    def test_create_experiment(self) -> None:
        """PFExperiment should be creatable."""
        exp = PFExperiment(n_particles=100, seed=42, name="test")
        assert exp.name == "test"
        assert exp.n_particles == 100

    def test_add_model(self) -> None:
        """add_model should register a model."""
        exp = PFExperiment()
        model = SimpleNamespace(name="test_model")
        exp.add_model("test", model)
        assert "test" in exp._models

    def test_add_dataset(self) -> None:
        """add_dataset should register a dataset."""
        exp = PFExperiment()
        data = pd.DataFrame({"returns": np.random.randn(100)})
        exp.add_dataset("test", data)
        assert "test" in exp._datasets

    def test_add_metric(self) -> None:
        """add_metric should register a metric."""
        exp = PFExperiment()
        exp.add_metric("test", lambda r: 0.0)
        assert "test" in exp._metrics

    def test_compare_empty(self) -> None:
        """compare with no results should return empty DataFrame."""
        exp = PFExperiment()
        df = exp.compare()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_save_and_load(self, tmp_path: Path) -> None:
        """save and load should round-trip."""
        exp = PFExperiment(n_particles=200, seed=123, name="save_test")
        # Manually add a result
        from particlefilterbox.experiment import ExperimentResult
        exp._results.append(ExperimentResult(
            model_name="sv",
            dataset_name="sp500",
            metrics={"loglike": -150.0},
            elapsed_time=1.5,
        ))

        save_path = tmp_path / "exp.json"
        exp.save(save_path)
        assert save_path.exists()

        loaded = PFExperiment.load(save_path)
        assert loaded.name == "save_test"
        assert loaded.n_particles == 200
        assert len(loaded._results) == 1
        assert loaded._results[0].metrics["loglike"] == -150.0

    def test_report(self) -> None:
        """report should return a BaseReport."""
        exp = PFExperiment(name="report_test")
        from particlefilterbox.experiment import ExperimentResult
        exp._results.append(ExperimentResult(
            model_name="sv",
            dataset_name="sp500",
            metrics={"loglike": -150.0},
        ))
        report = exp.report()
        assert report.title == "Experiment: report_test"
        assert len(report.sections) >= 1
