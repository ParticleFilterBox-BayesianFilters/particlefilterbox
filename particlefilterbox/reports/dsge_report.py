"""DSGE model report transformer.

Generates reports specialized for Dynamic Stochastic General Equilibrium
model results, including structural parameter estimates, impulse response
functions, and model comparison metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from particlefilterbox.reports.base import BaseReport


class DSGEReportTransformer:
    """Transform DSGE model results into a report.

    Generates sections for:
    - Model specification
    - Structural parameter estimates
    - Impulse response functions
    - Model fit and comparison

    Examples
    --------
    >>> transformer = DSGEReportTransformer()
    >>> report = transformer.transform(results)
    >>> report.to_html('dsge_report.html')
    """

    def __init__(self, title: str = "DSGE Model Report") -> None:
        self.title = title

    def transform(self, results: Any) -> BaseReport:
        """Transform DSGE results into a BaseReport.

        Parameters
        ----------
        results : Any
            DSGE model results object.

        Returns
        -------
        BaseReport
            Report with all sections populated.
        """
        report = BaseReport(title=self.title)

        self._add_model_specification(report, results)
        self._add_parameter_estimates(report, results)
        self._add_irf_section(report, results)
        self._add_model_fit(report, results)

        return report

    def _add_model_specification(self, report: BaseReport, results: Any) -> None:
        """Add model specification section."""
        model_name = getattr(results, "model_name", "DSGE")
        n_states = getattr(results, "n_states", "N/A")
        n_obs = getattr(results, "n_obs", "N/A")
        n_shocks = getattr(results, "n_shocks", "N/A")

        content = (
            f"DSGE model '{model_name}' with {n_states} state variables, "
            f"{n_obs} observable variables, and {n_shocks} structural shocks."
        )

        table = {
            "headers": ["Property", "Value"],
            "rows": [
                ["Model", str(model_name)],
                ["State variables", str(n_states)],
                ["Observables", str(n_obs)],
                ["Shocks", str(n_shocks)],
            ],
            "caption": "Model Specification",
        }

        report.add_section(title="Model Specification", content=content, tables=[table])

    def _add_parameter_estimates(self, report: BaseReport, results: Any) -> None:
        """Add structural parameter estimates section."""
        chain = getattr(results, "chain", None)
        param_names = getattr(results, "param_names", None)

        if chain is not None:
            chain_arr = np.asarray(chain)
            burn_in = chain_arr.shape[0] // 4
            post = chain_arr[burn_in:]

            if param_names is None:
                param_names = [f"theta_{i}" for i in range(post.shape[1])]

            rows = []
            for j, name in enumerate(param_names):
                if j >= post.shape[1]:
                    break
                samples = post[:, j]
                rows.append(
                    [
                        name,
                        f"{np.mean(samples):.4f}",
                        f"{np.std(samples):.4f}",
                        f"{np.percentile(samples, 2.5):.4f}",
                        f"{np.percentile(samples, 97.5):.4f}",
                    ]
                )

            table = {
                "headers": ["Parameter", "Mean", "Std", "2.5%", "97.5%"],
                "rows": rows,
                "caption": "Structural Parameter Estimates",
            }
            report.add_section(title="Structural Parameters", tables=[table])
        else:
            report.add_section(
                title="Structural Parameters",
                content="Parameter estimates not available.",
            )

    def _add_irf_section(self, report: BaseReport, results: Any) -> None:
        """Add impulse response functions section."""
        irf = getattr(results, "irf", None)
        if irf is not None:
            content = (
                "Impulse Response Functions (IRFs) show the dynamic response "
                "of endogenous variables to one-standard-deviation structural shocks."
            )
        else:
            content = "IRF data not available."
        report.add_section(title="Impulse Response Functions", content=content)

    def _add_model_fit(self, report: BaseReport, results: Any) -> None:
        """Add model fit section."""
        log_likelihood = getattr(results, "log_likelihood", None)
        log_evidence = getattr(results, "log_evidence", None)

        rows: list[list[str]] = []
        if log_likelihood is not None:
            rows.append(["Log-likelihood", f"{log_likelihood:.4f}"])
        if log_evidence is not None:
            rows.append(
                [
                    "Log-evidence (marginal likelihood)",
                    f"{log_evidence:.4f}",
                ]
            )

        content = "Model fit and comparison metrics."
        table = None
        if rows:
            table = {
                "headers": ["Metric", "Value"],
                "rows": rows,
                "caption": "Model Fit",
            }

        tables = [table] if table else []
        report.add_section(title="Model Fit", content=content, tables=tables)
