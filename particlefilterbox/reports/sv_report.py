"""Stochastic Volatility report transformer.

Generates reports specialized for stochastic volatility model results,
including volatility path estimates, parameter posteriors, and model fit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from particlefilterbox.reports.base import BaseReport


class SVReportTransformer:
    """Transform stochastic volatility results into a report.

    Generates sections specialized for SV model analysis including
    volatility path, parameter estimates, and model fit diagnostics.

    Examples
    --------
    >>> transformer = SVReportTransformer()
    >>> report = transformer.transform(results)
    >>> report.to_html('sv_report.html')
    """

    def __init__(self, title: str = "Stochastic Volatility Report") -> None:
        self.title = title

    def transform(self, results: Any) -> BaseReport:
        """Transform SV results into a BaseReport.

        Parameters
        ----------
        results : Any
            SV model results object.

        Returns
        -------
        BaseReport
            Report with all sections populated.
        """
        report = BaseReport(title=self.title)

        self._add_model_summary(report, results)
        self._add_volatility_path(report, results)
        self._add_parameter_estimates(report, results)
        self._add_model_fit(report, results)

        return report

    def _add_model_summary(self, report: BaseReport, results: Any) -> None:
        """Add model summary section."""
        content = (
            "Stochastic Volatility (SV) model analysis. "
            "The SV model captures time-varying volatility in financial returns "
            "via a latent log-volatility process: "
            "y_t = exp(h_t/2) * epsilon_t, "
            "h_t = mu + phi*(h_{t-1} - mu) + sigma_eta * eta_t."
        )
        report.add_section(title="Model Summary", content=content)

    def _add_volatility_path(self, report: BaseReport, results: Any) -> None:
        """Add volatility path section."""
        filtered_mean = getattr(results, "filtered_mean", None)
        if filtered_mean is not None:
            fm = np.asarray(filtered_mean)
            vol = fm[:, 0] if fm.ndim == 2 else fm

            content = (
                f"Estimated log-volatility path over {len(vol)} time steps. "
                f"Mean log-volatility: {np.mean(vol):.4f}, "
                f"Std: {np.std(vol):.4f}."
            )
        else:
            content = "Volatility path estimates not available."

        report.add_section(title="Volatility Path", content=content)

    def _add_parameter_estimates(self, report: BaseReport, results: Any) -> None:
        """Add parameter estimates section."""
        chain = getattr(results, "chain", None)
        param_names = getattr(results, "param_names", None)

        if chain is not None:
            chain_arr = np.asarray(chain)
            burn_in = chain_arr.shape[0] // 4
            post = chain_arr[burn_in:]

            if param_names is None:
                param_names = ["mu", "phi", "sigma_eta"][: post.shape[1]]

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
                "caption": "SV Parameter Estimates",
            }
            report.add_section(title="Parameter Estimates", tables=[table])
        else:
            report.add_section(
                title="Parameter Estimates",
                content="Parameter chain not available.",
            )

    def _add_model_fit(self, report: BaseReport, results: Any) -> None:
        """Add model fit section."""
        log_likelihood = getattr(results, "log_likelihood", None)
        content = "Model fit diagnostics."
        if log_likelihood is not None:
            content += f" Log-likelihood: {log_likelihood:.4f}."
        report.add_section(title="Model Fit", content=content)
