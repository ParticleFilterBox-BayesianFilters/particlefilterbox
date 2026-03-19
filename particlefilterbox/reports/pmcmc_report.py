"""PMCMC report transformer.

Generates comprehensive reports from PMCMC (Particle MCMC) results including
posterior summaries, trace diagnostics, distributions, ACF, and convergence.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from particlefilterbox.reports.base import BaseReport


class PMCMCReportTransformer:
    """Transform PMCMC results into a report.

    Generates sections for:
    - Posterior summary: mean, std, credible intervals for each parameter
    - Trace diagnostics: convergence assessment
    - Posterior distributions: shape and modality
    - ACF: mixing efficiency
    - Convergence: R-hat and effective sample size

    Examples
    --------
    >>> transformer = PMCMCReportTransformer()
    >>> report = transformer.transform(results)
    >>> report.to_html('pmcmc_report.html')
    """

    def __init__(self, title: str = "PMCMC Analysis Report") -> None:
        self.title = title

    def transform(self, results: Any) -> BaseReport:
        """Transform PMCMC results into a BaseReport.

        Parameters
        ----------
        results : PMCMCResults
            PMCMC results object.

        Returns
        -------
        BaseReport
            Report with all sections populated.
        """
        report = BaseReport(title=self.title)

        self._add_posterior_summary(report, results)
        self._add_trace_diagnostics(report, results)
        self._add_distributions(report, results)
        self._add_acf_section(report, results)
        self._add_convergence(report, results)

        return report

    def _add_posterior_summary(self, report: BaseReport, results: Any) -> None:
        """Add posterior summary section."""
        chain = getattr(results, "chain", None)
        param_names = getattr(results, "param_names", None)

        if chain is not None:
            chain_arr = np.asarray(chain)
            n_iter, k_params = chain_arr.shape
            burn_in = n_iter // 4
            post = chain_arr[burn_in:]

            if param_names is None:
                param_names = [f"param_{i}" for i in range(k_params)]

            rows = []
            for j, name in enumerate(param_names):
                samples = post[:, j]
                mean = float(np.mean(samples))
                std = float(np.std(samples))
                q025 = float(np.percentile(samples, 2.5))
                q975 = float(np.percentile(samples, 97.5))
                rows.append(
                    [
                        name,
                        f"{mean:.4f}",
                        f"{std:.4f}",
                        f"{q025:.4f}",
                        f"{q975:.4f}",
                    ]
                )

            table = {
                "headers": ["Parameter", "Mean", "Std", "2.5%", "97.5%"],
                "rows": rows,
                "caption": "Posterior Summary (after 25% burn-in)",
            }

            content = (
                f"PMCMC analysis with {n_iter} iterations and {k_params} parameters. "
                f"Burn-in: first {burn_in} iterations discarded."
            )

            report.add_section(title="Posterior Summary", content=content, tables=[table])
        else:
            report.add_section(title="Posterior Summary", content="Chain data not available.")

    def _add_trace_diagnostics(self, report: BaseReport, results: Any) -> None:
        """Add trace diagnostics section."""
        chain = getattr(results, "chain", None)
        if chain is not None:
            chain_arr = np.asarray(chain)
            n_iter = chain_arr.shape[0]
            content = (
                f"Trace diagnostics for {n_iter} MCMC iterations. "
                "Visual inspection of trace plots is recommended to assess "
                "stationarity and mixing."
            )
        else:
            content = "Trace data not available."

        report.add_section(title="Trace Diagnostics", content=content)

    def _add_distributions(self, report: BaseReport, results: Any) -> None:
        """Add posterior distributions section."""
        content = (
            "Posterior distributions show the marginal distribution of each "
            "parameter after discarding burn-in. Compare with prior distributions "
            "to assess how much the data has informed the posterior."
        )
        report.add_section(title="Posterior Distributions", content=content)

    def _add_acf_section(self, report: BaseReport, results: Any) -> None:
        """Add autocorrelation section."""
        chain = getattr(results, "chain", None)
        if chain is not None:
            chain_arr = np.asarray(chain)
            param_names = getattr(results, "param_names", None)
            if param_names is None:
                param_names = [f"param_{i}" for i in range(chain_arr.shape[1])]

            rows = []
            for j, name in enumerate(param_names):
                samples = chain_arr[:, j]
                mean = float(np.mean(samples))
                var = float(np.var(samples))
                if var > 1e-12:
                    centered = samples - mean
                    acf1 = float(np.mean(centered[:-1] * centered[1:]) / var)
                else:
                    acf1 = 0.0
                rows.append([name, f"{acf1:.4f}"])

            table = {
                "headers": ["Parameter", "Lag-1 ACF"],
                "rows": rows,
                "caption": "Autocorrelation at Lag 1",
            }

            content = (
                "Autocorrelation function (ACF) measures serial correlation in the chain. "
                "High ACF at lag 1 indicates poor mixing. Consider thinning the chain."
            )
            report.add_section(title="Autocorrelation", content=content, tables=[table])
        else:
            report.add_section(title="Autocorrelation", content="ACF data not available.")

    def _add_convergence(self, report: BaseReport, results: Any) -> None:
        """Add convergence diagnostics section."""
        r_hat = getattr(results, "r_hat", None)
        acceptance_rate = getattr(results, "acceptance_rate", None)

        content_parts = ["Convergence diagnostics for the PMCMC chain."]

        rows: list[list[str]] = []
        if acceptance_rate is not None:
            content_parts.append(f"Overall acceptance rate: {acceptance_rate:.4f}.")
            rows.append(["Acceptance rate", f"{acceptance_rate:.4f}"])

        if r_hat is not None:
            if isinstance(r_hat, dict):
                for name, val in r_hat.items():
                    rows.append([f"R-hat ({name})", f"{val:.4f}"])
            else:
                rows.append(["R-hat", f"{r_hat:.4f}"])

        table = (
            {
                "headers": ["Diagnostic", "Value"],
                "rows": rows,
                "caption": "Convergence Diagnostics",
            }
            if rows
            else None
        )

        tables = [table] if table else []
        report.add_section(
            title="Convergence Diagnostics",
            content=" ".join(content_parts),
            tables=tables,
        )
