"""Particle Filter report transformer.

Generates comprehensive reports from particle filter results including
summary statistics, filtered state plots, ESS diagnostics, weight
analysis, and log-likelihood information.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from particlefilterbox.reports.base import BaseReport


class PFReportTransformer:
    """Transform particle filter results into a report.

    Generates sections for:
    - Summary: particle count, time steps, log-likelihood
    - Filtered state: mean estimates with credible intervals
    - ESS diagnostics: effective sample size over time
    - Weight analysis: weight distribution statistics
    - Log-likelihood: cumulative log-likelihood

    Examples
    --------
    >>> transformer = PFReportTransformer()
    >>> report = transformer.transform(results)
    >>> report.to_html('pf_report.html')
    """

    def __init__(self, title: str = "Particle Filter Report") -> None:
        self.title = title

    def transform(self, results: Any) -> BaseReport:
        """Transform particle filter results into a BaseReport.

        Parameters
        ----------
        results : FilterResults
            Particle filter results object.

        Returns
        -------
        BaseReport
            Report with all sections populated.
        """
        report = BaseReport(title=self.title)

        self._add_summary_section(report, results)
        self._add_filtered_state_section(report, results)
        self._add_ess_section(report, results)
        self._add_weight_section(report, results)
        self._add_loglike_section(report, results)

        return report

    def _add_summary_section(self, report: BaseReport, results: Any) -> None:
        """Add summary section."""
        n_particles = getattr(results, "n_particles", "N/A")
        particles = getattr(results, "particles", None)
        log_likelihood = getattr(results, "log_likelihood", None)

        t_steps: int | str = "N/A"
        d: int | str = "N/A"
        if particles is not None:
            arr = np.asarray(particles)
            if arr.ndim == 3:
                t_steps, _, d = arr.shape

        content = (
            f"Particle filter analysis with {n_particles} particles "
            f"over {t_steps} time steps in a {d}-dimensional state space."
        )

        summary_table = {
            "headers": ["Property", "Value"],
            "rows": [
                ["Number of particles", str(n_particles)],
                ["Time steps", str(t_steps)],
                ["State dimension", str(d)],
                [
                    "Log-likelihood",
                    f"{log_likelihood:.4f}" if log_likelihood is not None else "N/A",
                ],
            ],
            "caption": "Filter Summary",
        }

        report.add_section(
            title="Summary",
            content=content,
            tables=[summary_table],
        )

    def _add_filtered_state_section(self, report: BaseReport, results: Any) -> None:
        """Add filtered state section."""
        filtered_mean = getattr(results, "filtered_mean", None)
        if filtered_mean is not None:
            fm = np.asarray(filtered_mean)
            n_states = fm.shape[1] if fm.ndim == 2 else 1

            content = (
                f"Filtered state estimates for {n_states} state variable(s). "
                "The filtered mean represents the weighted average of the particle "
                "approximation at each time step."
            )
        else:
            content = "Filtered state estimates computed from particle weights."

        report.add_section(title="Filtered State Estimates", content=content)

    def _add_ess_section(self, report: BaseReport, results: Any) -> None:
        """Add ESS diagnostics section."""
        ess = getattr(results, "ess", None)
        n_particles = getattr(results, "n_particles", None)

        if ess is not None:
            ess_arr = np.asarray(ess)
            content = (
                f"Effective Sample Size (ESS) ranges from {ess_arr.min():.1f} "
                f"to {ess_arr.max():.1f} with mean {ess_arr.mean():.1f}."
            )
            if n_particles is not None:
                pct_below = float(np.mean(ess_arr < n_particles / 2) * 100)
                content += f" ESS dropped below N/2 in {pct_below:.1f}% of time steps."

            ess_table = {
                "headers": ["Statistic", "Value"],
                "rows": [
                    ["Min ESS", f"{ess_arr.min():.1f}"],
                    ["Max ESS", f"{ess_arr.max():.1f}"],
                    ["Mean ESS", f"{ess_arr.mean():.1f}"],
                    ["Std ESS", f"{ess_arr.std():.1f}"],
                ],
                "caption": "ESS Statistics",
            }
            report.add_section(title="ESS Diagnostics", content=content, tables=[ess_table])
        else:
            report.add_section(
                title="ESS Diagnostics",
                content="ESS data not available in results.",
            )

    def _add_weight_section(self, report: BaseReport, results: Any) -> None:
        """Add weight analysis section."""
        weights = getattr(results, "weights", None)
        if weights is not None:
            w = np.asarray(weights)
            final_w = w[-1] if w.ndim == 2 else w
            final_w = final_w / final_w.sum()

            max_w = float(np.max(final_w))
            entropy = float(-np.sum(final_w[final_w > 0] * np.log(final_w[final_w > 0])))
            max_entropy = float(np.log(len(final_w)))

            content = (
                f"Weight analysis at the final time step. "
                f"Maximum weight: {max_w:.6f}. "
                f"Weight entropy: {entropy:.4f} (max: {max_entropy:.4f})."
            )

            weight_table = {
                "headers": ["Metric", "Value"],
                "rows": [
                    ["Max weight", f"{max_w:.6f}"],
                    ["Weight entropy", f"{entropy:.4f}"],
                    ["Max entropy (uniform)", f"{max_entropy:.4f}"],
                    [
                        "Entropy ratio",
                        f"{entropy / max_entropy:.4f}" if max_entropy > 0 else "N/A",
                    ],
                ],
                "caption": "Weight Analysis",
            }
            report.add_section(title="Weight Analysis", content=content, tables=[weight_table])
        else:
            report.add_section(title="Weight Analysis", content="Weight data not available.")

    def _add_loglike_section(self, report: BaseReport, results: Any) -> None:
        """Add log-likelihood section."""
        log_likelihood = getattr(results, "log_likelihood", None)
        log_likelihoods = getattr(results, "log_likelihoods", None)

        if log_likelihood is not None:
            content = f"Total log-likelihood estimate: {log_likelihood:.4f}."
            if log_likelihoods is not None:
                ll = np.asarray(log_likelihoods)
                content += (
                    f" Incremental log-likelihoods range from {ll.min():.4f} to {ll.max():.4f}."
                )
            report.add_section(title="Log-Likelihood", content=content)
        else:
            report.add_section(
                title="Log-Likelihood",
                content="Log-likelihood not available in results.",
            )
