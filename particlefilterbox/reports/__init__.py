"""Reports module for particlefilterbox.

Provides automated report generation from particle filter and PMCMC results.
Supports HTML, LaTeX, and Markdown output formats.

Examples
--------
>>> from particlefilterbox.reports import PFReportTransformer, BaseReport
>>> transformer = PFReportTransformer()
>>> report = transformer.transform(results)
>>> report.to_html('pf_report.html')
>>> report.to_markdown('pf_report.md')
"""

from __future__ import annotations

from particlefilterbox.reports.base import BaseReport
from particlefilterbox.reports.dsge_report import DSGEReportTransformer
from particlefilterbox.reports.pf_report import PFReportTransformer
from particlefilterbox.reports.pmcmc_report import PMCMCReportTransformer
from particlefilterbox.reports.sv_report import SVReportTransformer

__all__ = [
    "BaseReport",
    "PFReportTransformer",
    "PMCMCReportTransformer",
    "SVReportTransformer",
    "DSGEReportTransformer",
]
