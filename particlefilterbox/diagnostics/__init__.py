"""Diagnostics tools for particle filters and PMCMC.

This module provides monitoring, analysis, and diagnostic tools for
evaluating the quality and convergence of particle filtering methods.

Monitoring:
    - ESSMonitor: Real-time ESS monitoring with alerts
    - WeightAnalysis: Detailed weight distribution analysis (entropy, Gini, CV)

Convergence:
    - ConvergenceStudy: Convergence rate analysis (sqrt(N) verification)
    - DegeneracyDetector: Ancestral tree degeneracy detection

Model Comparison:
    - ModelComparison: Bayes factor-based model comparison

PMCMC:
    - PMCMCDiagnostics: Chain diagnostics (trace, ACF, ESS, R-hat, Geweke)
"""

from particlefilterbox.diagnostics.convergence import ConvergenceStudy
from particlefilterbox.diagnostics.degeneracy import DegeneracyDetector
from particlefilterbox.diagnostics.ess_monitor import ESSMonitor
from particlefilterbox.diagnostics.model_comparison import ModelComparison
from particlefilterbox.diagnostics.pmcmc_diagnostics import PMCMCDiagnostics
from particlefilterbox.diagnostics.weight_analysis import WeightAnalysis

__all__ = [
    # Monitoring
    "ESSMonitor",
    "WeightAnalysis",
    # Convergence
    "ConvergenceStudy",
    "DegeneracyDetector",
    # Model Comparison
    "ModelComparison",
    # PMCMC
    "PMCMCDiagnostics",
]
