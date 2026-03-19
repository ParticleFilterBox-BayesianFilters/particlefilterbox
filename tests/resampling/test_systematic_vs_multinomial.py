"""Critical test: systematic should have lower variance than multinomial."""

from __future__ import annotations

import numpy as np

from particlefilterbox.resampling.multinomial import multinomial_resample
from particlefilterbox.resampling.systematic import systematic_resample


class TestSystematicVsMultinomial:
    def test_systematic_lower_variance(self) -> None:
        """Repeat 1000x: variance of copy counts should be lower for systematic."""
        n = 100
        rng = np.random.default_rng(42)
        w = rng.dirichlet(np.ones(n))

        n_reps = 1000
        counts_sys = np.zeros((n_reps, n))
        counts_multi = np.zeros((n_reps, n))

        for rep in range(n_reps):
            idx_sys = systematic_resample(w, rng=np.random.default_rng(rep))
            idx_multi = multinomial_resample(w, rng=np.random.default_rng(rep + 100000))
            counts_sys[rep] = np.bincount(idx_sys, minlength=n)
            counts_multi[rep] = np.bincount(idx_multi, minlength=n)

        var_systematic = np.var(counts_sys, axis=0)
        var_multinomial = np.var(counts_multi, axis=0)

        # Systematic should have lower total variance
        total_var_sys = np.sum(var_systematic)
        total_var_multi = np.sum(var_multinomial)
        assert total_var_sys < total_var_multi, (
            f"Systematic variance ({total_var_sys:.2f}) should be less than "
            f"multinomial ({total_var_multi:.2f})"
        )
