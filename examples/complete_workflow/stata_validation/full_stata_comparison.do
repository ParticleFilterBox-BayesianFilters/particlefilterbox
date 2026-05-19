* ============================================================
* Stata: Full comparison - Linear-Gaussian and SV approximation
* ============================================================
* Consolida todos os benchmarks Stata para particlefilterbox.
*
* Uso:
*   stata -b do full_stata_comparison.do
*
* Saidas:
*   - results_stata_linear_gaussian.csv
*   - results_stata_sv_approx.csv (se sspace convergir)
* ============================================================

clear all
set more off

display "============================================"
display "PARTICLEFILTERBOX - Stata Validation Summary"
display "============================================"
display ""

* --- Benchmark 1: Linear-Gaussian Kalman Filter ---
display "--- Benchmark 1: Linear-Gaussian (Kalman, exato) ---"

import delimited "/home/guhaase/projetos/particlefilterbox/examples/bootstrap_sir/data/simulated_linear_gaussian.csv", clear

rename y_obs y
gen time = t + 1
tsset time

sspace (x L.x, state noconstant) (y x, noconstant), covstate(diagonal) covobserved(diagonal)
predict x_filt, state equation(x)
predict x_smooth, sstate equation(x)

rename x_true x_true_sim
gen rmse_filt = (x_filt - x_true_sim)^2
gen rmse_smooth = (x_smooth - x_true_sim)^2
quietly summarize rmse_filt
local rmse_f = sqrt(r(mean))
quietly summarize rmse_smooth
local rmse_s = sqrt(r(mean))

display "Kalman Filter RMSE: `rmse_f'"
display "Kalman Smoother RMSE: `rmse_s'"
display "Log-likelihood: " e(ll)
display ""

export delimited time x_true_sim y x_filt x_smooth using ///
    "/home/guhaase/projetos/particlefilterbox/examples/complete_workflow/stata_validation/results_stata_linear_gaussian.csv", replace

* --- Benchmark 2: SV Approximation ---
display "--- Benchmark 2: SV Approximation (linearizado) ---"

clear
import delimited "/home/guhaase/projetos/particlefilterbox/examples/stochastic_volatility/data/sp500_returns.csv", clear

gen time = _n
tsset time
gen log_y2 = log(returns^2 + 1e-8)
gen log_y2_adj = log_y2 + 1.2704

capture sspace (h L.h, state noconstant) (log_y2_adj h, noconstant), covstate(diagonal) covobserved(identity)
if _rc == 0 {
    display "SV Approx Log-likelihood: " e(ll)
    capture predict h_filt, state equation(h)
    capture predict h_smooth, sstate equation(h)
    capture export delimited time returns log_y2_adj h_filt h_smooth using ///
        "/home/guhaase/projetos/particlefilterbox/examples/complete_workflow/stata_validation/results_stata_sv_approx.csv", replace
}
else {
    display "NOTA: sspace falhou na estimacao SV aproximada (rc=" _rc ")"
}

display ""
display "============================================"
display "RESUMO DE LIMITACOES DO STATA:"
display "============================================"
display "Suportado:"
display "  - Kalman filter/smoother (linear-Gaussian): EXATO"
display "  - SV linearizado (sspace): APROXIMADO"
display ""
display "NAO suportado:"
display "  - Bootstrap PF, SIR, Auxiliary PF, RBPF"
display "  - FFBSm, FFBSi, two-filter, fixed-lag smoothers"
display "  - SMC sampler, SMC^2, IBIS, waste-free SMC"
display "  - PMMH, Particle Gibbs, PGAS"
display "  - SV com leverage, SV com jumps, factor SV"
display "  - Jump-diffusion (Merton, Kou, Bates)"
display "  - SIR epidemiologico"
display "  - ESS, weight diagnostics, convergence diagnostics"
display "============================================"
