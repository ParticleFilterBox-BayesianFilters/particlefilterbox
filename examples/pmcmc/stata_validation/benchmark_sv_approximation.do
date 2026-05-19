* ============================================================
* Stata Reference: SV parameter estimation (approximation)
* ============================================================
* NOTA: Stata nao possui PMCMC.
* Usa aproximacao linearizada do modelo SV via sspace.
* Resultados sao APROXIMADOS - nao devem ser tratados como ground truth.

clear all
set more off

import delimited "/home/guhaase/projetos/particlefilterbox/examples/bootstrap_sir/data/simulated_sv.csv", clear

rename y_obs y
gen time = t + 1
tsset time

* Aproximacao linearizada: log(y^2) = h + log(eps^2)
* log(eps^2) tem media -1.27 e variancia pi^2/2 ≈ 4.93
gen log_y2 = log(y^2 + 0.001)  // +0.001 para evitar log(0)

* Estimar via sspace com a aproximacao linearizada
* h_t = mu + phi*(h_{t-1} - mu) + sigma_h*eta_t
* log(y_t^2) ≈ h_t + const + error
sspace (h L.h, state noconstant) (log_y2 h, noconstant), covstate(diagonal) covobserved(diagonal)

display "============================================"
display "SV APROXIMADO via sspace (linearizacao)"
display "Log-likelihood (aproximada): " e(ll)
display "============================================"
display ""
display "AVISO: Estes resultados sao aproximacoes grosseiras."
display "A linearizacao do modelo SV introduz vieses significativos."
display "Use R (stochvol/pomp) ou particlefilterbox para resultados precisos."
display ""
display "LIMITACOES:"
display "- Stata nao tem PMMH"
display "- Stata nao tem Particle Gibbs"
display "- Stata nao tem PGAS"
display "- Apenas aproximacao linearizada do SV via sspace"

export delimited time y log_y2 using ///
    "/home/guhaase/projetos/particlefilterbox/examples/pmcmc/stata_validation/results_stata_sv_approx.csv", replace
