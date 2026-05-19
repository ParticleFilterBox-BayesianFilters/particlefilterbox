* ============================================================
* Stata Reference: MLE benchmark for SMC parameter estimation
* ============================================================
* NOTA: Stata nao possui SMC samplers.
* sspace fornece MLE via Kalman filter para benchmark.

clear all
set more off

import delimited "/home/guhaase/projetos/particlefilterbox/examples/bootstrap_sir/data/simulated_linear_gaussian.csv", clear

rename y_obs y
gen time = t + 1
tsset time

* Estimar parametros via MLE (sspace)
sspace (x L.x, state noconstant) (y x, noconstant), covstate(diagonal) covobserved(diagonal)

* Exibir parametros estimados
matrix list e(b)
display "Log-likelihood (MLE): " e(ll)
display "AIC: " -2*e(ll) + 2*e(k)
display "BIC: " -2*e(ll) + e(k)*ln(e(N))

display ""
display "LIMITACOES:"
display "- Stata nao tem SMC samplers"
display "- Stata nao tem IBIS ou waste-free SMC"
display "- Apenas MLE via Kalman filter como benchmark"
display "- Para modelo SV, estimacao e aproximada (linearizacao)"

* Exportar estimativas
matrix b = e(b)
svmat b, names(param)
export delimited param* using ///
    "/home/guhaase/projetos/particlefilterbox/examples/smc/stata_validation/results_stata_mle.csv", replace
