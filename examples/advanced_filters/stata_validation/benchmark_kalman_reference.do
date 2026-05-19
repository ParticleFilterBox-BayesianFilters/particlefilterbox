* ============================================================
* Stata Reference: Kalman Filter benchmark for Advanced Filters
* ============================================================
* NOTA: Stata nao possui Auxiliary PF, RBPF, Unscented PF ou
* Regularized PF. Este script fornece apenas o benchmark Kalman
* via sspace para comparacao com o RBPF no modelo linear-Gaussian.

clear all
set more off

import delimited "/home/guhaase/projetos/particlefilterbox/examples/bootstrap_sir/data/simulated_linear_gaussian.csv", clear

rename y_obs y
rename x_true x_true_sim
gen time = t + 1
tsset time

* Estimar via sspace (Kalman filter)
sspace (x L.x, state noconstant) (y x, noconstant), covstate(diagonal) covobserved(diagonal)

predict x_filtered, state equation(x)

gen sq_error = (x_filtered - x_true_sim)^2
quietly summarize sq_error
local rmse = sqrt(r(mean))

display "============================================"
display "Kalman Filter (sspace) - Linear-Gaussian SSM"
display "RMSE: `rmse'"
display "Log-likelihood: " e(ll)
display "============================================"
display ""
display "LIMITACOES:"
display "- Stata nao tem Auxiliary PF"
display "- Stata nao tem RBPF"
display "- Stata nao tem Unscented PF"
display "- Stata nao tem Regularized PF"
display "- Apenas benchmark Kalman para modelo linear-Gaussiano"

export delimited time x_true_sim y x_filtered using ///
    "/home/guhaase/projetos/particlefilterbox/examples/advanced_filters/stata_validation/results_stata_kalman_ref.csv", replace
