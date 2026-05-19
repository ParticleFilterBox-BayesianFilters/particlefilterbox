* ============================================================
* Stata Reference: Kalman Smoother for Particle Smoother benchmark
* ============================================================
* NOTA: Stata nao possui particle smoothers.
* sspace fornece Kalman smoother (exato) para modelos lineares-Gaussianos.
* Para o modelo SV, nao ha alternativa em Stata.

clear all
set more off

import delimited "/home/guhaase/projetos/particlefilterbox/examples/bootstrap_sir/data/simulated_linear_gaussian.csv", clear

rename y_obs y
rename x_true x_true_sim
gen time = t + 1
tsset time

sspace (x L.x, state noconstant) (y x, noconstant), covstate(diagonal) covobserved(diagonal)

* Kalman filter (one-step ahead)
predict x_filtered, state equation(x)

* Kalman smoother (usando todas as observacoes)
predict x_smoothed, sstate equation(x)

* RMSE comparacao
gen sq_err_filt = (x_filtered - x_true_sim)^2
gen sq_err_smooth = (x_smoothed - x_true_sim)^2

quietly summarize sq_err_filt
local rmse_filt = sqrt(r(mean))
quietly summarize sq_err_smooth
local rmse_smooth = sqrt(r(mean))

display "============================================"
display "Kalman Filter  RMSE: `rmse_filt'"
display "Kalman Smoother RMSE: `rmse_smooth'"
display "Smoothing gain: " round(`rmse_filt' - `rmse_smooth', 0.0001)
display "============================================"
display ""
display "LIMITACOES:"
display "- Stata nao tem particle smoothers (FFBSm, FFBSi, etc.)"
display "- Apenas Kalman smoother para modelo linear-Gaussiano"
display "- Modelo SV nao suportado para smoothing"

export delimited time x_true_sim y x_filtered x_smoothed using ///
    "/home/guhaase/projetos/particlefilterbox/examples/smoothers/stata_validation/results_stata_smoother.csv", replace
