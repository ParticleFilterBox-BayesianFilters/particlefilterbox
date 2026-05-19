* ============================================================
* Validacao Stata: Linear-Gaussian SSM via sspace (Kalman Filter)
* ============================================================
* NOTA: Stata nao possui particle filters nativos.
* Este script usa sspace (Kalman filter) como benchmark analitico
* para o modelo linear-Gaussian apenas.

clear all
set more off

* Importar dados
import delimited "/home/guhaase/projetos/particlefilterbox/examples/bootstrap_sir/data/simulated_linear_gaussian.csv", clear

* Renomear variaveis
rename y_obs y
rename x_true x_true_sim
gen time = t + 1
tsset time

* Estimar modelo linear-Gaussian via sspace
* x_t = phi * x_{t-1} + e_t,  e_t ~ N(0, sigma_x^2)
* y_t = x_t + u_t,            u_t ~ N(0, sigma_y^2)
sspace (x L.x, state noconstant) (y x, noconstant), covstate(diagonal) covobserved(diagonal)

* Exibir resultados
estimates store lg_model
estat ic

* Predizer estados filtrados
predict x_filtered, state equation(x)
predict x_filtered_se, state equation(x) rmse

* Calcular RMSE vs true state
gen sq_error = (x_filtered - x_true_sim)^2
quietly summarize sq_error
local rmse = sqrt(r(mean))
display "Kalman RMSE (Stata sspace): `rmse'"

* Exibir log-likelihood
display "Log-likelihood: " e(ll)

* Exportar resultados
export delimited time x_true_sim y x_filtered x_filtered_se using ///
    "/home/guhaase/projetos/particlefilterbox/examples/bootstrap_sir/stata_validation/results_stata_kalman.csv", replace

display "Resultados exportados para results_stata_kalman.csv"
display "NOTA: Esta validacao cobre APENAS o modelo linear-Gaussian."
display "Para modelos nao-lineares (SV), Stata nao oferece alternativa ao PF."
