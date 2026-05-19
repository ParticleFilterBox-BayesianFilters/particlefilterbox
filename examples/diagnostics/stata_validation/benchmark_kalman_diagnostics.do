* ============================================================
* Stata Reference: Kalman Filter diagnostics benchmark
* ============================================================
* NOTA: Stata nao possui particle filter diagnostics.
* Fornece diagnosticos do Kalman filter como benchmark para o
* modelo linear-Gaussian.
*
* Model:
*   x_t = x_{t-1} + w_t,   w_t ~ N(0, Q)
*   y_t = x_t     + v_t,   v_t ~ N(0, R)
*
* Diagnosticos computados:
*   - Prediction errors (innovations):   e_t = y_t - E[y_t | y_{1:t-1}]
*   - Innovation mean/std:                deve ser aprox. N(0, S_t)
*   - Normality test (skewness/kurtosis): sktest
*   - Log-likelihood (via Kalman recursion)
*
* Comparacao com particle filter (via particlefilterbox):
*   No caso linear-Gaussian, o particle filter deve reproduzir
*   (em distribuicao) os diagnosticos produzidos aqui pelo
*   Kalman filter, a menos de variabilidade Monte Carlo.
* ============================================================

clear all
set more off

* ------------------------------------------------------------
* 1. Carregar dados linear-Gaussianos simulados
* ------------------------------------------------------------
import delimited ///
    "/home/guhaase/projetos/particlefilterbox/examples/bootstrap_sir/data/simulated_linear_gaussian.csv", ///
    clear

rename y_obs y
gen time = t + 1
tsset time

* ------------------------------------------------------------
* 2. Ajustar modelo state-space via sspace
* ------------------------------------------------------------
sspace (x L.x, state noconstant) ///
       (y x,   noconstant),      ///
       covstate(diagonal) covobserved(diagonal)

* ------------------------------------------------------------
* 3. Prediction errors (innovations)
* ------------------------------------------------------------
predict y_pred, dynamic
gen innovation = y - y_pred

* ------------------------------------------------------------
* 4. Diagnosticos basicos das innovations
* ------------------------------------------------------------
summarize innovation
display "============================================"
display "Innovation mean: " r(mean)
display "Innovation std:  " r(sd)
display "Innovation min:  " r(min)
display "Innovation max:  " r(max)
display "============================================"

* Normalidade das innovations (Kalman filter bem especificado
* implica innovations Gaussianas com media zero)
sktest innovation

* Autocorrelacao das innovations (sob o modelo correto, devem
* ser serialmente nao-correlacionadas - white noise)
corrgram innovation, lags(10)

* ------------------------------------------------------------
* 5. Log-likelihood da Kalman recursion
* ------------------------------------------------------------
display "============================================"
display "Kalman Filter Diagnostics - Linear-Gaussian"
display "Log-likelihood: " e(ll)
display "N obs:          " e(N)
display "============================================"

* ------------------------------------------------------------
* 6. Limitacoes do Stata para diagnosticos de particle filter
* ------------------------------------------------------------
display ""
display "LIMITACOES:"
display "- Stata nao tem ESS (conceito de particle filter)"
display "- Stata nao tem weight diagnostics"
display "- Stata nao tem convergence diagnostics para PF"
display "- Stata nao tem degeneracy/resampling diagnostics"
display "- Apenas diagnosticos do Kalman filter (innovations)"
display ""
display "Para diagnosticos completos de particle filter use"
display "particlefilterbox.diagnostics (Python) ou pomp (R)."

* ------------------------------------------------------------
* 7. Exportar innovations para comparacao com Python
* ------------------------------------------------------------
export delimited time y innovation using ///
    "/home/guhaase/projetos/particlefilterbox/examples/diagnostics/stata_validation/results_stata_diagnostics.csv", ///
    replace

display ""
display "Results exported to results_stata_diagnostics.csv"
