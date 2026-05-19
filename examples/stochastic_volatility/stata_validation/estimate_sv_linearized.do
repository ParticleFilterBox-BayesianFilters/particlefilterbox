* ============================================================
* Stata: SV estimation via linearized approximation (sspace)
* ============================================================
* AVISO: Resultados APROXIMADOS. A linearizacao do modelo SV
* introduz vieses. Use particlefilterbox ou R para resultados precisos.
*
* Modelo SV original (nao linear, nao Gaussiano):
*     y_t      = exp(h_t / 2) * eps_t,    eps_t ~ N(0,1)
*     h_t      = mu + phi*(h_{t-1} - mu) + sigma_h * eta_t,  eta_t ~ N(0,1)
*
* Aproximacao linearizada (Harvey, Ruiz & Shephard 1994):
*     log(y_t^2) = h_t + log(eps_t^2)
*
* O termo log(eps_t^2) ~ log-chi^2(1) tem:
*     E[log(eps_t^2)]   = -1.2704
*     Var[log(eps_t^2)] = pi^2 / 2 = 4.9348
*
* Tratando log(eps_t^2) como Gaussiano (aproximacao QML), o modelo se
* torna linear-Gaussiano e e estimavel via state-space (sspace).
*
* VIESES CONHECIDOS:
*   1. phi subestimado (persistencia atenuada)
*   2. sigma_h superestimado (variancia espuria)
*   3. Leverage (rho) NAO ESTIMAVEL (correlacao perdida na linearizacao)
*   4. Outliers em y_t proximos de zero distorcem log(y_t^2)
* ============================================================

clear all
set more off

* --- Dados SP500 ---
import delimited "/home/guhaase/projetos/particlefilterbox/examples/stochastic_volatility/data/sp500_returns.csv", clear

gen time = _n
tsset time

* Aproximacao: log(y^2) = h + log(eps^2)
* Adiciona constante pequena para evitar log(0) em retornos nulos
gen log_y2 = log(returns^2 + 1e-8)

* Ajustar pela media de log(chi2(1)): -1.2704
* Apos o ajuste, log_y2_adj = h_t + (log(eps_t^2) - E[log(eps_t^2)])
* O ruido residual tem media zero e variancia pi^2/2 ~ 4.9348
gen log_y2_adj = log_y2 + 1.2704

* ============================================================
* Estimar via sspace
* ============================================================
* Equacao de estado:       h_t = phi * h_{t-1} + sigma_h * eta_t
* Equacao de observacao:   log_y2_adj = h_t + xi_t,  Var(xi_t) ~= 4.9348
*
* covstate(diagonal): variancia do choque eta_t estimada (sigma_h^2)
* covobserved(identity): variancia do erro de observacao fixada
* ============================================================
sspace (h L.h, state noconstant) (log_y2_adj h, noconstant), ///
    covstate(diagonal) covobserved(identity)

predict h_filtered, state equation(h)

* Volatilidade implicita: sigma_t = exp(h_t / 2)
gen sigma_filtered = exp(h_filtered / 2)

* Exibir resultados
display "============================================"
display "SV APROXIMADO (linearizacao) - SP500"
display "Log-likelihood (aproximada): " e(ll)
display "============================================"
display ""
display "VIESES CONHECIDOS DA APROXIMACAO:"
display "1. phi subestimado (menos persistente)"
display "2. sigma_h superestimado"
display "3. Leverage (rho) NAO ESTIMAVEL"
display "4. Outliers distorcem log(y^2)"
display ""
display "Para resultados PRECISOS use:"
display "  - particlefilterbox (Python, particle filter exato)"
display "  - R: stochvol (MCMC exato), bsvars (bayesiano)"

export delimited time returns h_filtered sigma_filtered using ///
    "/home/guhaase/projetos/particlefilterbox/examples/stochastic_volatility/stata_validation/results_stata_sv_linearized.csv", replace
