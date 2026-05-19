* ============================================================
* Stata Reference: DSGE estimation via sspace (Kalman filter)
* ============================================================
* NOTA: Stata pode estimar o DSGE linearizado via sspace.
* Para jump-diffusion e SIR, nao ha suporte.

clear all
set more off

import delimited "/home/guhaase/projetos/particlefilterbox/examples/applications/data/treasury_yields.csv", clear

gen time = t + 1
tsset time

rename output_gap_obs y_x
rename inflation_obs y_pi
rename interest_rate_obs y_r

* Modelo DSGE linearizado como state-space
* x_t = A*x_{t-1} + B*e_t (transicao)
* y_t = C*x_t + D*u_t (observacao)
sspace (x L.x y_pi, state noconstant) ///
       (pi L.pi x, state noconstant) ///
       (r L.r pi x, state noconstant) ///
       (y_x x, noconstant) ///
       (y_pi pi, noconstant) ///
       (y_r r, noconstant), ///
       covstate(diagonal) covobserved(diagonal)

predict x_filt, state equation(x)
predict pi_filt, state equation(pi)
predict r_filt, state equation(r)

display "============================================"
display "DSGE Linearizado via sspace - Kalman Filter"
display "Log-likelihood: " e(ll)
display "============================================"
display ""
display "LIMITACOES:"
display "- Stata nao tem particle filter para DSGE nao-linear"
display "- Stata nao tem suporte para jump-diffusion"
display "- Stata nao tem suporte para modelo SIR epidemiologico"
display "- Apenas DSGE linearizado via Kalman filter"

export delimited time y_x y_pi y_r x_filt pi_filt r_filt using ///
    "/home/guhaase/projetos/particlefilterbox/examples/applications/stata_validation/results_stata_dsge.csv", replace
