#!/usr/bin/env Rscript
# ===========================================================================
# Validacao: SIR (Sequential Importance Resampling) no modelo
# Stochastic Volatility (SV) usando pomp
#
# Modelo SV:
#   h_t = mu + phi*(h_{t-1} - mu) + sigma_h * w_t,  w_t ~ N(0,1)
#   y_t = exp(h_t/2) * v_t,                          v_t ~ N(0,1)
#
# Pacotes requeridos: pomp (>= 5.0)
# Instalar: install.packages("pomp")
# ===========================================================================

suppressPackageStartupMessages({
  library(pomp)
})

set.seed(42)

# ---------------------------------------------------------------------------
# 1. Carregar dados (mesmo dataset usado pelo particlefilterbox)
# ---------------------------------------------------------------------------
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) > 0) {
  script_dir <- dirname(normalizePath(script_path))
} else {
  script_dir <- getwd()
}
data_dir <- file.path(script_dir, "..", "data")
out_dir <- script_dir

df <- read.csv(file.path(data_dir, "simulated_sv.csv"))
cat(sprintf("Dados SV carregados: %d observacoes\n", nrow(df)))

# Parametros do modelo (identicos ao generate_data.py)
mu <- -1.0
phi_sv <- 0.97
sigma_h <- 0.15

# ---------------------------------------------------------------------------
# 2. Definir modelo SV no pomp
# ---------------------------------------------------------------------------
sv_model <- pomp(
  data = data.frame(time = df$t, y = df$y_obs),
  times = "time",
  t0 = -1,
  rprocess = discrete_time(
    step.fun = function(h, ..., mu = -1.0, phi = 0.97, sigma_h = 0.15) {
      c(h = rnorm(1,
                   mean = mu + phi * (h - mu),
                   sd = sigma_h))
    },
    delta.t = 1
  ),
  dmeasure = function(y, h, ..., log) {
    # y_t ~ N(0, exp(h_t))  =>  pdf = dnorm(y, 0, exp(h/2))
    dnorm(y, mean = 0, sd = exp(h / 2), log = log)
  },
  rinit = function(..., mu = -1.0, phi = 0.97, sigma_h = 0.15) {
    c(h = rnorm(1,
                mean = mu,
                sd = sigma_h / sqrt(1 - phi^2)))
  },
  statenames = "h",
  paramnames = c("mu", "phi", "sigma_h")
)

params_sv <- c(mu = mu, phi = phi_sv, sigma_h = sigma_h)

# ---------------------------------------------------------------------------
# 3. Rodar particle filter (SIR) com diferentes N
# ---------------------------------------------------------------------------
cat(sprintf("\n=== SIR Particle Filter (pomp) - Modelo SV ===\n"))

results <- data.frame()
for (N in c(100, 500, 1000, 5000)) {
  pf <- pfilter(sv_model, Np = N, params = params_sv)
  loglik <- logLik(pf)
  ess_vals <- eff_sample_size(pf)
  ess_mean <- mean(ess_vals)

  results <- rbind(results, data.frame(
    N = N,
    loglik = loglik,
    ESS_mean = ess_mean,
    ESS_N_pct = ess_mean / N * 100
  ))
  cat(sprintf("N=%5d: logLik=%.2f, ESS_mean=%.1f (%.1f%%)\n",
              N, loglik, ess_mean, ess_mean / N * 100))
}

# ---------------------------------------------------------------------------
# 4. Salvar resultados
# ---------------------------------------------------------------------------
write.csv(results, file.path(out_dir, "results_r_sir.csv"), row.names = FALSE)
cat(sprintf("\nResultados salvos em: %s\n",
            file.path(out_dir, "results_r_sir.csv")))
