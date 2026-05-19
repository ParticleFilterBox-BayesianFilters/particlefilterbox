#!/usr/bin/env Rscript
# ===========================================================================
# Validacao: Bootstrap PF no modelo Linear-Gaussian
# Compara com Kalman filter (KFAS) e particlefilterbox
#
# Pacotes requeridos: pomp (>= 5.0), KFAS
# Instalar: install.packages(c("pomp", "KFAS"))
# ===========================================================================

suppressPackageStartupMessages({
  library(pomp)
  library(KFAS)
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

df <- read.csv(file.path(data_dir, "simulated_linear_gaussian.csv"))
cat(sprintf("Dados carregados: %d observacoes\n", nrow(df)))

# Parametros do modelo (identicos ao generate_data.py)
phi <- 0.95
sigma_x <- 0.5
sigma_y <- 1.0

# ---------------------------------------------------------------------------
# 2. Kalman filter benchmark via KFAS
# ---------------------------------------------------------------------------
kfas_model <- SSModel(
  df$y_obs ~ SSMcustom(
    Z = matrix(1),
    T = matrix(phi),
    R = matrix(1),
    Q = matrix(sigma_x^2),
    a1 = matrix(0),
    P1 = matrix(sigma_x^2 / (1 - phi^2))
  ),
  H = matrix(sigma_y^2)
)

kf_out <- KFS(kfas_model, filtering = "state", smoothing = "state")
kf_loglik_obj <- KFS(kfas_model, filtering = "state", smoothing = "none")
kalman_loglik <- kf_loglik_obj$logLik
kalman_rmse <- sqrt(mean((kf_out$att[, 1] - df$x_true)^2))

cat(sprintf("\n=== Kalman Filter (KFAS) ===\n"))
cat(sprintf("Log-likelihood: %.4f\n", kalman_loglik))
cat(sprintf("Filtered RMSE:  %.6f\n", kalman_rmse))

# ---------------------------------------------------------------------------
# 3. Bootstrap Particle Filter via pomp
# ---------------------------------------------------------------------------
lg_model <- pomp(
  data = data.frame(time = df$t, y = df$y_obs),
  times = "time",
  t0 = -1,
  rprocess = discrete_time(
    step.fun = function(x, ..., phi = 0.95, sigma_x = 0.5) {
      c(x = rnorm(1, mean = phi * x, sd = sigma_x))
    },
    delta.t = 1
  ),
  dmeasure = function(y, x, ..., sigma_y = 1.0, log) {
    dnorm(y, mean = x, sd = sigma_y, log = log)
  },
  rinit = function(..., phi = 0.95, sigma_x = 0.5) {
    c(x = rnorm(1, mean = 0, sd = sigma_x / sqrt(1 - phi^2)))
  },
  statenames = "x",
  paramnames = c("phi", "sigma_x", "sigma_y")
)

params_lg <- c(phi = phi, sigma_x = sigma_x, sigma_y = sigma_y)

cat(sprintf("\n=== Bootstrap PF (pomp) ===\n"))

results <- data.frame()
for (N in c(100, 500, 1000, 5000)) {
  pf <- pfilter(lg_model, Np = N, params = params_lg)
  loglik <- logLik(pf)
  ess_vals <- eff_sample_size(pf)
  ess_mean <- mean(ess_vals)

  results <- rbind(results, data.frame(
    N = N,
    loglik = loglik,
    ESS_mean = ess_mean,
    ESS_N_pct = ess_mean / N * 100,
    kalman_loglik = kalman_loglik
  ))
  cat(sprintf("N=%5d: logLik=%.2f, ESS_mean=%.1f (%.1f%%)\n",
              N, loglik, ess_mean, ess_mean / N * 100))
}

cat(sprintf("\nKalman logLik (referencia exata): %.4f\n", kalman_loglik))

# ---------------------------------------------------------------------------
# 4. Salvar resultados
# ---------------------------------------------------------------------------
write.csv(results, file.path(out_dir, "results_r_bootstrap.csv"), row.names = FALSE)
cat(sprintf("\nResultados salvos em: %s\n",
            file.path(out_dir, "results_r_bootstrap.csv")))
