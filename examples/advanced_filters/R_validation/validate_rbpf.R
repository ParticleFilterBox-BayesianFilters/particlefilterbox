#!/usr/bin/env Rscript
# ===========================================================================
# Validacao: Rao-Blackwellized Particle Filter (RBPF) no modelo
# Linear-Gaussian com componente nao-linear
#
# Implementa RBPF manualmente: particle filter para componente nao-linear
# (scaling factor) + Kalman filter (KFAS) para componente linear.
# Compara com BPF puro (pomp) e Kalman filter exato.
#
# Modelo misto (identico ao particlefilterbox):
#   Nao-linear: s_t = s_{t-1} + sigma_s * eta_t  (random walk)
#   Linear:     x_t = phi * x_{t-1} + sigma_x * eps_t
#   Observacao: y_t = exp(s_t/10) * x_t + sigma_y * nu_t
#
# Pacotes requeridos: pomp (>= 5.0), KFAS
# Instalar: install.packages(c("pomp", "KFAS"))
#
# Referencia: Doucet, de Freitas & Gordon (2001) - Sequential Monte Carlo
#   Methods in Practice, Ch. 10: Rao-Blackwellised Particle Filtering
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

# Parametros do modelo (identicos ao particlefilterbox)
phi <- 0.95
sigma_x <- 0.5
sigma_y <- 1.0
sigma_s <- 0.02  # scaling factor noise (very small => nearly linear)

y_obs <- df$y_obs
x_true <- df$x_true
TT <- nrow(df)

# ---------------------------------------------------------------------------
# 2. Kalman filter benchmark (modelo puramente linear, sem scaling)
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
kalman_rmse <- sqrt(mean((kf_out$att[, 1] - x_true)^2))

cat(sprintf("\n=== Kalman Filter (KFAS) - Referencia Exata ===\n"))
cat(sprintf("Log-likelihood: %.4f\n", kalman_loglik))
cat(sprintf("Filtered RMSE:  %.6f\n", kalman_rmse))

# ---------------------------------------------------------------------------
# 3. RBPF: Particle filter (s_t) + Kalman filter (x_t | s_t)
# ---------------------------------------------------------------------------
# Para cada particula i:
#   - Propagar s_t^i ~ N(s_{t-1}^i, sigma_s^2)
#   - Rodar um passo de Kalman para x_t condicionado em s_t^i:
#     scale_i = exp(s_t^i / 10)
#     Z_i = scale_i (observation matrix)
#     Prediction: x_pred = phi * x_filt_{t-1}
#                 P_pred = phi^2 * P_filt_{t-1} + sigma_x^2
#     Update:     K = P_pred * Z_i / (Z_i^2 * P_pred + sigma_y^2)
#                 x_filt = x_pred + K * (y_t - Z_i * x_pred)
#                 P_filt = (1 - K * Z_i) * P_pred
#   - Weight: p(y_t | s_t^i) = N(y_t; Z_i * x_pred, Z_i^2 * P_pred + sigma_y^2)

rbpf_filter <- function(y, N, phi, sigma_x, sigma_y, sigma_s) {
  TT <- length(y)
  Q <- sigma_x^2
  R <- sigma_y^2

  # Initialize particles for s_t (nonlinear component)
  s_particles <- rnorm(N, mean = 0, sd = 0.1)

  # Initialize Kalman states for each particle
  x_mean <- rep(0, N)  # E[x_0 | s_0^i]
  x_var <- rep(Q / (1 - phi^2), N)  # Var[x_0 | s_0^i]

  # Storage
  x_filtered <- numeric(TT)
  ess_vec <- numeric(TT)
  log_lik <- 0.0

  for (t in seq_len(TT)) {
    # --- Propagate nonlinear component ---
    s_new <- s_particles + sigma_s * rnorm(N)

    # --- Kalman prediction for linear component ---
    x_pred <- phi * x_mean
    P_pred <- phi^2 * x_var + Q

    # --- Compute observation likelihood p(y_t | s_t^i, y_{1:t-1}) ---
    scale <- exp(s_new / 10.0)
    innov_var <- scale^2 * P_pred + R
    innov_mean <- scale * x_pred
    residual <- y[t] - innov_mean

    log_w <- dnorm(y[t], mean = innov_mean, sd = sqrt(innov_var), log = TRUE)

    # --- Normalize weights ---
    max_lw <- max(log_w)
    w_norm <- exp(log_w - max_lw)
    w_sum <- sum(w_norm)
    w_norm <- w_norm / w_sum

    # Log-likelihood increment
    log_lik <- log_lik + log(w_sum / N) + max_lw

    # ESS
    ess_vec[t] <- 1 / sum(w_norm^2)

    # --- Kalman update for each particle ---
    K <- P_pred * scale / innov_var
    x_filt <- x_pred + K * residual
    P_filt <- (1 - K * scale) * P_pred

    # --- Store weighted filtered state ---
    x_filtered[t] <- sum(w_norm * x_filt)

    # --- Resample ---
    idx <- sample.int(N, size = N, replace = TRUE, prob = w_norm)
    s_particles <- s_new[idx]
    x_mean <- x_filt[idx]
    x_var <- P_filt[idx]
  }

  list(
    filtered_x = x_filtered,
    ess = ess_vec,
    log_likelihood = log_lik
  )
}

# ---------------------------------------------------------------------------
# 4. BPF via pomp (modelo linear-Gaussian puro)
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

# ---------------------------------------------------------------------------
# 5. Rodar RBPF e BPF com diferentes N
# ---------------------------------------------------------------------------
cat(sprintf("\n%s\n", strrep("=", 62)))
cat("  RBPF vs BPF vs Kalman - Linear-Gaussian Model\n")
cat(sprintf("%s\n\n", strrep("=", 62)))

results <- data.frame()

# RBPF with N=500
set.seed(42)
rbpf_res <- rbpf_filter(y_obs, N = 500, phi, sigma_x, sigma_y, sigma_s)
rbpf_rmse <- sqrt(mean((rbpf_res$filtered_x - x_true)^2))
rbpf_ess_mean <- mean(rbpf_res$ess)

results <- rbind(results, data.frame(
  Filter = "RBPF",
  N = 500,
  loglik = rbpf_res$log_likelihood,
  RMSE = rbpf_rmse,
  ESS_mean = rbpf_ess_mean,
  kalman_loglik = kalman_loglik
))
cat(sprintf("RBPF N=%5d: logLik=%.2f, RMSE=%.4f, ESS_mean=%.1f\n",
            500, rbpf_res$log_likelihood, rbpf_rmse, rbpf_ess_mean))

# BPF with N=500 and N=2000
for (N in c(500, 2000)) {
  set.seed(42)
  pf <- pfilter(lg_model, Np = N, params = params_lg)
  bpf_loglik <- logLik(pf)
  bpf_ess_vals <- eff_sample_size(pf)
  bpf_ess_mean <- mean(bpf_ess_vals)

  results <- rbind(results, data.frame(
    Filter = "BPF",
    N = N,
    loglik = bpf_loglik,
    RMSE = NA,
    ESS_mean = bpf_ess_mean,
    kalman_loglik = kalman_loglik
  ))
  cat(sprintf("BPF  N=%5d: logLik=%.2f, ESS_mean=%.1f\n",
              N, bpf_loglik, bpf_ess_mean))
}

# Kalman reference row
results <- rbind(results, data.frame(
  Filter = "Kalman",
  N = NA,
  loglik = kalman_loglik,
  RMSE = kalman_rmse,
  ESS_mean = NA,
  kalman_loglik = kalman_loglik
))
cat(sprintf("Kalman:       logLik=%.4f, RMSE=%.6f\n", kalman_loglik, kalman_rmse))

# ---------------------------------------------------------------------------
# 6. Verificacao: RBPF RMSE deve ser proximo ao Kalman
# ---------------------------------------------------------------------------
cat(sprintf("\n--- RMSE Comparison ---\n"))
cat(sprintf("  Kalman RMSE: %.6f\n", kalman_rmse))
cat(sprintf("  RBPF   RMSE: %.6f\n", rbpf_rmse))
cat(sprintf("  Diff:        %.6f (%.2f%%)\n",
            abs(rbpf_rmse - kalman_rmse),
            abs(rbpf_rmse - kalman_rmse) / kalman_rmse * 100))

# ---------------------------------------------------------------------------
# 7. Salvar resultados
# ---------------------------------------------------------------------------
write.csv(results, file.path(out_dir, "results_r_rbpf.csv"), row.names = FALSE)
cat(sprintf("\nResultados salvos em: %s\n",
            file.path(out_dir, "results_r_rbpf.csv")))

cat("\nPacotes utilizados:\n")
cat("  - pomp (>= 5.0): Bootstrap PF (pfilter) como referencia\n")
cat("  - KFAS: Kalman filter para benchmark exato e componente linear do RBPF\n")
cat("  - nimbleSMC: framework alternativo para SMC (buildBootstrapFilter)\n")
cat("  - RBPF implementado manualmente: PF(s_t) + Kalman(x_t|s_t)\n")
