#!/usr/bin/env Rscript
# ===========================================================================
# Validacao: Particle Smoothing (FFBS) no modelo Stochastic Volatility
#
# Implementa Forward Filtering Backward Smoothing (FFBS) usando pomp para
# o forward pass (particle filter) e backward sampling manual.
# Compara filtered vs smoothed estimates e salva resultados para
# comparacao cruzada com particlefilterbox (Python).
#
# Modelo SV:
#   h_t = mu + phi*(h_{t-1} - mu) + sigma_h * w_t,  w_t ~ N(0,1)
#   y_t = exp(h_t/2) * v_t,                          v_t ~ N(0,1)
#
# Pacotes requeridos: pomp (>= 5.0)
# Instalar: install.packages("pomp")
#
# Referencia: Godsill, Doucet & West (2004) - Monte Carlo Smoothing for
#   Nonlinear Time Series
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

# Parametros do modelo SV (identicos ao particlefilterbox)
mu <- -1.0
phi_sv <- 0.97
sigma_h <- 0.15

# Usar primeiras T observacoes (consistente com Python scripts)
TT <- 100
y_obs <- df$y_obs[1:TT]
h_true <- df$h_true[1:TT]
t_idx <- df$t[1:TT]

cat(sprintf("Usando primeiras %d observacoes para smoothing\n", TT))

# ---------------------------------------------------------------------------
# 2. Definir modelo SV no pomp
# ---------------------------------------------------------------------------
sv_pomp <- pomp(
  data = data.frame(time = t_idx, y = y_obs),
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
# 3. Forward pass: Bootstrap Particle Filter via pomp (save states)
# ---------------------------------------------------------------------------
N <- 2000  # numero de particulas (consistente com Python)

cat(sprintf("\n=== Forward Filtering (pomp pfilter, N=%d) ===\n", N))

# Rodar pfilter salvando estados para backward pass
pf <- pfilter(sv_pomp, Np = N, params = params_sv,
              save.states = "filter")

pf_loglik <- logLik(pf)
pf_ess <- eff_sample_size(pf)
cat(sprintf("Log-likelihood: %.4f\n", pf_loglik))
cat(sprintf("ESS medio: %.1f (%.1f%%)\n", mean(pf_ess), mean(pf_ess) / N * 100))

# ---------------------------------------------------------------------------
# 4. Manual BPF for full particle/weight access (needed for FFBS)
# ---------------------------------------------------------------------------
# pomp nao expoe pesos individuais facilmente, entao implementamos BPF
# manualmente para ter acesso completo a particulas e pesos para FFBS.

cat(sprintf("\n=== Manual BPF + FFBS (N=%d) ===\n", N))

log_obs_density <- function(y, h) {
  # y ~ N(0, exp(h)), so log p(y|h) = -0.5*log(2*pi) - h/2 - 0.5*y^2*exp(-h)
  -0.5 * log(2 * pi) - h / 2 - 0.5 * y^2 * exp(-h)
}

# Forward filter: armazenar particulas e pesos normalizados
particles <- matrix(NA, nrow = N, ncol = TT)
weights <- matrix(NA, nrow = N, ncol = TT)
filtered_means <- numeric(TT)
filtered_vars <- numeric(TT)
log_lik <- 0.0

# Inicializacao: h_0 ~ N(mu, sigma_h^2/(1 - phi^2))
sd_stat <- sigma_h / sqrt(1 - phi_sv^2)
h_curr <- rnorm(N, mean = mu, sd = sd_stat)

for (t in seq_len(TT)) {
  if (t > 1) {
    # Propagate: h_t ~ N(mu + phi*(h_{t-1} - mu), sigma_h^2)
    h_curr <- mu + phi_sv * (h_curr - mu) + sigma_h * rnorm(N)
  }

  # Compute log-weights
  log_w <- log_obs_density(y_obs[t], h_curr)

  # Log-likelihood increment
  max_lw <- max(log_w)
  log_lik <- log_lik + max_lw + log(mean(exp(log_w - max_lw)))

  # Normalize weights
  w_norm <- exp(log_w - max_lw)
  w_norm <- w_norm / sum(w_norm)

  # Store
  particles[, t] <- h_curr
  weights[, t] <- w_norm

  # Filtered estimates
  filtered_means[t] <- sum(w_norm * h_curr)
  filtered_vars[t] <- sum(w_norm * (h_curr - filtered_means[t])^2)

  # Resample (multinomial)
  idx <- sample.int(N, size = N, replace = TRUE, prob = w_norm)
  h_curr <- h_curr[idx]
}

cat(sprintf("Manual BPF log-likelihood: %.4f\n", log_lik))

# ---------------------------------------------------------------------------
# 5. FFBS - Forward Filtering Backward Smoothing
# ---------------------------------------------------------------------------
# Algoritmo FFBSm (Godsill, Doucet & West, 2004):
#   Para t = T-1, ..., 0:
#     w_{t|T}^i propto w_t^i * p(h_{t+1}^j | h_t^i)
#     Resample backward trajectory from w_{t|T}^i

M <- 500  # numero de trajetorias backward (consistente com Python)
smoothed_trajectories <- matrix(NA, nrow = M, ncol = TT)

cat(sprintf("FFBS backward sampling: M=%d trajetorias\n", M))

for (m in seq_len(M)) {
  # Iniciar no tempo T: amostrar de p(h_T | y_{1:T})
  idx_T <- sample.int(N, size = 1, prob = weights[, TT])
  smoothed_trajectories[m, TT] <- particles[idx_T, TT]

  # Backward pass
  for (t in (TT - 1):1) {
    h_next <- smoothed_trajectories[m, t + 1]

    # Backward weights: w_t^i * p(h_{t+1} | h_t^i)
    # p(h_{t+1} | h_t^i) = N(h_{t+1}; mu + phi*(h_t^i - mu), sigma_h^2)
    h_pred_mean <- mu + phi_sv * (particles[, t] - mu)
    log_trans <- dnorm(h_next, mean = h_pred_mean, sd = sigma_h, log = TRUE)
    log_bw <- log(weights[, t]) + log_trans

    # Normalize
    max_lbw <- max(log_bw)
    bw_norm <- exp(log_bw - max_lbw)
    bw_norm <- bw_norm / sum(bw_norm)

    idx_t <- sample.int(N, size = 1, prob = bw_norm)
    smoothed_trajectories[m, t] <- particles[idx_t, t]
  }
}

# Compute smoothed estimates
smoothed_means <- colMeans(smoothed_trajectories)
smoothed_vars <- apply(smoothed_trajectories, 2, var)

# ---------------------------------------------------------------------------
# 6. Metricas de avaliacao
# ---------------------------------------------------------------------------
cat(sprintf("\n=== Resultados ===\n"))

filter_rmse <- sqrt(mean((filtered_means - h_true)^2))
smooth_rmse <- sqrt(mean((smoothed_means - h_true)^2))
filter_mae <- mean(abs(filtered_means - h_true))
smooth_mae <- mean(abs(smoothed_means - h_true))

cat(sprintf("Filtered RMSE:  %.6f\n", filter_rmse))
cat(sprintf("Smoothed RMSE:  %.6f\n", smooth_rmse))
cat(sprintf("Filtered MAE:   %.6f\n", filter_mae))
cat(sprintf("Smoothed MAE:   %.6f\n", smooth_mae))
cat(sprintf("RMSE improvement (filter -> smoother): %.2f%%\n",
            (filter_rmse - smooth_rmse) / filter_rmse * 100))

# Verificacao: smoother deve ter RMSE <= filter
if (smooth_rmse <= filter_rmse) {
  cat("[OK] Smoother RMSE <= Filter RMSE (esperado)\n")
} else {
  cat("[WARN] Smoother RMSE > Filter RMSE (pode ocorrer com poucos M)\n")
}

# ---------------------------------------------------------------------------
# 7. Salvar resultados
# ---------------------------------------------------------------------------
results <- data.frame(
  t = t_idx,
  h_true = h_true,
  filtered_mean = filtered_means,
  filtered_var = filtered_vars,
  smoothed_mean = smoothed_means,
  smoothed_var = smoothed_vars
)

write.csv(results, file.path(out_dir, "results_r_smoothed.csv"), row.names = FALSE)
cat(sprintf("\nResultados salvos em: %s\n",
            file.path(out_dir, "results_r_smoothed.csv")))

# Sumario
cat(sprintf("\n=== Sumario ===\n"))
cat(sprintf("Modelo: Stochastic Volatility (mu=%.1f, phi=%.2f, sigma_h=%.2f)\n",
            mu, phi_sv, sigma_h))
cat(sprintf("Dados: simulated_sv.csv (T=%d observacoes)\n", TT))
cat(sprintf("Particulas: N=%d, Trajetorias backward: M=%d\n", N, M))
cat(sprintf("Metodo: Forward Filtering Backward Smoothing (FFBSm)\n"))
cat(sprintf("Pacote: pomp (pfilter para forward pass)\n"))
