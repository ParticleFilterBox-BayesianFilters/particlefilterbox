#!/usr/bin/env Rscript
# ===========================================================================
# Validacao: Auxiliary Particle Filter (APF) no modelo Stochastic Volatility
#
# Implementa APF manualmente e compara com BPF (pomp) no mesmo modelo SV.
# O APF usa look-ahead weights baseados na media da transicao (first-stage),
# seguido de correcao (second-stage).
#
# Modelo SV:
#   h_t = mu + phi*(h_{t-1} - mu) + sigma_h * w_t,  w_t ~ N(0,1)
#   y_t = exp(h_t/2) * v_t,                          v_t ~ N(0,1)
#
# Pacotes requeridos: pomp (>= 5.0)
# Instalar: install.packages("pomp")
#
# Referencia: Pitt & Shephard (1999) - Filtering via Simulation:
#   Auxiliary Particle Filters
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

y_obs <- df$y_obs
h_true <- df$h_true
TT <- nrow(df)

# ---------------------------------------------------------------------------
# 2. Auxiliary Particle Filter (implementacao manual)
# ---------------------------------------------------------------------------
# APF (Pitt & Shephard 1999):
#   1. Compute look-ahead weights: g_t^i = p(y_t | mu_t^i)
#      where mu_t^i = E[h_t | h_{t-1}^i] = mu + phi*(h_{t-1}^i - mu)
#   2. Resample using first-stage weights: w_{t-1}^i * g_t^i
#   3. Propagate: h_t^i ~ p(h_t | h_{t-1}^{k_i})
#   4. Compute second-stage correction:
#      w_t^i = p(y_t | h_t^i) / p(y_t | mu_t^{k_i})

log_obs_density <- function(y, h) {
  # y ~ N(0, exp(h)), so log p(y|h) = -0.5*log(2*pi) - h/2 - 0.5*y^2*exp(-h)
  -0.5 * log(2 * pi) - h / 2 - 0.5 * y^2 * exp(-h)
}

apf_sv <- function(y, N, mu, phi, sigma_h) {
  TT <- length(y)

  # Storage
  particles <- matrix(NA, nrow = N, ncol = TT)
  log_weights <- matrix(NA, nrow = N, ncol = TT)
  ess_vec <- numeric(TT)
  log_lik <- 0.0

  # Initialize: h_0 ~ N(mu, sigma_h^2 / (1 - phi^2))
  sd_stat <- sigma_h / sqrt(1 - phi^2)
  h_prev <- rnorm(N, mean = mu, sd = sd_stat)

  for (t in seq_len(TT)) {
    # --- First stage: look-ahead weights ---
    # Predicted mean for each particle
    h_mean <- mu + phi * (h_prev - mu)

    # Look-ahead log-weights: p(y_t | mu_t^i)
    log_g <- log_obs_density(y[t], h_mean)

    # Uniform prior weights (after resampling at each step)
    log_first_stage <- log_g
    max_lfs <- max(log_first_stage)
    w_first <- exp(log_first_stage - max_lfs)
    w_first <- w_first / sum(w_first)

    # --- Resample using first-stage weights ---
    ancestors <- sample.int(N, size = N, replace = TRUE, prob = w_first)
    h_resampled <- h_prev[ancestors]
    h_mean_resampled <- h_mean[ancestors]

    # --- Propagate ---
    h_new <- mu + phi * (h_resampled - mu) + sigma_h * rnorm(N)

    # --- Second-stage correction weights ---
    # w_t^i = p(y_t | h_t^i) / p(y_t | mu_t^{k_i})
    log_num <- log_obs_density(y[t], h_new)
    log_den <- log_obs_density(y[t], h_mean_resampled)
    log_w <- log_num - log_den

    # Normalize
    max_lw <- max(log_w)
    w_norm <- exp(log_w - max_lw)
    w_sum <- sum(w_norm)
    w_norm <- w_norm / w_sum

    # Log-likelihood increment: log(mean(w)) + max_lw + log of first-stage normalizer
    log_lik <- log_lik + log(w_sum / N) + max_lw + log(sum(exp(log_first_stage - max_lfs))) - log(N) + max_lfs

    # ESS
    ess_vec[t] <- 1 / sum(w_norm^2)

    # Store weighted mean as filtered state
    particles[, t] <- sum(w_norm * h_new)
    log_weights[, t] <- log_w

    # Resample for next step (systematic resampling)
    idx <- sample.int(N, size = N, replace = TRUE, prob = w_norm)
    h_prev <- h_new[idx]
  }

  list(
    filtered_mean = particles[1, ],  # stored as weighted mean in row 1
    ess = ess_vec,
    log_likelihood = log_lik
  )
}

# ---------------------------------------------------------------------------
# 3. BPF via pomp (referencia)
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
# 4. Rodar APF e BPF com diferentes N
# ---------------------------------------------------------------------------
cat(sprintf("\n%s\n", strrep("=", 62)))
cat("  Auxiliary PF vs Bootstrap PF - Stochastic Volatility\n")
cat(sprintf("%s\n\n", strrep("=", 62)))

results <- data.frame()
N_values <- c(500, 1000, 2000)

for (N in N_values) {
  # APF (manual)
  set.seed(42)
  apf_res <- apf_sv(y_obs, N, mu, phi_sv, sigma_h)
  apf_loglik <- apf_res$log_likelihood
  apf_ess_mean <- mean(apf_res$ess)
  apf_ess_min <- min(apf_res$ess)

  # RMSE
  apf_rmse <- sqrt(mean((apf_res$filtered_mean - h_true)^2))

  results <- rbind(results, data.frame(
    Filter = "APF",
    N = N,
    loglik = apf_loglik,
    RMSE = apf_rmse,
    ESS_mean = apf_ess_mean,
    ESS_min = apf_ess_min,
    ESS_N_pct = apf_ess_mean / N * 100
  ))
  cat(sprintf("APF N=%5d: logLik=%.2f, RMSE=%.4f, ESS_mean=%.1f (%.1f%%)\n",
              N, apf_loglik, apf_rmse, apf_ess_mean, apf_ess_mean / N * 100))

  # BPF via pomp
  set.seed(42)
  pf <- pfilter(sv_model, Np = N, params = params_sv)
  bpf_loglik <- logLik(pf)
  bpf_ess_vals <- eff_sample_size(pf)
  bpf_ess_mean <- mean(bpf_ess_vals)
  bpf_ess_min <- min(bpf_ess_vals)

  results <- rbind(results, data.frame(
    Filter = "BPF",
    N = N,
    loglik = bpf_loglik,
    RMSE = NA,
    ESS_mean = bpf_ess_mean,
    ESS_min = bpf_ess_min,
    ESS_N_pct = bpf_ess_mean / N * 100
  ))
  cat(sprintf("BPF N=%5d: logLik=%.2f, ESS_mean=%.1f (%.1f%%)\n",
              N, bpf_loglik, bpf_ess_mean, bpf_ess_mean / N * 100))
}

# ---------------------------------------------------------------------------
# 5. Verificacao: APF deve ter ESS maior que BPF
# ---------------------------------------------------------------------------
cat(sprintf("\n--- ESS Comparison ---\n"))
for (N in N_values) {
  apf_ess <- results$ESS_N_pct[results$Filter == "APF" & results$N == N]
  bpf_ess <- results$ESS_N_pct[results$Filter == "BPF" & results$N == N]
  status <- ifelse(apf_ess > bpf_ess, "OK", "UNEXPECTED")
  cat(sprintf("  N=%d: APF ESS=%.1f%% vs BPF ESS=%.1f%% -- %s\n",
              N, apf_ess, bpf_ess, status))
}

# ---------------------------------------------------------------------------
# 6. Salvar resultados
# ---------------------------------------------------------------------------
write.csv(results, file.path(out_dir, "results_r_auxiliary.csv"), row.names = FALSE)
cat(sprintf("\nResultados salvos em: %s\n",
            file.path(out_dir, "results_r_auxiliary.csv")))

cat("\nPacotes utilizados:\n")
cat("  - pomp (>= 5.0): Bootstrap PF (pfilter) como referencia\n")
cat("  - APF implementado manualmente (Pitt & Shephard 1999)\n")
cat("  - nimbleSMC: alternativa para APF via buildAuxiliaryFilter\n")
