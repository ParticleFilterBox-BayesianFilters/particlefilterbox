#!/usr/bin/env Rscript
# ===========================================================================
# Validacao: SMC^2 (Sequential) no modelo Stochastic Volatility
#
# Implementa SMC^2 sequencial: para cada observacao t, roda um passo do
# particle filter interno para cada theta, acumula evidencia, e rejuvenesce
# quando ESS cai abaixo do limiar.
#
# Modelo SV:
#   h_t = mu + phi*(h_{t-1} - mu) + sigma_h * w_t,  w_t ~ N(0,1)
#   y_t = exp(h_t/2) * v_t,                          v_t ~ N(0,1)
#
# Priors:
#   mu      ~ N(-1, 1)
#   phi     ~ Beta(20, 1.5)
#   sigma_h ~ HalfNormal(0, 0.5)
#
# Pacotes requeridos: pomp (>= 5.0) - referencia
# Pacote de referencia: SMC (CRAN), pomp::pmcmc
#
# Uso: Rscript validate_smc2.R
# ===========================================================================

set.seed(42)

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) > 0) {
  script_dir <- dirname(normalizePath(script_path))
} else {
  script_dir <- getwd()
}
data_dir <- file.path(script_dir, "..", "data")
out_dir  <- script_dir

# ---------------------------------------------------------------------------
# 1. Carregar dados
# ---------------------------------------------------------------------------
df <- read.csv(file.path(data_dir, "simulated_sv.csv"))
T_OBS <- 200  # Usar primeiras 200 observacoes (como no Python)
y_obs <- df$y_obs[1:T_OBS]
cat(sprintf("Dados SV carregados: usando %d de %d observacoes\n", T_OBS, nrow(df)))

# Parametros verdadeiros
TRUE_MU      <- -1.0
TRUE_PHI     <- 0.97
TRUE_SIGMA_H <- 0.15

# ---------------------------------------------------------------------------
# 2. Funcoes auxiliares
# ---------------------------------------------------------------------------
log_prior <- function(mu, phi, sigma_h) {
  if (phi <= 0 || phi >= 1 || sigma_h <= 0) return(-Inf)
  lp <- dnorm(mu, -1, 1, log = TRUE)
  lp <- lp + dbeta(phi, 20, 1.5, log = TRUE)
  lp <- lp + dnorm(sigma_h, 0, 0.5, log = TRUE) + log(2)  # HalfNormal
  return(lp)
}

# ---------------------------------------------------------------------------
# 3. SMC^2 Sequential Algorithm
# ---------------------------------------------------------------------------
cat("\n=== SMC^2 Sequencial para Stochastic Volatility ===\n")

N_THETA      <- 500    # Outer particles (parameters)
N_X          <- 200    # Inner particles (states per theta)
N_MCMC_MOVES <- 3
ESS_THRESH   <- 0.5 * N_THETA

start_time <- proc.time()

# Initialize theta particles from prior
theta <- matrix(NA, nrow = N_THETA, ncol = 3)
colnames(theta) <- c("mu", "phi", "sigma_h")
for (i in 1:N_THETA) {
  theta[i, 1] <- rnorm(1, -1, 1)               # mu ~ N(-1, 1)
  theta[i, 2] <- rbeta(1, 20, 1.5)             # phi ~ Beta(20, 1.5)
  theta[i, 3] <- abs(rnorm(1, 0, 0.5))         # sigma_h ~ HalfNormal(0.5)
  if (theta[i, 3] < 0.01) theta[i, 3] <- 0.01
}

# Initialize inner particle filters: h_particles[i, ] = N_X particles for theta_i
h_particles <- matrix(NA, nrow = N_THETA, ncol = N_X)
for (i in 1:N_THETA) {
  mu_i <- theta[i, 1]; phi_i <- theta[i, 2]; sh_i <- theta[i, 3]
  sd_stat <- sh_i / sqrt(max(1 - phi_i^2, 1e-6))
  h_particles[i, ] <- rnorm(N_X, mean = mu_i, sd = sd_stat)
}

# Log-weights for outer SMC (theta particles)
log_w <- rep(0, N_THETA)
log_evidence <- 0.0
n_rejuvenations <- 0

# Cumulative log-likelihood per theta (for PMCMC)
cum_ll <- rep(0, N_THETA)

cat(sprintf("Processando %d observacoes sequencialmente...\n", T_OBS))

for (t in 1:T_OBS) {
  y_t <- y_obs[t]

  # --- Inner PF: one step for each theta particle ---
  incr_ll <- numeric(N_THETA)  # incremental log-likelihood at time t

  for (i in 1:N_THETA) {
    mu_i <- theta[i, 1]; phi_i <- theta[i, 2]; sh_i <- theta[i, 3]
    h <- h_particles[i, ]

    # Observation log-weights: p(y_t | h_t)
    # Use log_mean_exp(log_weights_inner + log_obs) to match Python convention
    # Inner weights are uniform (-log(N_X)) after resampling at each step
    log_obs_w <- dnorm(y_t, mean = 0, sd = exp(h / 2), log = TRUE)
    log_w_inner <- rep(-log(N_X), N_X)  # normalized uniform weights
    combined <- log_w_inner + log_obs_w
    max_c <- max(combined)
    incr_ll[i] <- max_c + log(mean(exp(combined - max_c)))

    # Resample inner particles (using observation weights)
    w <- exp(log_obs_w - max(log_obs_w))
    w_norm <- w / sum(w)
    cumw <- cumsum(w_norm)
    u <- (runif(1) + 0:(N_X - 1)) / N_X
    idx <- findInterval(u, cumw) + 1
    idx <- pmin(idx, N_X)
    h <- h[idx]

    # Propagate states
    h <- mu_i + phi_i * (h - mu_i) + sh_i * rnorm(N_X)
    h_particles[i, ] <- h
  }

  # Update cumulative log-likelihood
  cum_ll <- cum_ll + incr_ll

  # Update outer weights
  log_w <- log_w + incr_ll

  # Accumulate log-evidence: log p(y_t | y_{1:t-1})
  max_lw_outer <- max(log_w)
  w_outer <- exp(log_w - max_lw_outer)
  log_evidence <- log_evidence + max(incr_ll) + log(mean(exp(incr_ll - max(incr_ll))))

  # Normalize outer weights for ESS computation
  w_norm_outer <- w_outer / sum(w_outer)
  ess <- 1 / sum(w_norm_outer^2)

  # Progress
  if (t %% 50 == 0 || t == T_OBS) {
    cat(sprintf("  t=%3d: ESS=%.0f, log_evidence=%.2f\n", t, ess, log_evidence))
  }

  # --- Rejuvenation if ESS too low ---
  if (ess < ESS_THRESH) {
    n_rejuvenations <- n_rejuvenations + 1
    if (t <= 200) {  # Only print for first few
      cat(sprintf("  Rejuvenation %d at t=%d (ESS=%.0f)\n", n_rejuvenations, t, ess))
    }

    # Resample theta particles (and their inner PFs)
    cumw <- cumsum(w_norm_outer)
    u <- (runif(1) + 0:(N_THETA - 1)) / N_THETA
    idx <- findInterval(u, cumw) + 1
    idx <- pmin(idx, N_THETA)
    theta       <- theta[idx, , drop = FALSE]
    h_particles <- h_particles[idx, , drop = FALSE]
    cum_ll      <- cum_ll[idx]
    log_w       <- rep(0, N_THETA)

    # MCMC rejuvenation moves (PMCMC-style)
    cov_mat <- cov(theta) * 2.38^2 / 3
    cov_mat <- cov_mat + diag(1e-6, 3)
    chol_cov <- tryCatch(chol(cov_mat), error = function(e) chol(diag(diag(cov_mat))))

    accept_count <- 0
    total_count  <- 0

    for (m in 1:N_MCMC_MOVES) {
      for (i in 1:N_THETA) {
        proposal <- theta[i, ] + as.numeric(rnorm(3) %*% chol_cov)

        # Check constraints
        if (proposal[2] <= 0 || proposal[2] >= 1 || proposal[3] <= 0) next

        lp_prop <- log_prior(proposal[1], proposal[2], proposal[3])
        lp_curr <- log_prior(theta[i, 1], theta[i, 2], theta[i, 3])
        if (is.infinite(lp_prop)) next

        # Run fresh PF for proposal up to current time t
        mu_p <- proposal[1]; phi_p <- proposal[2]; sh_p <- proposal[3]
        sd_stat_p <- sh_p / sqrt(max(1 - phi_p^2, 1e-6))
        h_prop <- rnorm(N_X, mean = mu_p, sd = sd_stat_p)
        ll_prop <- 0

        for (s in 1:t) {
          log_obs <- dnorm(y_obs[s], mean = 0, sd = exp(h_prop / 2), log = TRUE)
          log_w_inn <- rep(-log(N_X), N_X)
          comb <- log_w_inn + log_obs
          max_c <- max(comb)
          ll_prop <- ll_prop + max_c + log(mean(exp(comb - max_c)))

          # Resample
          w_s <- exp(log_obs - max(log_obs))
          w_s_norm <- w_s / sum(w_s)
          cumw_s <- cumsum(w_s_norm)
          u_s <- (runif(1) + 0:(N_X - 1)) / N_X
          idx_s <- findInterval(u_s, cumw_s) + 1
          idx_s <- pmin(idx_s, N_X)
          h_prop <- h_prop[idx_s]

          # Propagate
          h_prop <- mu_p + phi_p * (h_prop - mu_p) + sh_p * rnorm(N_X)
        }

        log_alpha <- (lp_prop + ll_prop) - (lp_curr + cum_ll[i])
        total_count <- total_count + 1

        if (log(runif(1)) < log_alpha) {
          theta[i, ]       <- proposal
          h_particles[i, ] <- h_prop
          cum_ll[i]        <- ll_prop
          accept_count     <- accept_count + 1
        }
      }
    }
    if (total_count > 0 && n_rejuvenations <= 10) {
      cat(sprintf("    MCMC acceptance: %.1f%% (%d/%d)\n",
                  100 * accept_count / total_count, accept_count, total_count))
    }
  }
}

elapsed <- (proc.time() - start_time)["elapsed"]

# ---------------------------------------------------------------------------
# 4. Resultados
# ---------------------------------------------------------------------------
cat(sprintf("\n=== Resultados SMC^2 ===\n"))
cat(sprintf("Log-evidencia: %.4f\n", log_evidence))
cat(sprintf("Rejuvenations: %d\n", n_rejuvenations))
cat(sprintf("Tempo: %.1f segundos\n", elapsed))

param_names  <- c("mu", "phi", "sigma_h")
true_values  <- c(TRUE_MU, TRUE_PHI, TRUE_SIGMA_H)

results <- data.frame(
  parameter   = character(),
  true_value  = numeric(),
  smc2_mean   = numeric(),
  smc2_std    = numeric(),
  ci_lower    = numeric(),
  ci_upper    = numeric(),
  within_2sd  = logical(),
  stringsAsFactors = FALSE
)

# Use final weighted particles for parameter estimates
max_lw_final <- max(log_w)
w_final <- exp(log_w - max_lw_final)
w_final <- w_final / sum(w_final)

for (j in 1:3) {
  vals   <- theta[, j]
  m      <- sum(w_final * vals)
  s      <- sqrt(sum(w_final * (vals - m)^2))
  # Weighted quantiles (approximate via sorted resampling)
  ord    <- order(vals)
  cum_w  <- cumsum(w_final[ord])
  ci_lo  <- vals[ord[which(cum_w >= 0.025)[1]]]
  ci_hi  <- vals[ord[which(cum_w >= 0.975)[1]]]
  w2sd   <- (true_values[j] >= m - 2 * s) && (true_values[j] <= m + 2 * s)

  results <- rbind(results, data.frame(
    parameter  = param_names[j],
    true_value = true_values[j],
    smc2_mean  = m,
    smc2_std   = s,
    ci_lower   = ci_lo,
    ci_upper   = ci_hi,
    within_2sd = w2sd,
    stringsAsFactors = FALSE
  ))

  cat(sprintf("  %s: mean=%.4f, std=%.4f, 95%% CI=[%.4f, %.4f] (true=%.2f) %s\n",
              param_names[j], m, s, ci_lo, ci_hi, true_values[j],
              ifelse(w2sd, "OK", "WARN")))
}

# Add metadata rows
results <- rbind(results, data.frame(
  parameter = "log_evidence", true_value = NA,
  smc2_mean = log_evidence, smc2_std = NA,
  ci_lower = NA, ci_upper = NA, within_2sd = TRUE,
  stringsAsFactors = FALSE
))
results <- rbind(results, data.frame(
  parameter = "n_rejuvenations", true_value = NA,
  smc2_mean = n_rejuvenations, smc2_std = NA,
  ci_lower = NA, ci_upper = NA, within_2sd = TRUE,
  stringsAsFactors = FALSE
))
results <- rbind(results, data.frame(
  parameter = "elapsed_seconds", true_value = NA,
  smc2_mean = as.numeric(elapsed), smc2_std = NA,
  ci_lower = NA, ci_upper = NA, within_2sd = TRUE,
  stringsAsFactors = FALSE
))

# ---------------------------------------------------------------------------
# 5. Salvar
# ---------------------------------------------------------------------------
out_file <- file.path(out_dir, "results_r_smc2.csv")
write.csv(results, out_file, row.names = FALSE)
cat(sprintf("\nResultados salvos em: %s\n", out_file))

cat("\nPacotes R utilizados: base R (implementacao custom)\n")
cat("Implementacao: SMC^2 sequencial com bootstrap PF interno\n")
cat("Pacotes de referencia: SMC (CRAN), pomp (>= 5.0)\n")
