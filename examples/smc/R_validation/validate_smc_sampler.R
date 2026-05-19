#!/usr/bin/env Rscript
# ===========================================================================
# Validacao: SMC Sampler no modelo Linear-Gaussian
#
# Implementa um SMC sampler com tempering adaptativo para estimar
# parametros do modelo linear-Gaussian via posterior sampling.
#
# Modelo:
#   x_t = phi * x_{t-1} + sigma_x * w_t,  w_t ~ N(0,1)
#   y_t = x_t + sigma_y * v_t,             v_t ~ N(0,1)
#
# Priors:
#   phi     ~ Uniform(0, 1)
#   sigma_x ~ HalfNormal(0, 1)
#   sigma_y ~ HalfNormal(0, 2)
#
# Pacotes requeridos: KFAS (Kalman filter para log-likelihood exata)
# Pacote de referencia: SMC (Sequential Monte Carlo) - CRAN
#   install.packages("SMC")
#
# Uso: Rscript validate_smc_sampler.R
# ===========================================================================

suppressPackageStartupMessages({
  library(KFAS)
})

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
df <- read.csv(file.path(data_dir, "simulated_linear_gaussian.csv"))
y_obs <- df$y_obs
T_obs <- length(y_obs)
cat(sprintf("Dados Linear-Gaussian carregados: %d observacoes\n", T_obs))

# Parametros verdadeiros (referencia)
TRUE_PHI     <- 0.95
TRUE_SIGMA_X <- 0.5
TRUE_SIGMA_Y <- 1.0

# ---------------------------------------------------------------------------
# 2. Log-likelihood via Kalman filter (KFAS)
# ---------------------------------------------------------------------------
kalman_loglik <- function(phi, sigma_x, sigma_y) {
  # Validacoes
  if (abs(phi) >= 1 || sigma_x <= 0 || sigma_y <= 0) return(-Inf)

  model <- SSModel(
    y_obs ~ SSMcustom(
      Z  = matrix(1, 1, 1),
      T  = matrix(phi, 1, 1),
      R  = matrix(sigma_x, 1, 1),
      Q  = matrix(1, 1, 1),
      a1 = matrix(0, 1, 1),
      P1 = matrix(sigma_x^2 / (1 - phi^2), 1, 1),
      P1inf = matrix(0, 1, 1)
    ),
    H = matrix(sigma_y^2, 1, 1)
  )
  kf <- KFS(model, filtering = "state", smoothing = "none")
  return(kf$logLik)
}

# ---------------------------------------------------------------------------
# 3. Prior e posterior components
# ---------------------------------------------------------------------------
log_prior <- function(theta) {
  phi     <- theta[1]
  sigma_x <- theta[2]
  sigma_y <- theta[3]

  if (phi <= 0 || phi >= 1 || sigma_x <= 0 || sigma_y <= 0) return(-Inf)

  lp <- 0  # phi ~ U(0,1): log(1) = 0
  lp <- lp + dnorm(sigma_x, 0, 1, log = TRUE) + log(2)  # HalfNormal(1)
  lp <- lp + dnorm(sigma_y, 0, 2, log = TRUE) + log(2)  # HalfNormal(2)
  return(lp)
}

log_target <- function(theta, gamma_t) {
  lp <- log_prior(theta)
  if (is.infinite(lp)) return(-Inf)
  ll <- kalman_loglik(theta[1], theta[2], theta[3])
  if (is.infinite(ll)) return(-Inf)
  return(lp + gamma_t * ll)
}

# ---------------------------------------------------------------------------
# 4. SMC Sampler with Adaptive Tempering
# ---------------------------------------------------------------------------
cat("\n=== SMC Sampler com Tempering Adaptativo ===\n")

N_PARTICLES  <- 2000
N_MCMC_MOVES <- 5
ESS_TARGET   <- 0.5 * N_PARTICLES

# Initialize particles from prior
particles <- matrix(NA, nrow = N_PARTICLES, ncol = 3)
colnames(particles) <- c("phi", "sigma_x", "sigma_y")
for (i in 1:N_PARTICLES) {
  particles[i, 1] <- runif(1, 0, 1)          # phi ~ U(0,1)
  particles[i, 2] <- abs(rnorm(1, 0, 1))     # sigma_x ~ HalfNormal(1)
  particles[i, 3] <- abs(rnorm(1, 0, 2))     # sigma_y ~ HalfNormal(2)
}

log_weights <- rep(0, N_PARTICLES)
weights     <- rep(1 / N_PARTICLES, N_PARTICLES)
gamma_curr  <- 0.0
log_evidence <- 0.0

# Cache log-likelihoods for current particles
ll_cache <- numeric(N_PARTICLES)
for (i in 1:N_PARTICLES) {
  ll_cache[i] <- kalman_loglik(particles[i, 1], particles[i, 2], particles[i, 3])
}

step <- 0

while (gamma_curr < 1.0) {
  step <- step + 1

  # --- Adaptive tempering: find delta_gamma via bisection ---
  find_delta <- function(delta) {
    gamma_new <- min(gamma_curr + delta, 1.0)
    dg <- gamma_new - gamma_curr
    log_inc <- dg * ll_cache
    log_inc <- log_inc - max(log_inc)
    w_inc <- exp(log_inc)
    ess <- (sum(w_inc))^2 / sum(w_inc^2)
    return(ess - ESS_TARGET)
  }

  # Check if we can jump to gamma=1
  test_full <- find_delta(1.0 - gamma_curr)
  if (test_full >= 0) {
    delta_gamma <- 1.0 - gamma_curr
  } else {
    # Bisection
    lo <- 1e-6
    hi <- 1.0 - gamma_curr
    for (iter in 1:50) {
      mid <- (lo + hi) / 2
      if (find_delta(mid) > 0) {
        lo <- mid
      } else {
        hi <- mid
      }
    }
    delta_gamma <- (lo + hi) / 2
  }

  gamma_new <- min(gamma_curr + delta_gamma, 1.0)

  # --- Reweight ---
  log_inc <- (gamma_new - gamma_curr) * ll_cache
  log_max <- max(log_inc)
  w_inc <- exp(log_inc - log_max)
  log_evidence <- log_evidence + log_max + log(mean(w_inc))

  weights <- w_inc / sum(w_inc)

  cat(sprintf("  Step %d: gamma %.4f -> %.4f, ESS = %.0f\n",
              step, gamma_curr, gamma_new,
              1 / sum(weights^2)))

  gamma_curr <- gamma_new

  # --- Resample (systematic) ---
  cumw <- cumsum(weights)
  u <- (runif(1) + 0:(N_PARTICLES - 1)) / N_PARTICLES
  idx <- findInterval(u, cumw) + 1
  idx <- pmin(idx, N_PARTICLES)
  particles <- particles[idx, , drop = FALSE]
  ll_cache  <- ll_cache[idx]
  weights   <- rep(1 / N_PARTICLES, N_PARTICLES)

  # --- MCMC rejuvenation (random walk Metropolis) ---
  # Adaptive covariance from current particles
  cov_mat <- cov(particles) * 2.38^2 / 3
  # Ensure positive-definite
  cov_mat <- cov_mat + diag(1e-6, 3)
  chol_cov <- tryCatch(chol(cov_mat), error = function(e) chol(diag(diag(cov_mat))))

  accept_count <- 0
  total_proposals <- 0

  for (m in 1:N_MCMC_MOVES) {
    for (i in 1:N_PARTICLES) {
      proposal <- particles[i, ] + as.numeric(rnorm(3) %*% chol_cov)

      # Reflect to enforce constraints
      if (proposal[1] <= 0 || proposal[1] >= 1) next
      if (proposal[2] <= 0) next
      if (proposal[3] <= 0) next

      ll_prop <- kalman_loglik(proposal[1], proposal[2], proposal[3])
      lp_prop <- log_prior(proposal)
      lp_curr <- log_prior(particles[i, ])

      log_alpha <- (lp_prop + gamma_curr * ll_prop) -
                   (lp_curr + gamma_curr * ll_cache[i])

      total_proposals <- total_proposals + 1
      if (log(runif(1)) < log_alpha) {
        particles[i, ] <- proposal
        ll_cache[i]    <- ll_prop
        accept_count   <- accept_count + 1
      }
    }
  }

  if (total_proposals > 0) {
    cat(sprintf("    MCMC acceptance rate: %.1f%%\n",
                100 * accept_count / total_proposals))
  }
}

# ---------------------------------------------------------------------------
# 5. Resultados
# ---------------------------------------------------------------------------
cat(sprintf("\n=== Resultados SMC Sampler ===\n"))
cat(sprintf("Log-evidencia: %.4f\n", log_evidence))
cat(sprintf("Numero de steps: %d\n", step))

param_names  <- c("phi", "sigma_x", "sigma_y")
true_values  <- c(TRUE_PHI, TRUE_SIGMA_X, TRUE_SIGMA_Y)

results <- data.frame(
  parameter   = character(),
  true_value  = numeric(),
  smc_mean    = numeric(),
  smc_std     = numeric(),
  ci_lower    = numeric(),
  ci_upper    = numeric(),
  within_2sd  = logical(),
  stringsAsFactors = FALSE
)

for (j in 1:3) {
  vals   <- particles[, j]
  m      <- mean(vals)
  s      <- sd(vals)
  ci_lo  <- quantile(vals, 0.025)
  ci_hi  <- quantile(vals, 0.975)
  w2sd   <- (true_values[j] >= m - 2 * s) && (true_values[j] <= m + 2 * s)

  results <- rbind(results, data.frame(
    parameter  = param_names[j],
    true_value = true_values[j],
    smc_mean   = m,
    smc_std    = s,
    ci_lower   = as.numeric(ci_lo),
    ci_upper   = as.numeric(ci_hi),
    within_2sd = w2sd,
    stringsAsFactors = FALSE
  ))

  cat(sprintf("  %s: mean=%.4f, std=%.4f, 95%% CI=[%.4f, %.4f] (true=%.2f) %s\n",
              param_names[j], m, s, ci_lo, ci_hi, true_values[j],
              ifelse(w2sd, "OK", "WARN")))
}

# Add log-evidence and n_steps rows
results <- rbind(results, data.frame(
  parameter = "log_evidence", true_value = NA,
  smc_mean = log_evidence, smc_std = NA,
  ci_lower = NA, ci_upper = NA, within_2sd = TRUE,
  stringsAsFactors = FALSE
))
results <- rbind(results, data.frame(
  parameter = "n_steps", true_value = NA,
  smc_mean = step, smc_std = NA,
  ci_lower = NA, ci_upper = NA, within_2sd = TRUE,
  stringsAsFactors = FALSE
))

# ---------------------------------------------------------------------------
# 6. Salvar
# ---------------------------------------------------------------------------
out_file <- file.path(out_dir, "results_r_smc_sampler.csv")
write.csv(results, out_file, row.names = FALSE)
cat(sprintf("\nResultados salvos em: %s\n", out_file))

# ---------------------------------------------------------------------------
# 7. Referencia: Kalman log-likelihood com parametros verdadeiros
# ---------------------------------------------------------------------------
ref_ll <- kalman_loglik(TRUE_PHI, TRUE_SIGMA_X, TRUE_SIGMA_Y)
cat(sprintf("\nReferencia Kalman (params verdadeiros): loglik = %.4f\n", ref_ll))
cat("\nPacotes R utilizados: KFAS (Kalman filter)\n")
cat("Pacote de referencia: SMC (CRAN) - SMC samplers e tempering adaptativo\n")
cat("  Instalacao: install.packages('SMC')\n")
