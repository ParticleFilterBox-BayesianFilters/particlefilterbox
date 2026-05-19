#!/usr/bin/env Rscript
# ==============================================================================
# validate_sv_leverage.R
# Cross-validation of SV with leverage (correlated innovations) via
# stochvol::svlsample. Compared against particlefilterbox's PMMH posterior
# for (mu, phi, sigma, rho).
#
# Model:
#   y_t = exp(h_t / 2) * eps_t
#   h_{t+1} = mu + phi * (h_t - mu) + sigma * eta_t
#   corr(eps_t, eta_t) = rho   (leverage; typically rho < 0 for equities)
#
# Required R packages: stochvol (>= 3.0), coda
# Install: install.packages(c("stochvol", "coda"))
#
# Usage: Rscript validate_sv_leverage.R
# ==============================================================================

suppressPackageStartupMessages({
  has_stochvol <- requireNamespace("stochvol", quietly = TRUE)
  has_coda     <- requireNamespace("coda", quietly = TRUE)
})

cat(strrep("=", 71), "\n")
cat("SV with leverage validation via stochvol::svlsample\n")
cat("Dataset: simulated_sv_leverage.csv (true rho = -0.5)\n")
cat(strrep("=", 71), "\n\n")

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  }
  return(getwd())
}
script_dir <- get_script_dir()

# ------------------------------------------------------------------------------
# Reference Bootstrap PF (R) for SV with leverage. Direct algorithm-level
# cross-check against the Python bootstrap PF using identical fixed params.
# ------------------------------------------------------------------------------
bootstrap_pf_sv_lev_R <- function(y, mu, phi, sigma, rho = 0.0,
                                  n_particles = 1000, seed = 42) {
  set.seed(seed)
  T <- length(y)
  var_stat <- sigma^2 / (1 - phi^2)
  particles <- rnorm(n_particles, mu, sqrt(max(var_stat, 1e-10)))
  etas <- rep(0.0, n_particles)
  fmean <- numeric(T); fq05 <- numeric(T); fq95 <- numeric(T)
  var_cond <- max(1 - rho^2, 1e-10)
  for (t in seq_len(T)) {
    vol <- exp(particles / 2)
    eps <- y[t] / vol
    if (t == 1 || abs(rho) < 1e-10) {
      log_w <- -0.5 * log(2 * pi) - log(vol) - 0.5 * eps^2
    } else {
      mean_cond <- rho * etas
      log_w <- -0.5 * log(2 * pi * var_cond) -
               0.5 * (eps - mean_cond)^2 / var_cond - log(vol)
    }
    max_lw <- max(log_w)
    w <- exp(log_w - max_lw); w <- w / sum(w)
    fmean[t] <- sum(w * particles)
    o <- order(particles); cw <- cumsum(w[o])
    fq05[t] <- particles[o][which(cw >= 0.05)[1]]
    fq95[t] <- particles[o][which(cw >= 0.95)[1]]
    idx <- sample.int(n_particles, n_particles, replace = TRUE, prob = w)
    particles <- particles[idx]
    etas <- rnorm(n_particles)
    particles <- mu + phi * (particles - mu) + sigma * etas
  }
  list(h_mean = fmean, h_q05 = fq05, h_q95 = fq95)
}

# ------------------------------------------------------------------------------
# 1. Load simulated leverage dataset
# ------------------------------------------------------------------------------
data_path <- file.path(script_dir, "..", "data", "simulated_sv_leverage.csv")
if (!file.exists(data_path)) {
  stop("Simulated leverage dataset not found at: ", data_path)
}
sim <- read.csv(data_path, stringsAsFactors = FALSE)
T_USE <- min(200, nrow(sim))
sim <- sim[1:T_USE, ]
y      <- sim$y_obs
h_true <- sim$h_true
cat(sprintf("Loaded %d observations from simulated SV-leverage data\n", T_USE))
cat(sprintf("  y mean = %.4f, sd = %.4f\n", mean(y), sd(y)))
cat(sprintf("  true h range: [%.3f, %.3f]\n\n", min(h_true), max(h_true)))

TRUE_PARAMS <- list(mu = -1.0, phi = 0.97, sigma = 0.15, rho = -0.5)

output_csv <- file.path(script_dir, "results_r_sv_leverage.csv")

# ------------------------------------------------------------------------------
# 2. Run stochvol::svlsample (or graceful fallback)
# ------------------------------------------------------------------------------
if (!has_stochvol) {
  cat("WARNING: package 'stochvol' not installed; writing empty placeholder.\n")
  cat("Install with: install.packages('stochvol')\n")
  empty <- data.frame(
    section = character(),
    t = integer(),
    y_obs = numeric(),
    h_true = numeric(),
    h_mean = numeric(),
    h_sd = numeric(),
    h_q05 = numeric(),
    h_q95 = numeric(),
    vol_mean = numeric(),
    parameter = character(),
    true_value = numeric(),
    post_mean = numeric(),
    post_sd = numeric(),
    ci_low = numeric(),
    ci_high = numeric(),
    ess = numeric(),
    stringsAsFactors = FALSE
  )
  write.csv(empty, output_csv, row.names = FALSE)
  cat("Empty placeholder saved to:", output_csv, "\n")
  quit(status = 0)
}

library(stochvol)
if (has_coda) library(coda)

set.seed(42)

cat("Running stochvol::svlsample (draws=10000, burnin=2000) ...\n")
# Priors:
#   mu  ~ Normal(-1, 1)            -> priormu = c(-1, 1)
#   phi: (phi+1)/2 ~ Beta(20, 1.5)  -> priorphi = c(20, 1.5)
#   sigma ~ scale 0.5 (broader, lets sigma vary near true 0.15)
#   rho: (rho+1)/2 ~ Beta(4, 4)    -> priorrho = c(4, 4)  [centered at 0]
t0 <- Sys.time()
fit <- svlsample(
  y          = y,
  draws      = 10000,
  burnin     = 2000,
  priormu    = c(-1, 1),
  priorphi   = c(20, 1.5),
  priorsigma = 0.5,
  priorrho   = c(4, 4),
  quiet      = TRUE
)
cat(sprintf("  done in %.1f sec\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))

para_draws <- as.data.frame(fit$para[[1]])
colnames(para_draws) <- tolower(colnames(para_draws))
need_cols <- c("mu", "phi", "sigma", "rho")
for (col in need_cols) {
  if (!(col %in% colnames(para_draws))) {
    stop("Expected column '", col, "' missing from svlsample output")
  }
}

post_summary <- data.frame(
  parameter  = need_cols,
  true_value = c(TRUE_PARAMS$mu, TRUE_PARAMS$phi, TRUE_PARAMS$sigma, TRUE_PARAMS$rho),
  post_mean  = sapply(need_cols, function(p) mean(para_draws[[p]])),
  post_sd    = sapply(need_cols, function(p) sd(para_draws[[p]])),
  ci_low     = sapply(need_cols, function(p) quantile(para_draws[[p]], 0.025)),
  ci_high    = sapply(need_cols, function(p) quantile(para_draws[[p]], 0.975)),
  stringsAsFactors = FALSE
)
if (has_coda) {
  post_summary$ess <- sapply(need_cols, function(p) effectiveSize(para_draws[[p]]))
} else {
  post_summary$ess <- NA_real_
}

cat("\nPosterior summary (R / stochvol::svlsample):\n")
print(post_summary, row.names = FALSE, digits = 4)

cat(sprintf("\nrho posterior:  mean=%.4f  sd=%.4f  95%% CI=[%.4f, %.4f]\n",
            post_summary$post_mean[post_summary$parameter == "rho"],
            post_summary$post_sd[post_summary$parameter == "rho"],
            post_summary$ci_low[post_summary$parameter == "rho"],
            post_summary$ci_high[post_summary$parameter == "rho"]))
cat(sprintf("rho true value: %.4f\n", TRUE_PARAMS$rho))

# ------------------------------------------------------------------------------
# 3. Latent log-volatility summary (filtered/smoothed)
# ------------------------------------------------------------------------------
latent_mat <- as.matrix(fit$latent[[1]])
T_eff <- min(T_USE, ncol(latent_mat))
latent_mat <- latent_mat[, 1:T_eff]

h_mean <- colMeans(latent_mat)
h_sd   <- apply(latent_mat, 2, sd)
h_q05  <- apply(latent_mat, 2, quantile, 0.05)
h_q95  <- apply(latent_mat, 2, quantile, 0.95)

filtered_df <- data.frame(
  section    = "filtered",
  t          = seq_len(T_eff) - 1L,
  y_obs      = y[1:T_eff],
  h_true     = h_true[1:T_eff],
  h_mean     = h_mean,
  h_sd       = h_sd,
  h_q05      = h_q05,
  h_q95      = h_q95,
  vol_mean   = exp(h_mean / 2.0),
  parameter  = "",
  true_value = NA_real_,
  post_mean  = NA_real_,
  post_sd    = NA_real_,
  ci_low     = NA_real_,
  ci_high    = NA_real_,
  ess        = NA_real_,
  stringsAsFactors = FALSE
)

params_df <- data.frame(
  section    = "parameters",
  t          = NA_integer_,
  y_obs      = NA_real_,
  h_true     = NA_real_,
  h_mean     = NA_real_,
  h_sd       = NA_real_,
  h_q05      = NA_real_,
  h_q95      = NA_real_,
  vol_mean   = NA_real_,
  parameter  = post_summary$parameter,
  true_value = post_summary$true_value,
  post_mean  = post_summary$post_mean,
  post_sd    = post_summary$post_sd,
  ci_low     = post_summary$ci_low,
  ci_high    = post_summary$ci_high,
  ess        = post_summary$ess,
  stringsAsFactors = FALSE
)

# Posterior draws (long format)
draws_df <- data.frame(
  section    = "draws",
  t          = NA_integer_,
  y_obs      = NA_real_,
  h_true     = NA_real_,
  h_mean     = NA_real_,
  h_sd       = NA_real_,
  h_q05      = NA_real_,
  h_q95      = NA_real_,
  vol_mean   = NA_real_,
  parameter  = rep(need_cols, each = nrow(para_draws)),
  true_value = NA_real_,
  post_mean  = c(para_draws$mu, para_draws$phi, para_draws$sigma, para_draws$rho),
  post_sd    = NA_real_,
  ci_low     = NA_real_,
  ci_high    = NA_real_,
  ess        = NA_real_,
  stringsAsFactors = FALSE
)

combined <- rbind(filtered_df, params_df, draws_df)

# ------------------------------------------------------------------------------
# Reference Bootstrap PF run on the same data with TRUE params to enable a
# direct algorithm-level comparison with Python's BPF (corr expected > 0.95).
# ------------------------------------------------------------------------------
cat("\n--- R Bootstrap PF reference run (true params) ---\n")
bpf_ref <- bootstrap_pf_sv_lev_R(
  y = y, mu = TRUE_PARAMS$mu, phi = TRUE_PARAMS$phi,
  sigma = TRUE_PARAMS$sigma, rho = TRUE_PARAMS$rho,
  n_particles = 20000, seed = 42
)
cat(sprintf("  filtered h range: [%.3f, %.3f]\n",
            min(bpf_ref$h_mean), max(bpf_ref$h_mean)))

bpf_filtered_df <- data.frame(
  section    = "filtered_bpf",
  t          = seq_len(T_USE) - 1L,
  y_obs      = y,
  h_true     = h_true,
  h_mean     = bpf_ref$h_mean,
  h_sd       = NA_real_,
  h_q05      = bpf_ref$h_q05,
  h_q95      = bpf_ref$h_q95,
  vol_mean   = exp(bpf_ref$h_mean / 2.0),
  parameter  = "",
  true_value = NA_real_,
  post_mean  = NA_real_,
  post_sd    = NA_real_,
  ci_low     = NA_real_,
  ci_high    = NA_real_,
  ess        = NA_real_,
  stringsAsFactors = FALSE
)
combined <- rbind(combined, bpf_filtered_df)

write.csv(combined, output_csv, row.names = FALSE)
cat(sprintf("\nResults saved (%d rows) to: %s\n", nrow(combined), output_csv))

# ------------------------------------------------------------------------------
# 4. Cross-check vs particlefilterbox (Python)
# ------------------------------------------------------------------------------
py_path <- file.path(script_dir, "..", "solutions", "results_sv_leverage.csv")
if (file.exists(py_path)) {
  cat("\n--- Cross-check vs particlefilterbox (Python) ---\n")
  py_df <- read.csv(py_path, stringsAsFactors = FALSE)
  py_filt <- py_df[py_df$section == "filtered", ]
  n_match <- min(nrow(py_filt), T_eff)
  if (n_match > 5) {
    rho_h <- suppressWarnings(cor(h_mean[1:n_match], py_filt$h_mean[1:n_match]))
    cat(sprintf("  Pearson corr(h_mean R, h_mean Python) = %.4f over %d obs\n",
                rho_h, n_match))
    cat(sprintf("  Mean abs diff in h_mean = %.4f\n",
                mean(abs(h_mean[1:n_match] - py_filt$h_mean[1:n_match]))))
  }
  py_par <- py_df[py_df$section == "parameters", ]
  if (nrow(py_par) > 0) {
    cat("  Posterior mean comparison (R vs Python):\n")
    for (i in seq_len(nrow(post_summary))) {
      pname  <- post_summary$parameter[i]
      py_row <- py_par[py_par$parameter == pname, ]
      if (nrow(py_row) == 1) {
        py_mean <- as.numeric(py_row$post_mean[1])
        cat(sprintf("    %-6s  true=%.3f   R=%.4f   Py=%.4f   |diff|=%.4f\n",
                    pname, post_summary$true_value[i],
                    post_summary$post_mean[i], py_mean,
                    abs(post_summary$post_mean[i] - py_mean)))
      }
    }
  }
}

# ------------------------------------------------------------------------------
# 5. Diagnostic against true latent path
# ------------------------------------------------------------------------------
rho_true <- suppressWarnings(cor(h_mean, h_true[1:T_eff]))
rmse <- sqrt(mean((h_mean - h_true[1:T_eff])^2))
cat(sprintf("\nLatent recovery: corr(h_mean, h_true) = %.4f, RMSE = %.4f\n",
            rho_true, rmse))

cat("\nDone.\n")
