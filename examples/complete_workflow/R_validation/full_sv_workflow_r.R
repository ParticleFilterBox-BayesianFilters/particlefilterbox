#!/usr/bin/env Rscript
# ==============================================================================
# full_sv_workflow_r.R
#
# Replication of the end-to-end Stochastic Volatility workflow from
# `solutions/solution_01_sv_workflow.py` using the R ecosystem.
#
# Pipeline (mirrors the Python solution):
#   1.  Load last 250 SP500 daily returns from data/sp500_returns.csv
#   2.  Estimate basic SV model y_t = exp(h_t/2) eps_t,
#                              h_t = mu + phi (h_{t-1} - mu) + sigma eta_t
#       via stochvol::svsample (multi-move MCMC sampler).
#   3.  Extract filtered (posterior-mean) latent log-volatility h_t and
#       posterior summaries for (mu, phi, sigma).
#   4.  Smoothing / latent draws come naturally from stochvol output.
#   5.  Forecast log-volatility 25 business days ahead by simulating from
#       the posterior of (mu, phi, sigma, h_T) -- equivalent to the
#       particle-cloud forecast used by the Python solution.
#
# Note on scale conventions:
#   Python solution operates on RAW DECIMAL returns (sd ~ 0.012).
#   stochvol expects PERCENTAGE returns (sd ~ 1.2). We scale by 100 inside
#   the script and translate back when reporting results that should be
#   comparable to the Python pipeline:
#     mu_decimal      = mu_percent - 2 * log(100)
#     h_decimal[t]    = h_percent[t] - 2 * log(100)
#     vol_decimal[t]  = exp(h_decimal[t] / 2) = exp(h_percent[t]/2) / 100
#
# Required R packages: stochvol (>= 3.0), coda
# Usage: Rscript full_sv_workflow_r.R
# ==============================================================================

suppressPackageStartupMessages({
  has_stochvol <- requireNamespace("stochvol", quietly = TRUE)
  has_coda     <- requireNamespace("coda",     quietly = TRUE)
})

cat(strrep("=", 78), "\n", sep = "")
cat("FASE 9.4 / full_sv_workflow_r.R   (stochvol replication of Python pipeline)\n")
cat(strrep("=", 78), "\n\n", sep = "")

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", args, value = TRUE)
  if (length(fa) > 0) return(dirname(normalizePath(sub("^--file=", "", fa[1]))))
  getwd()
}
script_dir <- get_script_dir()

T_USE       <- 250L      # last 250 obs (matches Python solution_01)
H_FORECAST  <- 25L       # forecast horizon (business days) -- matches Python
DRAWS       <- 6000L
BURNIN      <- 1500L
SCALE_LOG   <- 2.0 * log(100.0)   # = 2 log(100), additive shift on mu/h_t
SEED        <- 42L

output_csv <- file.path(script_dir, "results_r_sv_workflow.csv")

# ------------------------------------------------------------------------------
# Graceful early exit if stochvol missing
# ------------------------------------------------------------------------------
write_empty_and_exit <- function(reason) {
  cat("WARNING: ", reason, "\n", sep = "")
  empty <- data.frame(
    section          = character(),
    t                = integer(),
    horizon          = integer(),
    date             = character(),
    y                = numeric(),
    h_mean_perc      = numeric(),
    h_mean_dec       = numeric(),
    h_q05_dec        = numeric(),
    h_q95_dec        = numeric(),
    vol_mean_dec     = numeric(),
    vol_q05_dec      = numeric(),
    vol_q95_dec      = numeric(),
    parameter        = character(),
    post_mean_perc   = numeric(),
    post_mean_dec    = numeric(),
    post_sd          = numeric(),
    ci95_lower_dec   = numeric(),
    ci95_upper_dec   = numeric(),
    ess              = numeric(),
    elapsed_seconds  = numeric(),
    stringsAsFactors = FALSE
  )
  write.csv(empty, output_csv, row.names = FALSE)
  cat("Empty placeholder written to ", output_csv, "\n", sep = "")
  quit(status = 0)
}

if (!has_stochvol) write_empty_and_exit("package 'stochvol' is not installed")

library(stochvol)
if (has_coda) library(coda)

# ------------------------------------------------------------------------------
# Helper: Bootstrap PF for SV (matches the Python implementation's recipe).
#
# Used to produce a filtered h_t series at a fixed (plug-in) theta that can
# be compared one-to-one with Python's `filtered_h_plugin` column. This is
# the cleanest algorithmic cross-check (same algo, same theta, same data)
# and should yield a Pearson correlation > 0.95 with the Python output.
# ------------------------------------------------------------------------------
bootstrap_pf_sv <- function(y, mu, phi, sigma, n_particles = 1500, seed = 42L) {
  set.seed(seed)
  T <- length(y)
  var_stat <- sigma^2 / (1 - phi^2)
  particles <- rnorm(n_particles, mu, sqrt(max(var_stat, 1e-10)))
  fmean <- numeric(T); fq05 <- numeric(T); fq95 <- numeric(T)
  ess_h <- numeric(T); log_lik <- 0.0
  for (t in seq_len(T)) {
    vol <- exp(particles / 2)
    log_w <- -0.5 * log(2 * pi) - log(vol) - 0.5 * (y[t] / vol)^2
    m_lw <- max(log_w)
    w <- exp(log_w - m_lw); s <- sum(w)
    log_lik <- log_lik + m_lw + log(s) - log(n_particles)
    w <- w / s
    fmean[t] <- sum(w * particles)
    o <- order(particles); cw <- cumsum(w[o])
    fq05[t] <- particles[o][which(cw >= 0.05)[1]]
    fq95[t] <- particles[o][which(cw >= 0.95)[1]]
    ess_h[t] <- 1 / sum(w^2)
    if (ess_h[t] < n_particles / 2) {
      idx <- sample.int(n_particles, n_particles, replace = TRUE, prob = w)
      particles <- particles[idx]
    }
    particles <- mu + phi * (particles - mu) + sigma * rnorm(n_particles)
  }
  list(h_mean = fmean, h_q05 = fq05, h_q95 = fq95,
       ess = ess_h, log_lik = log_lik)
}

# ------------------------------------------------------------------------------
# 1. Load SP500 daily returns (last T_USE obs, mirroring solution_01)
# ------------------------------------------------------------------------------
data_path <- file.path(script_dir, "..", "data", "sp500_returns.csv")
if (!file.exists(data_path)) stop("SP500 dataset not found at: ", data_path)

sp500 <- read.csv(data_path, stringsAsFactors = FALSE)
n_total <- nrow(sp500)
T_eff <- min(T_USE, n_total)
sub <- sp500[(n_total - T_eff + 1):n_total, , drop = FALSE]

raw_returns <- as.numeric(sub$returns)
returns_pct <- raw_returns * 100.0
dates       <- as.character(sub$date)

cat(sprintf("Loaded %d returns (last %d of %d total): %s -> %s\n",
            T_eff, T_eff, n_total, dates[1], dates[T_eff]))
cat(sprintf("  raw  : mean=%+.6f  sd=%.6f\n", mean(raw_returns), sd(raw_returns)))
cat(sprintf("  pct  : mean=%+.4f   sd=%.4f\n\n", mean(returns_pct), sd(returns_pct)))

# ------------------------------------------------------------------------------
# 2. stochvol::svsample on percentage returns
# ------------------------------------------------------------------------------
set.seed(SEED)
cat(sprintf("Running stochvol::svsample (draws=%d, burnin=%d) ...\n",
            DRAWS, BURNIN))
t0 <- Sys.time()
fit <- svsample(
  y          = returns_pct,
  draws      = DRAWS,
  burnin     = BURNIN,
  priormu    = c(0, 10),       # weak prior on mu (percent scale)
  priorphi   = c(20, 1.5),     # high-persistence prior, matches Python
  priorsigma = 1,              # default scale for sigma
  quiet      = TRUE
)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("  done in %.2f s\n", elapsed))

# ---- Posterior parameter draws ----
para_draws <- as.data.frame(fit$para[[1]])
colnames(para_draws) <- tolower(colnames(para_draws))
need_cols <- c("mu", "phi", "sigma")
for (c_ in need_cols) {
  if (!(c_ %in% colnames(para_draws))) stop("missing column ", c_, " in stochvol output")
}

post_mean_perc <- vapply(need_cols, function(p) mean(para_draws[[p]]), numeric(1))
post_sd_       <- vapply(need_cols, function(p) sd(para_draws[[p]]),   numeric(1))
ci95           <- t(vapply(need_cols, function(p) quantile(para_draws[[p]], c(0.025, 0.975)),
                           numeric(2)))
ess_par <- if (has_coda) {
  vapply(need_cols, function(p) effectiveSize(para_draws[[p]]), numeric(1))
} else rep(NA_real_, length(need_cols))

# Decimal-scale equivalents (only mu shifts; phi and sigma are scale-invariant)
post_mean_dec <- post_mean_perc
post_mean_dec["mu"] <- post_mean_perc["mu"] - SCALE_LOG
ci95_dec <- ci95
ci95_dec[which(rownames(ci95_dec) == "mu"), ] <- ci95["mu", ] - SCALE_LOG

cat("\nPosterior parameter summary (percent scale  |  decimal scale):\n")
for (i in seq_along(need_cols)) {
  cat(sprintf("  %-6s  mean=%+.4f / %+.4f   sd=%.4f   CI95=[%+.4f, %+.4f] / [%+.4f, %+.4f]   ESS=%.0f\n",
              need_cols[i],
              post_mean_perc[i], post_mean_dec[i],
              post_sd_[i],
              ci95[i, 1], ci95[i, 2],
              ci95_dec[i, 1], ci95_dec[i, 2],
              ess_par[i]))
}

# ------------------------------------------------------------------------------
# 3. Filtered / smoothed latent log-volatility (posterior mean over draws)
# ------------------------------------------------------------------------------
latent_mat <- as.matrix(fit$latent[[1]])   # n_draws x T
T_eff_lat  <- ncol(latent_mat)
if (T_eff_lat != T_eff) {
  T_eff <- min(T_eff, T_eff_lat)
  latent_mat <- latent_mat[, 1:T_eff]
  raw_returns <- raw_returns[1:T_eff]
  returns_pct <- returns_pct[1:T_eff]
  dates       <- dates[1:T_eff]
}

h_mean_perc <- colMeans(latent_mat)
h_q05_perc  <- apply(latent_mat, 2, quantile, 0.05)
h_q95_perc  <- apply(latent_mat, 2, quantile, 0.95)

h_mean_dec <- h_mean_perc - SCALE_LOG
h_q05_dec  <- h_q05_perc  - SCALE_LOG
h_q95_dec  <- h_q95_perc  - SCALE_LOG

vol_mean_dec <- exp(h_mean_dec / 2.0)
vol_q05_dec  <- exp(h_q05_dec  / 2.0)
vol_q95_dec  <- exp(h_q95_dec  / 2.0)

cat(sprintf("\nFiltered / smoothed h (decimal scale): range = [%+.3f, %+.3f]\n",
            min(h_mean_dec), max(h_mean_dec)))
cat(sprintf("Implied volatility (decimal): range = [%.5f, %.5f]\n",
            min(vol_mean_dec), max(vol_mean_dec)))

# ------------------------------------------------------------------------------
# 4. Forecast (H business days ahead) by simulating from posterior of theta and h_T
# ------------------------------------------------------------------------------
set.seed(SEED + 1L)
n_draws <- nrow(para_draws)
h_T_draws <- latent_mat[, T_eff]   # length n_draws
fwd <- matrix(0.0, n_draws, H_FORECAST)
mu_d  <- para_draws$mu
phi_d <- para_draws$phi
sig_d <- para_draws$sigma
h_curr <- h_T_draws
for (k in seq_len(H_FORECAST)) {
  h_curr <- mu_d + phi_d * (h_curr - mu_d) + sig_d * rnorm(n_draws)
  fwd[, k] <- h_curr
}
fwd_dec <- fwd - SCALE_LOG

fwd_q05_h <- apply(fwd_dec, 2, quantile, 0.05)
fwd_q50_h <- apply(fwd_dec, 2, quantile, 0.50)
fwd_q95_h <- apply(fwd_dec, 2, quantile, 0.95)
fwd_q05_v <- exp(fwd_q05_h / 2.0)
fwd_q50_v <- exp(fwd_q50_h / 2.0)
fwd_q95_v <- exp(fwd_q95_h / 2.0)

# Generate forecast dates: tslist of business days starting day after last_date
last_date <- as.Date(dates[T_eff])
fwd_dates <- character(H_FORECAST)
d <- last_date
i <- 0L
while (i < H_FORECAST) {
  d <- d + 1L
  wd <- as.POSIXlt(d)$wday    # 0 Sun .. 6 Sat
  if (wd >= 1L && wd <= 5L) {
    i <- i + 1L
    fwd_dates[i] <- format(d, "%Y-%m-%d")
  }
}

cat(sprintf("\nForecast (decimal scale):\n"))
for (h in c(1L, 5L, 20L)) {
  if (h <= H_FORECAST) {
    cat(sprintf("  h=%2d  median vol=%.5f  90%% CI=[%.5f, %.5f]\n",
                h, fwd_q50_v[h], fwd_q05_v[h], fwd_q95_v[h]))
  }
}

# ------------------------------------------------------------------------------
# 5. Persist long-format CSV
# ------------------------------------------------------------------------------
COMMON_COLS <- c(
  "section", "t", "horizon", "date", "y",
  "h_mean_perc", "h_mean_dec", "h_q05_dec", "h_q95_dec",
  "vol_mean_dec", "vol_q05_dec", "vol_q95_dec",
  "parameter", "post_mean_perc", "post_mean_dec", "post_sd",
  "ci95_lower_dec", "ci95_upper_dec", "ess", "elapsed_seconds"
)

mk_row <- function(...) {
  vals <- list(...)
  for (col in COMMON_COLS) {
    if (is.null(vals[[col]])) {
      vals[[col]] <- if (col %in% c("section", "date", "parameter")) NA_character_ else NA_real_
    }
  }
  as.data.frame(vals[COMMON_COLS], stringsAsFactors = FALSE)
}

filtered_df <- data.frame(
  section          = "filtered",
  t                = seq_len(T_eff) - 1L,
  horizon          = NA_integer_,
  date             = dates,
  y                = raw_returns,
  h_mean_perc      = h_mean_perc,
  h_mean_dec       = h_mean_dec,
  h_q05_dec        = h_q05_dec,
  h_q95_dec        = h_q95_dec,
  vol_mean_dec     = vol_mean_dec,
  vol_q05_dec      = vol_q05_dec,
  vol_q95_dec      = vol_q95_dec,
  parameter        = NA_character_,
  post_mean_perc   = NA_real_,
  post_mean_dec    = NA_real_,
  post_sd          = NA_real_,
  ci95_lower_dec   = NA_real_,
  ci95_upper_dec   = NA_real_,
  ess              = NA_real_,
  elapsed_seconds  = elapsed,
  stringsAsFactors = FALSE
)

params_df <- data.frame(
  section          = "parameters",
  t                = NA_integer_,
  horizon          = NA_integer_,
  date             = NA_character_,
  y                = NA_real_,
  h_mean_perc      = NA_real_,
  h_mean_dec       = NA_real_,
  h_q05_dec        = NA_real_,
  h_q95_dec        = NA_real_,
  vol_mean_dec     = NA_real_,
  vol_q05_dec      = NA_real_,
  vol_q95_dec      = NA_real_,
  parameter        = need_cols,
  post_mean_perc   = unname(post_mean_perc),
  post_mean_dec    = unname(post_mean_dec),
  post_sd          = unname(post_sd_),
  ci95_lower_dec   = unname(ci95_dec[, 1]),
  ci95_upper_dec   = unname(ci95_dec[, 2]),
  ess              = unname(ess_par),
  elapsed_seconds  = elapsed,
  stringsAsFactors = FALSE
)

forecast_df <- data.frame(
  section          = "forecast",
  t                = NA_integer_,
  horizon          = seq_len(H_FORECAST),
  date             = fwd_dates,
  y                = NA_real_,
  h_mean_perc      = NA_real_,
  h_mean_dec       = fwd_q50_h,
  h_q05_dec        = fwd_q05_h,
  h_q95_dec        = fwd_q95_h,
  vol_mean_dec     = fwd_q50_v,
  vol_q05_dec      = fwd_q05_v,
  vol_q95_dec      = fwd_q95_v,
  parameter        = NA_character_,
  post_mean_perc   = NA_real_,
  post_mean_dec    = NA_real_,
  post_sd          = NA_real_,
  ci95_lower_dec   = NA_real_,
  ci95_upper_dec   = NA_real_,
  ess              = NA_real_,
  elapsed_seconds  = elapsed,
  stringsAsFactors = FALSE
)

# ------------------------------------------------------------------------------
# 6. Bootstrap PF in R at the SAME plug-in theta used by the Python solution.
#    Python's solution_01_sv_workflow.py uses
#       mu_plugin = 2 * log(sd(returns_decimal))
#       theta_plugin = (mu_plugin, 0.95, 0.20)
#    operating on RAW DECIMAL returns. Replicating this here gives a clean
#    algorithmic cross-check vs Python's `filtered_h_plugin` column.
# ------------------------------------------------------------------------------
mu_plugin_dec <- 2.0 * log(sd(raw_returns))
theta_plugin  <- c(mu_plugin_dec, 0.95, 0.20)
cat(sprintf("\nBootstrap PF at plug-in theta = (%+.4f, %.2f, %.2f) ...\n",
            theta_plugin[1], theta_plugin[2], theta_plugin[3]))
t0 <- Sys.time()
bpf <- bootstrap_pf_sv(raw_returns,
                       mu = theta_plugin[1],
                       phi = theta_plugin[2],
                       sigma = theta_plugin[3],
                       n_particles = 1500L,
                       seed = SEED)
elapsed_bpf <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("  done in %.2f s; loglik=%+.2f, ESS mean/min = %.0f / %.0f\n",
            elapsed_bpf, bpf$log_lik, mean(bpf$ess), min(bpf$ess)))

bpf_filtered_df <- data.frame(
  section          = "filtered_bpf_plugin",
  t                = seq_len(T_eff) - 1L,
  horizon          = NA_integer_,
  date             = dates,
  y                = raw_returns,
  h_mean_perc      = bpf$h_mean + SCALE_LOG,
  h_mean_dec       = bpf$h_mean,
  h_q05_dec        = bpf$h_q05,
  h_q95_dec        = bpf$h_q95,
  vol_mean_dec     = exp(bpf$h_mean / 2.0),
  vol_q05_dec      = exp(bpf$h_q05 / 2.0),
  vol_q95_dec      = exp(bpf$h_q95 / 2.0),
  parameter        = NA_character_,
  post_mean_perc   = NA_real_,
  post_mean_dec    = NA_real_,
  post_sd          = NA_real_,
  ci95_lower_dec   = NA_real_,
  ci95_upper_dec   = NA_real_,
  ess              = bpf$ess,
  elapsed_seconds  = elapsed_bpf,
  stringsAsFactors = FALSE
)

# ------------------------------------------------------------------------------
# 7. Secondary run on simulated_sv.csv (clean ground truth) -- ensures the
#    pipeline recovers known h_true with high correlation, providing an
#    additional cross-validation anchor.
# ------------------------------------------------------------------------------
sim_path <- file.path(script_dir, "..", "data", "simulated_sv.csv")
sim_section_df <- NULL
if (file.exists(sim_path)) {
  cat("\n--- Secondary run on simulated_sv.csv (T=200, matched scale) ---\n")
  sim_df <- read.csv(sim_path, stringsAsFactors = FALSE)
  T_sim  <- min(200L, nrow(sim_df))
  y_sim  <- as.numeric(sim_df$y_obs[1:T_sim])
  h_true_sim <- as.numeric(sim_df$h_true[1:T_sim])

  set.seed(SEED + 2L)
  fit_sim <- svsample(
    y          = y_sim,
    draws      = DRAWS,
    burnin     = BURNIN,
    priormu    = c(-1, 1),
    priorphi   = c(20, 1.5),
    priorsigma = 0.5,
    quiet      = TRUE
  )
  latent_sim <- as.matrix(fit_sim$latent[[1]])
  T_sim_eff  <- min(T_sim, ncol(latent_sim))
  h_sim_mean <- colMeans(latent_sim[, 1:T_sim_eff])
  h_sim_q05  <- apply(latent_sim[, 1:T_sim_eff], 2, quantile, 0.05)
  h_sim_q95  <- apply(latent_sim[, 1:T_sim_eff], 2, quantile, 0.95)

  cor_h_true <- suppressWarnings(cor(h_sim_mean, h_true_sim[1:T_sim_eff]))
  cat(sprintf("  cor(h_R_smooth, h_true) = %.4f\n", cor_h_true))

  sim_section_df <- data.frame(
    section          = "filtered_simulated",
    t                = seq_len(T_sim_eff) - 1L,
    horizon          = NA_integer_,
    date             = NA_character_,
    y                = y_sim[1:T_sim_eff],
    h_mean_perc      = h_sim_mean,             # already on simulated (no rescale)
    h_mean_dec       = h_sim_mean,
    h_q05_dec        = h_sim_q05,
    h_q95_dec        = h_sim_q95,
    vol_mean_dec     = exp(h_sim_mean / 2.0),
    vol_q05_dec      = exp(h_sim_q05 / 2.0),
    vol_q95_dec      = exp(h_sim_q95 / 2.0),
    parameter        = NA_character_,
    post_mean_perc   = NA_real_,
    post_mean_dec    = NA_real_,
    post_sd          = NA_real_,
    ci95_lower_dec   = NA_real_,
    ci95_upper_dec   = NA_real_,
    ess              = NA_real_,
    elapsed_seconds  = NA_real_,
    stringsAsFactors = FALSE
  )
  # Append posterior summary on simulated dataset under a distinct section
  para_sim <- as.data.frame(fit_sim$para[[1]])
  colnames(para_sim) <- tolower(colnames(para_sim))
  sim_params_df <- data.frame(
    section          = "parameters_simulated",
    t                = NA_integer_,
    horizon          = NA_integer_,
    date             = NA_character_,
    y                = NA_real_,
    h_mean_perc      = NA_real_,
    h_mean_dec       = NA_real_,
    h_q05_dec        = NA_real_,
    h_q95_dec        = NA_real_,
    vol_mean_dec     = NA_real_,
    vol_q05_dec      = NA_real_,
    vol_q95_dec      = NA_real_,
    parameter        = c("mu", "phi", "sigma_h"),
    post_mean_perc   = c(mean(para_sim$mu), mean(para_sim$phi), mean(para_sim$sigma)),
    post_mean_dec    = c(mean(para_sim$mu), mean(para_sim$phi), mean(para_sim$sigma)),
    post_sd          = c(sd(para_sim$mu),   sd(para_sim$phi),   sd(para_sim$sigma)),
    ci95_lower_dec   = c(quantile(para_sim$mu, 0.025),
                         quantile(para_sim$phi, 0.025),
                         quantile(para_sim$sigma, 0.025)),
    ci95_upper_dec   = c(quantile(para_sim$mu, 0.975),
                         quantile(para_sim$phi, 0.975),
                         quantile(para_sim$sigma, 0.975)),
    ess              = if (has_coda) c(effectiveSize(para_sim$mu),
                                       effectiveSize(para_sim$phi),
                                       effectiveSize(para_sim$sigma))
                       else rep(NA_real_, 3),
    elapsed_seconds  = NA_real_,
    stringsAsFactors = FALSE
  )
  sim_section_df <- rbind(sim_section_df, sim_params_df)
}

combined <- rbind(filtered_df, params_df, forecast_df, bpf_filtered_df)
if (!is.null(sim_section_df)) combined <- rbind(combined, sim_section_df)
write.csv(combined, output_csv, row.names = FALSE)
cat(sprintf("\nResults saved (%d rows) -> %s\n", nrow(combined), output_csv))
cat("Done.\n")
