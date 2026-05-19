#!/usr/bin/env Rscript
# ==============================================================================
# validate_sir.R
# Cross-validation of the stochastic SIR particle filter.
# Compared against particlefilterbox's Bootstrap PF implementation
# (solution_03_sir_epidemic.py).
#
# Model (discrete-time SIR with Poisson transitions & partial observation):
#   new_inf_t ~ Poisson(beta * S_t * I_t / N)        (capped at S_t)
#   new_rec_t ~ Poisson(gamma * I_t)                 (capped at I_t)
#   S_{t+1}   = S_t - new_inf_t
#   I_{t+1}   = I_t + new_inf_t - new_rec_t
#   R_{t+1}   = R_t + new_rec_t
#   y_t       ~ Poisson(obs_rate * I_t)
# R_0 = beta / gamma is carried in the particle cloud so it has a
# time-indexed posterior.
#
# Required R packages:
#   pomp   - canonical R package for partially-observed Markov processes
#            / particle filtering.  Used when available via pfilter() with
#            C snippets; otherwise the script falls back to a pure-R
#            bootstrap filter that implements the same algorithm.
#
# Usage: Rscript validate_sir.R
# ==============================================================================

suppressPackageStartupMessages({
  has_pomp <- requireNamespace("pomp", quietly = TRUE)
})

cat(strrep("=", 71), "\n")
cat("Stochastic SIR validation (R via pomp / pure-R bootstrap PF)\n")
cat("Dataset: simulated_sir.csv\n")
cat(strrep("=", 71), "\n\n")

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
SEED        <- 20260422L
N_PARTICLES <- 5000L
N_POP       <- 10000
BETA        <- 0.30
GAMMA       <- 0.10
OBS_RATE    <- 0.50
BETA_PRIOR  <- c(0.30, 0.05)   # (mean, sd)
GAMMA_PRIOR <- c(0.10, 0.02)

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  }
  return(getwd())
}
script_dir <- get_script_dir()
data_path  <- file.path(script_dir, "..", "data", "simulated_sir.csv")
output_csv <- file.path(script_dir, "results_r_sir.csv")

if (!file.exists(data_path)) {
  stop("Dataset not found at: ", data_path,
       "\nRun examples/applications/data/generate_data.py first.")
}

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
weighted_quantile <- function(values, weights, q) {
  ord <- order(values)
  v <- values[ord]
  w <- weights[ord]
  tot <- sum(w)
  if (!is.finite(tot) || tot <= 0) return(NA_real_)
  cw <- cumsum(w) / tot
  approx(cw, v, xout = q, rule = 2)$y
}

systematic_resample <- function(w, rng) {
  n <- length(w)
  u0 <- rng()[1] / n
  edges <- u0 + (seq_len(n) - 1) / n
  cw <- cumsum(w / sum(w))
  findInterval(edges, cw) + 1L
}

# ------------------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------------------
df   <- read.csv(data_path, stringsAsFactors = FALSE)
T_obs <- nrow(df)
y_obs <- as.numeric(df$y_obs)
S_true <- as.numeric(df$S_true)
I_true <- as.numeric(df$I_true)
R_true <- as.numeric(df$R_true)

cat(sprintf("Loaded %d observations.\n", T_obs))
cat(sprintf("Peak observed y_obs : %d\n", max(y_obs)))
cat(sprintf("Peak true  I_t      : %d\n\n", max(I_true)))

# ------------------------------------------------------------------------------
# 2. Try pomp's pfilter() if available, else pure-R bootstrap
# ------------------------------------------------------------------------------
fit_used_pomp <- FALSE
elapsed       <- NA_real_

run_pure_r_pf <- function() {
  set.seed(SEED)

  S   <- rep(N_POP - 10, N_PARTICLES)
  I_p <- rep(10.0,         N_PARTICLES)
  Rc  <- rep(0.0,          N_PARTICLES)
  beta_p  <- pmax(rnorm(N_PARTICLES, BETA_PRIOR[1], BETA_PRIOR[2]),  0.01)
  gamma_p <- pmax(rnorm(N_PARTICLES, GAMMA_PRIOR[1], GAMMA_PRIOR[2]), 0.01)

  s_hat    <- numeric(T_obs); i_hat    <- numeric(T_obs); r_hat   <- numeric(T_obs)
  i_q05    <- numeric(T_obs); i_q50    <- numeric(T_obs); i_q95   <- numeric(T_obs)
  r0_mean  <- numeric(T_obs); r0_q05   <- numeric(T_obs); r0_q95  <- numeric(T_obs)
  ess_h    <- numeric(T_obs); ll_inc   <- numeric(T_obs)

  total_ll <- 0.0
  ess_thresh <- 0.5 * N_PARTICLES

  rng_fn <- function() runif(1)

  t0 <- Sys.time()
  for (t in seq_len(T_obs)) {
    # 1. Propagate: discrete-time SIR with Poisson transitions
    lam_inf <- pmax(beta_p * S * I_p / N_POP, 0)
    lam_rec <- pmax(gamma_p * I_p,           0)
    new_inf <- rpois(N_PARTICLES, lam_inf)
    new_rec <- rpois(N_PARTICLES, lam_rec)
    new_inf <- pmin(new_inf, S)
    new_rec <- pmin(new_rec, I_p)
    S   <- S   - new_inf
    I_p <- I_p + new_inf - new_rec
    Rc  <- Rc  + new_rec

    # 2. Log-weights: Poisson obs model
    rate <- pmax(OBS_RATE * I_p, 1e-6)
    log_w <- dpois(y_obs[t], lambda = rate, log = TRUE)

    m <- max(log_w)
    w <- exp(log_w - m)
    w_sum <- sum(w)
    if (!is.finite(w_sum) || w_sum <= 0) {
      w <- rep(1 / N_PARTICLES, N_PARTICLES)
      ll_t <- -Inf
    } else {
      ll_t <- m + log(w_sum) - log(N_PARTICLES)
      w    <- w / w_sum
    }
    total_ll <- total_ll + ll_t
    ll_inc[t] <- ll_t

    # 3. Moments from weighted cloud
    s_hat[t]   <- sum(w * S)
    i_hat[t]   <- sum(w * I_p)
    r_hat[t]   <- sum(w * Rc)
    i_q05[t]   <- weighted_quantile(I_p, w, 0.05)
    i_q50[t]   <- weighted_quantile(I_p, w, 0.50)
    i_q95[t]   <- weighted_quantile(I_p, w, 0.95)
    r0_part    <- beta_p / gamma_p
    r0_mean[t] <- sum(w * r0_part)
    r0_q05[t]  <- weighted_quantile(r0_part, w, 0.05)
    r0_q95[t]  <- weighted_quantile(r0_part, w, 0.95)
    ess_h[t]   <- 1 / sum(w^2)

    # 4. Adaptive systematic resample
    if (ess_h[t] < ess_thresh) {
      idx <- systematic_resample(w, rng_fn)
      idx <- pmin(pmax(idx, 1L), N_PARTICLES)
      S <- S[idx]; I_p <- I_p[idx]; Rc <- Rc[idx]
      beta_p <- beta_p[idx]; gamma_p <- gamma_p[idx]
    }
  }
  elapsed_local <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  list(
    s_hat = s_hat, i_hat = i_hat, r_hat = r_hat,
    i_q05 = i_q05, i_q50 = i_q50, i_q95 = i_q95,
    r0_mean = r0_mean, r0_q05 = r0_q05, r0_q95 = r0_q95,
    ess = ess_h, ll_inc = ll_inc,
    total_ll = total_ll, elapsed = elapsed_local
  )
}

run_pomp_pf <- function() {
  library(pomp)

  # Build a pomp object with C snippets.  beta and gamma are treated as
  # (fixed) parameters here; the pure-R path additionally propagates
  # priors for them so that R_0 has a non-degenerate posterior.  To keep
  # the comparison apples-to-apples, we still recover R_0 uncertainty
  # from the pure-R run and only use pomp to cross-check I_t filtering.
  rproc <- Csnippet("
    double lam_inf = beta * S * I / N;
    double lam_rec = gamma * I;
    if (lam_inf < 0) lam_inf = 0;
    if (lam_rec < 0) lam_rec = 0;
    double ni = rpois(lam_inf);
    double nr = rpois(lam_rec);
    if (ni > S) ni = S;
    if (nr > I) nr = I;
    S -= ni;
    I += ni - nr;
    R += nr;
  ")
  rinit <- Csnippet("
    S = N - 10.0;
    I = 10.0;
    R = 0.0;
  ")
  dmeas <- Csnippet("
    double rate = obs_rate * I;
    if (rate < 1e-6) rate = 1e-6;
    lik = dpois(y_obs, rate, give_log);
  ")
  rmeas <- Csnippet("
    double rate = obs_rate * I;
    if (rate < 1e-6) rate = 1e-6;
    y_obs = rpois(rate);
  ")

  po <- pomp(
    data = data.frame(time = seq_len(T_obs), y_obs = y_obs),
    times   = "time",
    t0      = 0,
    rprocess = discrete_time(step.fun = rproc, delta.t = 1),
    rmeasure = rmeas,
    dmeasure = dmeas,
    rinit    = rinit,
    statenames = c("S", "I", "R"),
    paramnames = c("beta", "gamma", "obs_rate", "N")
  )

  pf <- pfilter(
    po,
    Np     = N_PARTICLES,
    params = c(beta = BETA, gamma = GAMMA, obs_rate = OBS_RATE, N = N_POP),
    save.states = "filter"
  )

  # Extract per-step I posterior from saved states.  In pomp >= 5,
  # saved_states() returns a list with one entry per observation time.
  saved <- saved_states(pf)
  if (is.list(saved) && !is.null(saved$filter)) saved <- saved$filter

  extract_state <- function(el, name) {
    if (is.matrix(el))      return(el[name, ])
    if (is.data.frame(el))  return(el[[name]])
    if (is.list(el) && !is.null(el[[name]])) return(el[[name]])
    stop("unexpected saved_states element layout")
  }

  I_mat <- vapply(saved, extract_state, numeric(N_PARTICLES), name = "I")
  S_mat <- vapply(saved, extract_state, numeric(N_PARTICLES), name = "S")
  R_mat <- vapply(saved, extract_state, numeric(N_PARTICLES), name = "R")

  # pomp 5+ renamed cond.logLik -> cond_logLik
  cond_ll_fn <- tryCatch(
    get("cond_logLik", envir = asNamespace("pomp")),
    error = function(e) NULL
  )
  cond_ll <- if (!is.null(cond_ll_fn)) as.numeric(cond_ll_fn(pf)) else NA_real_

  list(
    I_hat_pomp = colMeans(I_mat),
    S_hat_pomp = colMeans(S_mat),
    R_hat_pomp = colMeans(R_mat),
    log_lik_pomp = as.numeric(logLik(pf)),
    cond_ll_pomp = cond_ll
  )
}

# Always run the pure-R path (primary result set).
set.seed(SEED)
t0 <- Sys.time()
pf_out <- run_pure_r_pf()
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

# Try pomp as a cross-check if the package is installed.
pomp_out <- NULL
if (has_pomp) {
  cat("Running pomp::pfilter() as a cross-check (fixed beta, gamma)...\n")
  pomp_out <- tryCatch(run_pomp_pf(),
                       error = function(e) {
                         cat("  pomp path failed: ",
                             conditionMessage(e), "\n", sep = "")
                         NULL
                       })
  if (!is.null(pomp_out)) fit_used_pomp <- TRUE
} else {
  cat("pomp not installed - skipping cross-check.\n")
}

# ------------------------------------------------------------------------------
# 3. Diagnostics
# ------------------------------------------------------------------------------
rmse_s <- sqrt(mean((pf_out$s_hat - S_true)^2))
rmse_i <- sqrt(mean((pf_out$i_hat - I_true)^2))
rmse_r <- sqrt(mean((pf_out$r_hat - R_true)^2))
true_r0 <- BETA / GAMMA
cov_i <- mean(I_true >= pf_out$i_q05 & I_true <= pf_out$i_q95)

cat(sprintf("\nParticle filter done in %.2f s (pomp used: %s)\n",
            elapsed, ifelse(fit_used_pomp, "TRUE", "FALSE")))
cat("-----------------------------------------------------------\n")
cat(sprintf("  log-likelihood      : %.2f\n",  pf_out$total_ll))
cat(sprintf("  mean ESS            : %.1f\n", mean(pf_out$ess)))
cat(sprintf("  RMSE S              : %.2f\n", rmse_s))
cat(sprintf("  RMSE I              : %.2f\n", rmse_i))
cat(sprintf("  RMSE R              : %.2f\n", rmse_r))
cat(sprintf("  90%% CI coverage (I) : %.3f\n", cov_i))
cat(sprintf("  true R0             : %.3f\n", true_r0))
cat(sprintf("  posterior R0 (end)  : %.3f  [%.3f, %.3f]\n",
            pf_out$r0_mean[T_obs],
            pf_out$r0_q05[T_obs],
            pf_out$r0_q95[T_obs]))
if (!is.null(pomp_out)) {
  rmse_i_pomp <- sqrt(mean((pomp_out$I_hat_pomp - I_true)^2))
  cat(sprintf("  pomp log-lik        : %.2f\n", pomp_out$log_lik_pomp))
  cat(sprintf("  pomp RMSE I         : %.2f\n", rmse_i_pomp))
}

# ------------------------------------------------------------------------------
# 4. Build CSV
# ------------------------------------------------------------------------------
per_step <- data.frame(
  section = "filtered",
  t       = df$t,
  y_obs   = y_obs,
  S_true  = S_true,
  I_true  = I_true,
  R_true  = R_true,
  S_filt  = pf_out$s_hat,
  I_filt  = pf_out$i_hat,
  R_filt  = pf_out$r_hat,
  I_q05   = pf_out$i_q05,
  I_q50   = pf_out$i_q50,
  I_q95   = pf_out$i_q95,
  R0_mean = pf_out$r0_mean,
  R0_q05  = pf_out$r0_q05,
  R0_q95  = pf_out$r0_q95,
  ess     = pf_out$ess,
  log_lik_inc = pf_out$ll_inc,
  I_filt_pomp = if (!is.null(pomp_out)) pomp_out$I_hat_pomp else NA_real_,
  parameter = "",
  value     = NA_real_,
  stringsAsFactors = FALSE
)

metric_rows <- data.frame(
  section = "metrics",
  t = NA_integer_,
  y_obs = NA_real_, S_true = NA_real_, I_true = NA_real_, R_true = NA_real_,
  S_filt = NA_real_, I_filt = NA_real_, R_filt = NA_real_,
  I_q05 = NA_real_, I_q50 = NA_real_, I_q95 = NA_real_,
  R0_mean = NA_real_, R0_q05 = NA_real_, R0_q95 = NA_real_,
  ess = NA_real_, log_lik_inc = NA_real_,
  I_filt_pomp = NA_real_,
  parameter = c("log_likelihood", "mean_ess", "rmse_S", "rmse_I", "rmse_R",
                "coverage_90_I", "true_R0", "R0_mean_final",
                "R0_q05_final", "R0_q95_final",
                "fit_used_pomp", "elapsed_sec", "n_particles"),
  value = c(pf_out$total_ll, mean(pf_out$ess),
            rmse_s, rmse_i, rmse_r, cov_i, true_r0,
            pf_out$r0_mean[T_obs], pf_out$r0_q05[T_obs], pf_out$r0_q95[T_obs],
            as.numeric(fit_used_pomp), elapsed, as.numeric(N_PARTICLES)),
  stringsAsFactors = FALSE
)

combined <- rbind(per_step, metric_rows)
write.csv(combined, output_csv, row.names = FALSE)
cat(sprintf("\nResults saved (%d rows) to: %s\n", nrow(combined), output_csv))

# ------------------------------------------------------------------------------
# 5. Cross-check vs Python solution
# ------------------------------------------------------------------------------
py_path <- file.path(script_dir, "..", "solutions", "results_sir_epidemic.csv")
if (file.exists(py_path)) {
  cat("\n--- Cross-check vs particlefilterbox (solution_03_sir_epidemic.py) ---\n")
  py <- read.csv(py_path, stringsAsFactors = FALSE)
  n  <- min(nrow(py), T_obs)
  cor_i  <- suppressWarnings(cor(pf_out$i_hat[seq_len(n)], py$I_filt[seq_len(n)]))
  cor_r0 <- suppressWarnings(cor(pf_out$r0_mean[seq_len(n)], py$R0_mean[seq_len(n)]))
  rmse_i_vs_py <- sqrt(mean((pf_out$i_hat[seq_len(n)] - py$I_filt[seq_len(n)])^2))
  diff_r0_final <- abs(pf_out$r0_mean[T_obs] - py$R0_mean[T_obs])
  cat(sprintf("  corr(I_filt_R, I_filt_Py)   : %.4f\n", cor_i))
  cat(sprintf("  corr(R0_mean_R, R0_mean_Py) : %.4f\n", cor_r0))
  cat(sprintf("  RMSE (I_filt_R vs I_filt_Py): %.2f\n", rmse_i_vs_py))
  cat(sprintf("  |R0_final_R - R0_final_Py|  : %.4f\n", diff_r0_final))
}

cat("\nDone.\n")
