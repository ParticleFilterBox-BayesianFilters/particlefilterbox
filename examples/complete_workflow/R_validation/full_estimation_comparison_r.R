#!/usr/bin/env Rscript
# ==============================================================================
# full_estimation_comparison_r.R
#
# Cross-validation of the SMC / PMMH / PGAS comparison from
# `solutions/solution_02_estimation_comparison.py`. We reuse the same simulated
# dataset (simulated_sv.csv, T=200, true theta = (-1.0, 0.97, 0.15)) and
# estimate the basic SV model with:
#
#   1. stochvol::svsample            -- multimove MCMC  (matches Python "PGAS")
#   2. pomp::pmcmc                    -- particle MMH    (matches Python "PMMH")
#   3. (optional) SMC::*              -- if the `SMC` package is available we
#                                       run an SMC-sampler analogue; otherwise
#                                       the row is emitted with NA metrics.
#
# For each (method, parameter) we report:
#   posterior_mean / std / 95%CI / bias / abs_bias / sq_error / within_ci
#   ess / time_seconds / log_evidence (when the method produces it)
#
# Output: results_r_estimation.csv (long format, matches Python's
# results_estimation_comparison.csv schema for direct cross-CSV comparison).
#
# Required: stochvol, pomp; optional: SMC, coda
# Usage:    Rscript full_estimation_comparison_r.R
# ==============================================================================

suppressPackageStartupMessages({
  has_stochvol <- requireNamespace("stochvol", quietly = TRUE)
  has_pomp     <- requireNamespace("pomp",     quietly = TRUE)
  has_SMC      <- requireNamespace("SMC",      quietly = TRUE)
  has_coda     <- requireNamespace("coda",     quietly = TRUE)
})

cat(strrep("=", 78), "\n", sep = "")
cat("FASE 9.4 / full_estimation_comparison_r.R\n")
cat("Methods: stochvol-MCMC, pomp-PMMH",
    if (has_SMC) ", SMC::*" else " (SMC package NOT installed)", "\n", sep = "")
cat(strrep("=", 78), "\n\n", sep = "")

# ------------------------------------------------------------------------------
# Helpers / config
# ------------------------------------------------------------------------------
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", args, value = TRUE)
  if (length(fa) > 0) return(dirname(normalizePath(sub("^--file=", "", fa[1]))))
  getwd()
}
script_dir <- get_script_dir()

SEED         <- 42L
T_USE        <- 200L
TRUE_MU      <- -1.0
TRUE_PHI     <-  0.97
TRUE_SIGMA   <-  0.15
THETA_TRUE   <- c(mu = TRUE_MU, phi = TRUE_PHI, sigma_h = TRUE_SIGMA)
PARAM_NAMES  <- c("mu", "phi", "sigma_h")

# Sampler tuning -- conservative to keep total runtime O(1 minute).
DRAWS_STOCHVOL  <- 4000L
BURN_STOCHVOL   <- 1000L
NMCMC_PMMH      <- 800L
NP_PMMH         <- 200L
BURN_PMMH       <- 200L
N_SMC_PARTICLES <- 300L     # only used if SMC package available

output_csv <- file.path(script_dir, "results_r_estimation.csv")

# Column schema (matches Python CSV; one row per (method, param) plus
# a global row with param = "all" carrying summary statistics).
COL_NAMES <- c("method", "param", "true",
               "posterior_mean", "posterior_std",
               "ci95_lower", "ci95_upper",
               "bias", "abs_bias", "sq_error", "within_ci",
               "ess", "time_seconds", "log_evidence")

mk_param_row <- function(method, name, true_val, post_mean, post_std,
                         ci_lo, ci_hi, ess, elapsed, log_evid = NA_real_) {
  in_ci <- !is.na(ci_lo) && !is.na(ci_hi) && (true_val >= ci_lo) && (true_val <= ci_hi)
  data.frame(
    method = method, param = name, true = true_val,
    posterior_mean = post_mean, posterior_std = post_std,
    ci95_lower = ci_lo, ci95_upper = ci_hi,
    bias = post_mean - true_val,
    abs_bias = abs(post_mean - true_val),
    sq_error = (post_mean - true_val)^2,
    within_ci = in_ci,
    ess = ess, time_seconds = elapsed, log_evidence = log_evid,
    stringsAsFactors = FALSE
  )
}

mk_summary_row <- function(method, post_means, ess_vec, elapsed, log_evid = NA_real_) {
  bias_avg <- mean(post_means - THETA_TRUE)
  abs_bias_avg <- mean(abs(post_means - THETA_TRUE))
  rmse_sq <- mean((post_means - THETA_TRUE)^2)
  ess_min <- if (all(is.na(ess_vec))) NA_real_ else suppressWarnings(min(ess_vec, na.rm = TRUE))
  data.frame(
    method = method, param = "all", true = NA_real_,
    posterior_mean = NA_real_, posterior_std = NA_real_,
    ci95_lower = NA_real_, ci95_upper = NA_real_,
    bias = bias_avg, abs_bias = abs_bias_avg, sq_error = rmse_sq,
    within_ci = NA, ess = ess_min,
    time_seconds = elapsed, log_evidence = log_evid,
    stringsAsFactors = FALSE
  )
}

empty_method_rows <- function(method, reason = NA_character_) {
  rows <- do.call(rbind, lapply(seq_along(PARAM_NAMES), function(i) {
    mk_param_row(method, PARAM_NAMES[i], THETA_TRUE[i],
                 NA_real_, NA_real_, NA_real_, NA_real_,
                 NA_real_, NA_real_, NA_real_)
  }))
  rows <- rbind(rows, mk_summary_row(method,
                                     rep(NA_real_, 3), rep(NA_real_, 3),
                                     NA_real_, NA_real_))
  attr(rows, "reason") <- reason
  rows
}

# ------------------------------------------------------------------------------
# 1. Load simulated dataset (T=200) -- matches Python solution_02
# ------------------------------------------------------------------------------
data_path <- file.path(script_dir, "..", "data", "simulated_sv.csv")
if (!file.exists(data_path)) stop("Simulated dataset not found at: ", data_path)

sim_df <- read.csv(data_path, stringsAsFactors = FALSE)
T_eff  <- min(T_USE, nrow(sim_df))
y_obs  <- as.numeric(sim_df$y_obs[1:T_eff])
h_true <- as.numeric(sim_df$h_true[1:T_eff])

cat(sprintf("Loaded %d observations from simulated_sv.csv\n", T_eff))
cat(sprintf("  true theta = (mu=%.4f, phi=%.4f, sigma=%.4f)\n",
            TRUE_MU, TRUE_PHI, TRUE_SIGMA))
cat(sprintf("  y_obs    : mean=%+.4f  sd=%.4f\n", mean(y_obs), sd(y_obs)))
cat(sprintf("  h_true   : mean=%+.4f  sd=%.4f\n\n", mean(h_true), sd(h_true)))

all_rows <- list()

# ==============================================================================
# Method 1: stochvol::svsample  (multimove MCMC -- "PGAS-like" gold standard)
# ==============================================================================
cat(strrep("-", 78), "\n", sep = "")
cat(sprintf("[1/3] stochvol::svsample (draws=%d, burnin=%d)\n",
            DRAWS_STOCHVOL, BURN_STOCHVOL))
cat(strrep("-", 78), "\n", sep = "")

if (!has_stochvol) {
  cat("stochvol not installed -- emitting empty rows.\n")
  all_rows[["stochvol"]] <- empty_method_rows("stochvol", "package not installed")
} else {
  library(stochvol); if (has_coda) library(coda)
  set.seed(SEED)
  t0 <- Sys.time()
  fit_sv <- svsample(
    y          = y_obs,
    draws      = DRAWS_STOCHVOL,
    burnin     = BURN_STOCHVOL,
    priormu    = c(-1, 1),       # informative on mu (matches Python prior)
    priorphi   = c(20, 1.5),     # high-persistence prior
    priorsigma = 0.5,            # tight scale prior
    quiet      = TRUE
  )
  elapsed_sv <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat(sprintf("  finished in %.2f s\n", elapsed_sv))
  pd <- as.data.frame(fit_sv$para[[1]])
  colnames(pd) <- tolower(colnames(pd))

  sv_mean <- c(mu = mean(pd$mu), phi = mean(pd$phi), sigma_h = mean(pd$sigma))
  sv_std  <- c(mu = sd(pd$mu),   phi = sd(pd$phi),   sigma_h = sd(pd$sigma))
  sv_ci_lo <- c(quantile(pd$mu, 0.025),
                quantile(pd$phi, 0.025),
                quantile(pd$sigma, 0.025))
  sv_ci_hi <- c(quantile(pd$mu, 0.975),
                quantile(pd$phi, 0.975),
                quantile(pd$sigma, 0.975))
  sv_ess <- if (has_coda) {
    c(effectiveSize(pd$mu), effectiveSize(pd$phi), effectiveSize(pd$sigma))
  } else rep(NA_real_, 3)

  for (i in seq_along(PARAM_NAMES)) {
    cat(sprintf("  %-7s  mean=%+.4f  std=%.4f  CI95=[%+.4f, %+.4f]  ESS=%.0f\n",
                PARAM_NAMES[i], sv_mean[i], sv_std[i], sv_ci_lo[i], sv_ci_hi[i],
                sv_ess[i]))
  }
  rows_sv <- do.call(rbind, lapply(seq_along(PARAM_NAMES), function(i) {
    mk_param_row("stochvol", PARAM_NAMES[i], unname(THETA_TRUE[i]),
                 unname(sv_mean[i]), unname(sv_std[i]),
                 unname(sv_ci_lo[i]), unname(sv_ci_hi[i]),
                 unname(sv_ess[i]), elapsed_sv)
  }))
  rows_sv <- rbind(rows_sv,
                   mk_summary_row("stochvol", unname(sv_mean), unname(sv_ess),
                                  elapsed_sv))
  all_rows[["stochvol"]] <- rows_sv
}

# ==============================================================================
# Method 2: pomp::pmcmc  (PMMH)
# ==============================================================================
cat("\n", strrep("-", 78), "\n", sep = "")
cat(sprintf("[2/3] pomp::pmcmc (Nmcmc=%d, Np=%d, burn=%d)\n",
            NMCMC_PMMH, NP_PMMH, BURN_PMMH))
cat(strrep("-", 78), "\n", sep = "")

if (!has_pomp) {
  cat("pomp not installed -- emitting empty rows.\n")
  all_rows[["pomp"]] <- empty_method_rows("pomp_pmmh", "package not installed")
} else {
  library(pomp)

  sv_pomp <- pomp(
    data = data.frame(time = 1:T_eff, y = y_obs),
    times = "time", t0 = 0,
    rprocess = discrete_time(
      step.fun = Csnippet("h = mu + phi * (h - mu) + sigma * rnorm(0, 1);"),
      delta.t = 1
    ),
    rinit = Csnippet("h = mu;"),
    rmeasure = Csnippet("y = exp(h / 2.0) * rnorm(0, 1);"),
    dmeasure = Csnippet(
      "lik = dnorm(y, 0.0, exp(h / 2.0), give_log);
       if (!give_log) lik = exp(lik);"
    ),
    dprior = Csnippet(
      "lik = dnorm(mu, -1.0, 1.0, 1) +
              dbeta(phi, 20.0, 1.5, 1) +
              dgamma(sigma, 2.5, 0.025, 1);
       if (!give_log) lik = exp(lik);"
    ),
    statenames  = "h",
    paramnames  = c("mu", "phi", "sigma"),
    partrans    = parameter_trans(log = "sigma", logit = "phi")
  )

  set.seed(SEED)
  t0 <- Sys.time()
  pmmh_fit <- tryCatch(
    pmcmc(
      sv_pomp,
      Nmcmc    = NMCMC_PMMH,
      Np       = NP_PMMH,
      proposal = mvn_diag_rw(c(mu = 0.04, phi = 0.005, sigma = 0.012)),
      params   = c(mu = TRUE_MU, phi = TRUE_PHI, sigma = TRUE_SIGMA),
      verbose  = FALSE
    ),
    error = function(e) { cat("  pmcmc failed: ", conditionMessage(e), "\n"); NULL }
  )
  elapsed_pmmh <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat(sprintf("  finished in %.2f s\n", elapsed_pmmh))

  if (is.null(pmmh_fit)) {
    all_rows[["pomp"]] <- empty_method_rows("pomp_pmmh", "pmcmc threw an error")
  } else {
    tr <- traces(pmmh_fit)            # mcmc object
    tr_mat <- as.matrix(tr)
    keep_iters <- (BURN_PMMH + 1):nrow(tr_mat)
    if (length(keep_iters) < 50) keep_iters <- 1:nrow(tr_mat)
    mu_chain   <- tr_mat[keep_iters, "mu"]
    phi_chain  <- tr_mat[keep_iters, "phi"]
    sig_chain  <- tr_mat[keep_iters, "sigma"]

    pm_mean <- c(mu = mean(mu_chain), phi = mean(phi_chain), sigma_h = mean(sig_chain))
    pm_std  <- c(mu = sd(mu_chain),   phi = sd(phi_chain),   sigma_h = sd(sig_chain))
    pm_lo   <- c(quantile(mu_chain, 0.025),
                 quantile(phi_chain, 0.025),
                 quantile(sig_chain, 0.025))
    pm_hi   <- c(quantile(mu_chain, 0.975),
                 quantile(phi_chain, 0.975),
                 quantile(sig_chain, 0.975))
    pm_ess <- if (has_coda) {
      c(effectiveSize(mu_chain), effectiveSize(phi_chain), effectiveSize(sig_chain))
    } else rep(NA_real_, 3)

    for (i in seq_along(PARAM_NAMES)) {
      cat(sprintf("  %-7s  mean=%+.4f  std=%.4f  CI95=[%+.4f, %+.4f]  ESS=%.1f\n",
                  PARAM_NAMES[i], pm_mean[i], pm_std[i], pm_lo[i], pm_hi[i], pm_ess[i]))
    }
    rows_pm <- do.call(rbind, lapply(seq_along(PARAM_NAMES), function(i) {
      mk_param_row("pomp_pmmh", PARAM_NAMES[i], unname(THETA_TRUE[i]),
                   unname(pm_mean[i]), unname(pm_std[i]),
                   unname(pm_lo[i]), unname(pm_hi[i]),
                   unname(pm_ess[i]), elapsed_pmmh)
    }))
    rows_pm <- rbind(rows_pm,
                     mk_summary_row("pomp_pmmh", unname(pm_mean),
                                    unname(pm_ess), elapsed_pmmh))
    all_rows[["pomp"]] <- rows_pm
  }
}

# ==============================================================================
# Method 3: SMC::* (only if installed) -- otherwise emit empty rows.
#
# We try `SMC::AdPMCMC` / `SMC::SISR` if available, else write NA rows so the
# cross_validation_report can render the SMC line uniformly with the Python
# CSV's three-method layout.
# ==============================================================================
cat("\n", strrep("-", 78), "\n", sep = "")
cat("[3/3] SMC sampler (R package 'SMC' -- optional)\n")
cat(strrep("-", 78), "\n", sep = "")

if (has_SMC) {
  library(SMC)
  cat("SMC package detected. Note: there is no canonical SV sampler in SMC,\n",
      "so we run a tempered IS sampler with the bootstrap-PF likelihood.\n",
      sep = "")
  # Implement a minimal IBIS-style SMC sampler in pure R using a bootstrap PF
  # for the marginal likelihood. We avoid relying on a specific function from
  # the SMC package (its API varies across versions) and use the package as a
  # "presence flag" to align with the Python pipeline naming.
  bootstrap_pf <- function(theta, y, np = 80) {
    mu <- theta[1]; phi <- theta[2]; sig <- theta[3]
    if (abs(phi) >= 1 || sig <= 0) return(-Inf)
    var_stat <- sig^2 / (1 - phi^2)
    parts <- rnorm(np, mu, sqrt(max(var_stat, 1e-10)))
    ll <- 0
    for (t in seq_along(y)) {
      vol <- exp(parts / 2)
      lw <- -0.5 * log(2 * pi) - log(vol) - 0.5 * (y[t] / vol)^2
      m <- max(lw); w <- exp(lw - m); s <- sum(w)
      if (s < 1e-300) return(-Inf)
      ll <- ll + m + log(s) - log(np)
      w <- w / s
      idx <- sample.int(np, np, replace = TRUE, prob = w)
      parts <- parts[idx]
      parts <- mu + phi * (parts - mu) + sig * rnorm(np)
    }
    ll
  }
  prior_sample <- function(n) {
    cbind(mu = rnorm(n, -1, 1),
          phi = rbeta(n, 20, 1.5),
          sigma_h = 1 / rgamma(n, 2.5, scale = 1 / 0.025))
  }
  prior_logpdf <- function(theta) {
    mu <- theta[1]; phi <- theta[2]; sig <- theta[3]
    if (phi <= 0 || phi >= 1 || sig <= 0) return(-Inf)
    dnorm(mu, -1, 1, log = TRUE) +
      dbeta(phi, 20, 1.5, log = TRUE) +
      dgamma(1 / sig, shape = 2.5, scale = 1 / 0.025, log = TRUE) -
      2 * log(sig)
  }

  set.seed(SEED)
  t0 <- Sys.time()
  N_PARTS <- N_SMC_PARTICLES
  particles <- prior_sample(N_PARTS)
  log_lik <- vapply(seq_len(N_PARTS),
                    function(i) bootstrap_pf(particles[i, ], y_obs, np = 60),
                    numeric(1))
  log_prior <- vapply(seq_len(N_PARTS),
                      function(i) prior_logpdf(particles[i, ]), numeric(1))
  # Tempering
  betas <- c(0.0, 0.05, 0.15, 0.35, 0.65, 1.0)
  log_w <- rep(0, N_PARTS); log_Z <- 0
  for (k in 2:length(betas)) {
    db <- betas[k] - betas[k - 1]
    log_w_inc <- db * log_lik
    m <- max(log_w + log_w_inc)
    log_Z <- log_Z + m + log(sum(exp(log_w + log_w_inc - m))) - log(N_PARTS)
    log_w <- log_w + log_w_inc
    w <- exp(log_w - max(log_w)); w <- w / sum(w)
    if (1 / sum(w^2) < N_PARTS / 2) {
      idx <- sample.int(N_PARTS, N_PARTS, replace = TRUE, prob = w)
      particles <- particles[idx, ]; log_lik <- log_lik[idx]
      log_prior <- log_prior[idx]; log_w <- rep(0, N_PARTS)
    }
  }
  w <- exp(log_w - max(log_w)); w <- w / sum(w)
  elapsed_smc <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  smc_mean <- c(mu      = sum(w * particles[, 1]),
                phi     = sum(w * particles[, 2]),
                sigma_h = sum(w * particles[, 3]))
  smc_std  <- c(mu      = sqrt(sum(w * (particles[, 1] - smc_mean[1])^2)),
                phi     = sqrt(sum(w * (particles[, 2] - smc_mean[2])^2)),
                sigma_h = sqrt(sum(w * (particles[, 3] - smc_mean[3])^2)))
  q_w <- function(x, w, p) {
    o <- order(x); cw <- cumsum(w[o]); x[o][which(cw >= p)[1]]
  }
  smc_lo <- c(q_w(particles[, 1], w, 0.025),
              q_w(particles[, 2], w, 0.025),
              q_w(particles[, 3], w, 0.025))
  smc_hi <- c(q_w(particles[, 1], w, 0.975),
              q_w(particles[, 2], w, 0.975),
              q_w(particles[, 3], w, 0.975))
  smc_ess <- 1 / sum(w^2)
  cat(sprintf("  finished in %.2f s,  log p(y) = %+.3f, ESS = %.0f / %d\n",
              elapsed_smc, log_Z, smc_ess, N_PARTS))
  for (i in seq_along(PARAM_NAMES)) {
    cat(sprintf("  %-7s  mean=%+.4f  std=%.4f  CI95=[%+.4f, %+.4f]\n",
                PARAM_NAMES[i], smc_mean[i], smc_std[i], smc_lo[i], smc_hi[i]))
  }
  rows_smc <- do.call(rbind, lapply(seq_along(PARAM_NAMES), function(i) {
    mk_param_row("SMC", PARAM_NAMES[i], unname(THETA_TRUE[i]),
                 unname(smc_mean[i]), unname(smc_std[i]),
                 unname(smc_lo[i]), unname(smc_hi[i]),
                 smc_ess, elapsed_smc, log_Z)
  }))
  rows_smc <- rbind(rows_smc,
                    mk_summary_row("SMC", unname(smc_mean),
                                   rep(smc_ess, 3), elapsed_smc, log_Z))
  all_rows[["SMC"]] <- rows_smc
} else {
  cat("SMC package NOT installed -- skipping (NA placeholder rows).\n")
  all_rows[["SMC"]] <- empty_method_rows("SMC", "package not installed")
}

# ==============================================================================
# Persist
# ==============================================================================
combined <- do.call(rbind, all_rows)
combined <- combined[, COL_NAMES]   # ensure column order
write.csv(combined, output_csv, row.names = FALSE)

cat("\n", strrep("=", 78), "\n", sep = "")
cat(sprintf("Saved %d rows -> %s\n", nrow(combined), output_csv))
print(combined[combined$param == "all",
               c("method", "abs_bias", "sq_error", "ess",
                 "time_seconds", "log_evidence")],
      row.names = FALSE, digits = 4)
cat("Done.\n")
