#!/usr/bin/env Rscript
# ==============================================================================
# validate_applications_comparison.R
# Compare particlefilterbox (Python) and R-based validation results for the
# three applications covered in Phase 7:
#   1. Merton jump-diffusion (yuima / direct MLE)
#   2. Stochastic SIR with Poisson observations (pomp / pure-R PF)
#   3. DSGE state-space (dlm Kalman filter)
#
# The script:
#   * Reads each Python solution CSV (examples/applications/solutions/)
#   * Reads each R validation CSV (examples/applications/R_validation/)
#     -- running the corresponding R script if a CSV is missing.
#   * Computes side-by-side metrics (correlation, RMSE, |diff|) and writes
#     results_r_comparison.csv.
#
# Required R packages:
#   yuima (jump-diffusion, optional)  - fallback: stats4::mle on Merton mixture
#   pomp  (SIR, optional)             - fallback: pure-R bootstrap PF
#   dlm                               - Kalman filter for DSGE benchmark
#
# Usage: Rscript validate_applications_comparison.R
# ==============================================================================

suppressPackageStartupMessages({
  has_yuima <- requireNamespace("yuima", quietly = TRUE)
  has_pomp  <- requireNamespace("pomp",  quietly = TRUE)
  has_dlm   <- requireNamespace("dlm",   quietly = TRUE)
})

cat(strrep("=", 71), "\n")
cat("particlefilterbox Phase 7 - R vs Python comparison\n")
cat(strrep("=", 71), "\n\n")
cat("R package availability:\n")
cat(sprintf("  yuima : %s\n", ifelse(has_yuima, "yes", "NO (fallback used)")))
cat(sprintf("  pomp  : %s\n", ifelse(has_pomp,  "yes", "NO (pure-R fallback)")))
cat(sprintf("  dlm   : %s\n\n", ifelse(has_dlm,  "yes", "MISSING - DSGE skipped")))

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
r_dir      <- script_dir
app_dir    <- normalizePath(file.path(script_dir, ".."))
data_dir   <- file.path(app_dir, "data")
sol_dir    <- file.path(app_dir, "solutions")
out_csv    <- file.path(r_dir, "results_r_comparison.csv")

ensure_r_result <- function(csv_name, script_name) {
  csv_path <- file.path(r_dir, csv_name)
  if (!file.exists(csv_path)) {
    sp <- file.path(r_dir, script_name)
    cat(sprintf("[*] %s missing - running %s ...\n", csv_name, script_name))
    status <- system2("Rscript", args = shQuote(sp), stdout = TRUE,
                      stderr = TRUE)
    attr_status <- attr(status, "status")
    if (!is.null(attr_status) && attr_status != 0) {
      cat("    script exited with status ", attr_status, "\n", sep = "")
    }
  }
  csv_path
}

# ------------------------------------------------------------------------------
# Row collector
# ------------------------------------------------------------------------------
rows <- list()
push_row <- function(application, metric, r_value, py_value,
                     abs_diff = NA_real_, rel_diff = NA_real_, note = "") {
  rows[[length(rows) + 1L]] <<- data.frame(
    application = application,
    metric      = metric,
    R_value     = r_value,
    Py_value    = py_value,
    abs_diff    = abs_diff,
    rel_diff    = rel_diff,
    note        = note,
    stringsAsFactors = FALSE
  )
}

safe_abs_diff <- function(a, b) {
  if (is.na(a) || is.na(b)) return(NA_real_)
  abs(a - b)
}
safe_rel_diff <- function(a, b) {
  if (is.na(a) || is.na(b)) return(NA_real_)
  den <- max(abs(a), abs(b), 1e-12)
  abs(a - b) / den
}

# =============================================================================
# 1. Jump-diffusion
# =============================================================================
cat("[1/3] Jump-diffusion\n")
r_jd_csv  <- ensure_r_result("results_r_jump_diff.csv", "validate_jump_diffusion.R")
py_jd_csv <- file.path(sol_dir, "results_jump_diffusion.csv")
py_jd_met <- file.path(sol_dir, "results_jump_diffusion_metrics.csv")

if (file.exists(r_jd_csv) && file.exists(py_jd_csv)) {
  r_jd  <- read.csv(r_jd_csv,  stringsAsFactors = FALSE)
  py_jd <- read.csv(py_jd_csv, stringsAsFactors = FALSE)

  r_filt <- r_jd[r_jd$section == "filtered", ]
  r_par  <- r_jd[r_jd$section == "parameters", ]
  r_met  <- r_jd[r_jd$section == "metrics", ]

  n <- min(nrow(r_filt), nrow(py_jd))
  cor_p <- suppressWarnings(cor(r_filt$p_jump[seq_len(n)],
                                py_jd$p_jump[seq_len(n)]))
  agree <- mean(r_filt$jump_pred[seq_len(n)] == py_jd$jump_pred[seq_len(n)])

  get_param <- function(df, key) {
    v <- df$value[df$parameter == key]
    if (length(v) == 0) NA_real_ else as.numeric(v[1])
  }
  mu_r      <- get_param(r_par, "mu")
  sigma_r   <- get_param(r_par, "sigma")
  lam_r     <- get_param(r_par, "lambda")
  mu_j_r    <- get_param(r_par, "mu_j")
  sigma_j_r <- get_param(r_par, "sigma_j")
  prec_r    <- get_param(r_met, "precision")
  rec_r     <- get_param(r_met, "recall")
  f1_r      <- get_param(r_met, "f1")

  py_met <- if (file.exists(py_jd_met))
              read.csv(py_jd_met, stringsAsFactors = FALSE) else NULL
  py_get <- function(df, key) {
    if (is.null(df)) return(NA_real_)
    v <- df$value[df$metric == key]
    if (length(v) == 0) NA_real_ else as.numeric(v[1])
  }
  prec_py <- py_get(py_met, "precision")
  rec_py  <- py_get(py_met, "recall")
  f1_py   <- py_get(py_met, "f1")
  ll_py   <- py_get(py_met, "log_likelihood")
  ll_r    <- get_param(r_par, "log_lik")

  # True values used by the data generator (generate_data.py).
  truth <- c(mu = 0.0005, sigma = 0.01, lambda = 0.05,
             mu_j = -0.02, sigma_j = 0.03)

  push_row("jump_diff", "mu",        mu_r,      truth["mu"],
           safe_abs_diff(mu_r, truth["mu"]),
           safe_rel_diff(mu_r, truth["mu"]),
           "R MLE vs true generator")
  push_row("jump_diff", "sigma",     sigma_r,   truth["sigma"],
           safe_abs_diff(sigma_r, truth["sigma"]),
           safe_rel_diff(sigma_r, truth["sigma"]), "R MLE vs true")
  push_row("jump_diff", "lambda",    lam_r,     truth["lambda"],
           safe_abs_diff(lam_r, truth["lambda"]),
           safe_rel_diff(lam_r, truth["lambda"]),
           "jump intensity R MLE vs true")
  push_row("jump_diff", "mu_jump",   mu_j_r,    truth["mu_j"],
           safe_abs_diff(mu_j_r, truth["mu_j"]),
           safe_rel_diff(mu_j_r, truth["mu_j"]), "R MLE vs true")
  push_row("jump_diff", "sigma_jump", sigma_j_r, truth["sigma_j"],
           safe_abs_diff(sigma_j_r, truth["sigma_j"]),
           safe_rel_diff(sigma_j_r, truth["sigma_j"]), "R MLE vs true")

  push_row("jump_diff", "precision", prec_r, prec_py,
           safe_abs_diff(prec_r, prec_py),
           safe_rel_diff(prec_r, prec_py), "jump detection")
  push_row("jump_diff", "recall",    rec_r, rec_py,
           safe_abs_diff(rec_r, rec_py),
           safe_rel_diff(rec_r, rec_py), "jump detection")
  push_row("jump_diff", "f1",        f1_r, f1_py,
           safe_abs_diff(f1_r, f1_py),
           safe_rel_diff(f1_r, f1_py), "jump detection")
  push_row("jump_diff", "log_likelihood", ll_r, ll_py,
           safe_abs_diff(ll_r, ll_py),
           safe_rel_diff(ll_r, ll_py),
           "R mixture MLE vs Py PF marginal")
  push_row("jump_diff", "corr_p_jump", cor_p, NA_real_,
           NA_real_, NA_real_, "Pearson corr(p_jump_R, p_jump_Py)")
  push_row("jump_diff", "detection_agreement", agree, NA_real_,
           NA_real_, NA_real_, "P(jump_pred_R == jump_pred_Py)")

  cat(sprintf("  jumps (R MLE): lambda=%.4f  mu_j=%.4f  sigma_j=%.4f\n",
              lam_r, mu_j_r, sigma_j_r))
  cat(sprintf("  precision R/Py: %.3f / %.3f   recall: %.3f / %.3f\n",
              prec_r, prec_py, rec_r, rec_py))
  cat(sprintf("  corr(p_jump R, Py) = %.4f, detection agreement = %.3f\n",
              cor_p, agree))
} else {
  cat("  [skip] required CSVs not found.\n")
  push_row("jump_diff", "status", NA_real_, NA_real_, NA_real_, NA_real_,
           "CSV missing")
}

# =============================================================================
# 2. SIR
# =============================================================================
cat("\n[2/3] Stochastic SIR\n")
r_sir_csv  <- ensure_r_result("results_r_sir.csv", "validate_sir.R")
py_sir_csv <- file.path(sol_dir, "results_sir_epidemic.csv")

if (file.exists(r_sir_csv) && file.exists(py_sir_csv)) {
  r_sir  <- read.csv(r_sir_csv,  stringsAsFactors = FALSE)
  py_sir <- read.csv(py_sir_csv, stringsAsFactors = FALSE)
  r_filt <- r_sir[r_sir$section == "filtered", ]
  r_met  <- r_sir[r_sir$section == "metrics", ]

  n <- min(nrow(r_filt), nrow(py_sir))
  cor_i  <- suppressWarnings(cor(r_filt$I_filt[seq_len(n)],
                                 py_sir$I_filt[seq_len(n)]))
  cor_r0 <- suppressWarnings(cor(r_filt$R0_mean[seq_len(n)],
                                 py_sir$R0_mean[seq_len(n)]))
  rmse_i <- sqrt(mean((r_filt$I_filt[seq_len(n)] -
                       py_sir$I_filt[seq_len(n)])^2))

  r0_r_final  <- r_filt$R0_mean[n]
  r0_py_final <- py_sir$R0_mean[n]
  r0_true     <- 0.3 / 0.1

  get_met <- function(df, key) {
    v <- df$value[df$parameter == key]
    if (length(v) == 0) NA_real_ else as.numeric(v[1])
  }
  ll_r  <- get_met(r_met, "log_likelihood")
  cov_r <- get_met(r_met, "coverage_90_I")
  rmse_I_r  <- get_met(r_met, "rmse_I")
  rmse_I_py <- sqrt(mean((py_sir$I_filt - py_sir$I_true)^2))
  ll_py <- sum(py_sir$log_lik_inc)

  push_row("sir", "R0_final", r0_r_final, r0_py_final,
           safe_abs_diff(r0_r_final, r0_py_final),
           safe_rel_diff(r0_r_final, r0_py_final),
           sprintf("true R0 = %.2f", r0_true))
  push_row("sir", "R0_vs_true_R", r0_r_final, r0_true,
           safe_abs_diff(r0_r_final, r0_true),
           safe_rel_diff(r0_r_final, r0_true), "R posterior mean vs truth")
  push_row("sir", "R0_vs_true_Py", r0_py_final, r0_true,
           safe_abs_diff(r0_py_final, r0_true),
           safe_rel_diff(r0_py_final, r0_true), "Py posterior mean vs truth")
  push_row("sir", "rmse_I",     rmse_I_r, rmse_I_py,
           safe_abs_diff(rmse_I_r, rmse_I_py),
           safe_rel_diff(rmse_I_r, rmse_I_py), "filter RMSE on I_t")
  push_row("sir", "log_likelihood", ll_r, ll_py,
           safe_abs_diff(ll_r, ll_py),
           safe_rel_diff(ll_r, ll_py), "marginal log-lik from PF")
  push_row("sir", "coverage_90_I", cov_r, NA_real_,
           NA_real_, NA_real_, "P(I_true within R 90% band)")
  push_row("sir", "corr_I",        cor_i, NA_real_,
           NA_real_, NA_real_, "Pearson corr(I_filt R, Py)")
  push_row("sir", "corr_R0",       cor_r0, NA_real_,
           NA_real_, NA_real_, "Pearson corr(R0_mean R, Py)")
  push_row("sir", "rmse_R_vs_Py",  rmse_i, NA_real_,
           NA_real_, NA_real_, "RMSE(I_filt R, I_filt Py)")

  cat(sprintf("  R0: R=%.3f  Py=%.3f  true=%.3f\n",
              r0_r_final, r0_py_final, r0_true))
  cat(sprintf("  RMSE I : R=%.2f  Py=%.2f\n", rmse_I_r, rmse_I_py))
  cat(sprintf("  corr(I_filt R, Py) = %.4f, corr(R0_mean) = %.4f\n",
              cor_i, cor_r0))
} else {
  cat("  [skip] required CSVs not found.\n")
  push_row("sir", "status", NA_real_, NA_real_, NA_real_, NA_real_,
           "CSV missing")
}

# =============================================================================
# 3. DSGE via dlm::Kalman filter
# =============================================================================
cat("\n[3/3] DSGE (AR(1) state-space via dlm Kalman filter)\n")
py_dsge_csv <- file.path(sol_dir, "results_dsge.csv")
ty_csv      <- file.path(data_dir, "treasury_yields.csv")

if (has_dlm && file.exists(py_dsge_csv) && file.exists(ty_csv)) {
  library(dlm)
  ty <- read.csv(ty_csv, stringsAsFactors = FALSE)
  py <- read.csv(py_dsge_csv, stringsAsFactors = FALSE)

  # The DSGE solution in solution_01_dsge.py models three latent AR(1)
  # states (output_gap, inflation, interest_rate) observed with Gaussian
  # noise.  The analytical benchmark is a diagonal DLM with:
  #   x_t = F x_{t-1} + w_t,  w_t ~ N(0, Q)
  #   y_t = x_t        + v_t, v_t ~ N(0, R)
  # F, Q, R are taken from the same generative params as the Python
  # solution (AR coefficients 0.9 / 0.5-ish etc., observation noise sds
  # 0.2 / 0.15 / 0.1).  This gives a clean Kalman-filter benchmark that
  # the particle filter should asymptotically match.

  F_mat <- diag(c(0.9, 0.5, 0.8))
  Q_mat <- diag(c(0.1, 0.1, 0.05)^2)
  R_mat <- diag(c(0.2, 0.15, 0.1)^2)

  mod <- dlm(
    FF = diag(3), V  = R_mat,
    GG = F_mat,  W  = Q_mat,
    m0 = c(0, 0, 0), C0 = diag(3)
  )

  y_mat <- as.matrix(ty[, c("output_gap_obs", "inflation_obs",
                            "interest_rate_obs")])
  kf <- dlmFilter(y_mat, mod)

  # kf$m has length T+1 (includes m0 at position 1)
  m_filt <- kf$m[-1, , drop = FALSE]
  T_obs  <- nrow(m_filt)

  x_hat_R  <- m_filt[, 1]
  pi_hat_R <- m_filt[, 2]
  r_hat_R  <- m_filt[, 3]

  n <- min(T_obs, nrow(py))
  rmse_x_py  <- sqrt(mean((py$output_gap_filt    - py$output_gap_true)^2))
  rmse_pi_py <- sqrt(mean((py$inflation_filt     - py$inflation_true)^2))
  rmse_r_py  <- sqrt(mean((py$interest_rate_filt - py$interest_rate_true)^2))

  rmse_x_R   <- sqrt(mean((x_hat_R[seq_len(n)]  - py$output_gap_true[seq_len(n)])^2))
  rmse_pi_R  <- sqrt(mean((pi_hat_R[seq_len(n)] - py$inflation_true[seq_len(n)])^2))
  rmse_r_R   <- sqrt(mean((r_hat_R[seq_len(n)]  - py$interest_rate_true[seq_len(n)])^2))

  cor_x  <- suppressWarnings(cor(x_hat_R[seq_len(n)],  py$output_gap_filt[seq_len(n)]))
  cor_pi <- suppressWarnings(cor(pi_hat_R[seq_len(n)], py$inflation_filt[seq_len(n)]))
  cor_r  <- suppressWarnings(cor(r_hat_R[seq_len(n)],  py$interest_rate_filt[seq_len(n)]))

  push_row("dsge", "rmse_output_gap", rmse_x_R, rmse_x_py,
           safe_abs_diff(rmse_x_R, rmse_x_py),
           safe_rel_diff(rmse_x_R, rmse_x_py),
           "R Kalman vs Py PF vs true state")
  push_row("dsge", "rmse_inflation", rmse_pi_R, rmse_pi_py,
           safe_abs_diff(rmse_pi_R, rmse_pi_py),
           safe_rel_diff(rmse_pi_R, rmse_pi_py), "")
  push_row("dsge", "rmse_interest", rmse_r_R, rmse_r_py,
           safe_abs_diff(rmse_r_R, rmse_r_py),
           safe_rel_diff(rmse_r_R, rmse_r_py), "")
  push_row("dsge", "corr_output_gap", cor_x, NA_real_,
           NA_real_, NA_real_, "corr(KF_R, PF_Py)")
  push_row("dsge", "corr_inflation", cor_pi, NA_real_,
           NA_real_, NA_real_, "corr(KF_R, PF_Py)")
  push_row("dsge", "corr_interest",  cor_r, NA_real_,
           NA_real_, NA_real_, "corr(KF_R, PF_Py)")

  cat(sprintf("  RMSE output_gap  R/Py: %.4f / %.4f\n", rmse_x_R,  rmse_x_py))
  cat(sprintf("  RMSE inflation   R/Py: %.4f / %.4f\n", rmse_pi_R, rmse_pi_py))
  cat(sprintf("  RMSE interest    R/Py: %.4f / %.4f\n", rmse_r_R,  rmse_r_py))
  cat(sprintf("  corr(KF, PF): output_gap %.4f  inflation %.4f  interest %.4f\n",
              cor_x, cor_pi, cor_r))
} else {
  cat("  [skip] dlm not installed or DSGE CSV missing.\n")
  push_row("dsge", "status", NA_real_, NA_real_, NA_real_, NA_real_,
           "dlm missing or CSV missing")
}

# ------------------------------------------------------------------------------
# 4. Write comparison table
# ------------------------------------------------------------------------------
out <- do.call(rbind, rows)
write.csv(out, out_csv, row.names = FALSE)
cat(sprintf("\nComparison table written to: %s  (%d rows)\n",
            out_csv, nrow(out)))

cat("\nDone.\n")
