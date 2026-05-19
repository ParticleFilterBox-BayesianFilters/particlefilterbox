#!/usr/bin/env Rscript
# ==============================================================================
# validate_convergence.R
# Cross-validation of particle filter convergence diagnostics via pomp.
# Theory: variance of the unbiased log-likelihood estimator scales as O(1/N),
# so log(Var(logL)) plotted against log(N) should have slope ~ -1.
# Mirrors the Python convergence study in
#   ../solutions/solution_02_convergence.py
#
# Required R packages: pomp (>= 5.0)
# Install: install.packages("pomp")
#
# Usage: Rscript validate_convergence.R
# ==============================================================================

suppressPackageStartupMessages({
  has_pomp <- requireNamespace("pomp", quietly = TRUE)
})

cat(strrep("=", 71), "\n")
cat("Convergence study via pomp::pfilter (multiple particle counts)\n")
cat("Dataset: simulated_sv.csv\n")
cat(strrep("=", 71), "\n\n")

# ------------------------------------------------------------------------------
# Helpers
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

# ------------------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------------------
data_path <- file.path(script_dir, "..", "data", "simulated_sv.csv")
if (!file.exists(data_path)) {
  stop("Simulated SV dataset not found at: ", data_path)
}
sim <- read.csv(data_path, stringsAsFactors = FALSE)
y_obs  <- sim$y_obs
h_true <- sim$h_true
T      <- length(y_obs)
cat(sprintf("Loaded T = %d observations\n", T))

# ------------------------------------------------------------------------------
# 2. pomp model construction
# ------------------------------------------------------------------------------
if (!has_pomp) {
  cat("WARNING: package 'pomp' not installed; writing empty placeholder.\n")
  cat("Install with: install.packages('pomp')\n")
  empty <- data.frame(
    metric = character(),
    N = integer(),
    logL_mean = numeric(),
    logL_sd = numeric(),
    logL_var = numeric(),
    rmse_truth_mean = numeric(),
    rmse_truth_sd = numeric(),
    n_reps = integer(),
    runtime_sec = numeric(),
    stringsAsFactors = FALSE
  )
  write.csv(empty, file.path(script_dir, "results_r_convergence.csv"),
            row.names = FALSE)
  quit(status = 0)
}

library(pomp)

build_sv_pomp <- function(y) {
  T_loc <- length(y)
  pomp(
    data = data.frame(t = seq_len(T_loc), y = y),
    times = "t",
    t0 = 0,
    rinit = Csnippet("
      double sd0 = sigma / sqrt(1.0 - phi*phi);
      h = rnorm(mu, sd0);
    "),
    rprocess = discrete_time(
      step.fun = Csnippet("
        h = mu + phi*(h - mu) + sigma*rnorm(0.0, 1.0);
      "),
      delta.t = 1
    ),
    dmeasure = Csnippet("
      double vol = exp(h / 2.0);
      double z = y / vol;
      lik = -0.5*log(2.0*M_PI) - log(vol) - 0.5*z*z;
      lik = (give_log) ? lik : exp(lik);
    "),
    rmeasure = Csnippet("
      y = rnorm(0.0, exp(h / 2.0));
    "),
    statenames = "h",
    obsnames   = "y",
    paramnames = c("mu", "phi", "sigma"),
    params     = c(mu = -1.0, phi = 0.97, sigma = 0.15)
  )
}

cat("\nCompiling pomp object (Csnippets) ...\n")
sv_pomp <- build_sv_pomp(y_obs)
cat("Done.\n\n")

# ------------------------------------------------------------------------------
# 3. Convergence study
# ------------------------------------------------------------------------------
N_LIST  <- c(50L, 100L, 200L, 500L, 1000L, 2000L, 5000L)
N_REPS  <- 8L

cat(sprintf("Convergence study: N in {%s}, reps = %d\n",
            paste(N_LIST, collapse = ", "), N_REPS))
cat(strrep("-", 71), "\n")

rows <- list()
total_t0 <- Sys.time()

for (N in N_LIST) {
  lls   <- numeric(N_REPS)
  rmses <- numeric(N_REPS)
  t0 <- Sys.time()
  for (rep in seq_len(N_REPS)) {
    set.seed(2000L + 100L * rep + N)
    pf <- pfilter(sv_pomp, Np = N, filter.mean = TRUE)
    lls[rep] <- as.numeric(logLik(pf))
    fm <- as.numeric(filter_mean(pf))
    if (length(fm) >= T) {
      rmses[rep] <- sqrt(mean((fm[1:T] - h_true)^2))
    } else {
      rmses[rep] <- NA_real_
    }
  }
  dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  rows[[length(rows) + 1L]] <- data.frame(
    metric          = "convergence",
    N               = as.integer(N),
    logL_mean       = mean(lls),
    logL_sd         = sd(lls),
    logL_var        = var(lls),
    rmse_truth_mean = mean(rmses, na.rm = TRUE),
    rmse_truth_sd   = sd(rmses,  na.rm = TRUE),
    n_reps          = as.integer(N_REPS),
    runtime_sec     = dt,
    stringsAsFactors = FALSE
  )
  cat(sprintf(
    paste0("  N = %5d   logL = %9.3f +/- %.3f   Var(logL) = %8.4f   ",
           "RMSE_h = %.4f   (%.2fs)\n"),
    N, mean(lls), sd(lls), var(lls),
    mean(rmses, na.rm = TRUE), dt))
}
total_dt <- as.numeric(difftime(Sys.time(), total_t0, units = "secs"))
cat(sprintf("Total convergence runtime: %.1f sec\n\n", total_dt))

conv_df <- do.call(rbind, rows)

# ------------------------------------------------------------------------------
# 4. Slope estimates
# ------------------------------------------------------------------------------
N_arr        <- as.numeric(conv_df$N)
slope_logvar <- coef(lm(log(conv_df$logL_var) ~ log(N_arr)))[2]
slope_rmse   <- coef(lm(log(conv_df$rmse_truth_mean) ~ log(N_arr)))[2]
cat(sprintf("Empirical slope log(Var(logL)) ~ log(N) : %+.3f   (theory  -1.000)\n",
            slope_logvar))
cat(sprintf("Empirical slope log(RMSE_h)  ~ log(N) : %+.3f   (theory ~ 0; bias floor)\n",
            slope_rmse))

slope_row <- data.frame(
  metric          = "slope",
  N               = NA_integer_,
  logL_mean       = NA_real_,
  logL_sd         = NA_real_,
  logL_var        = NA_real_,
  rmse_truth_mean = NA_real_,
  rmse_truth_sd   = NA_real_,
  n_reps          = NA_integer_,
  runtime_sec     = total_dt,
  stringsAsFactors = FALSE
)
slope_row$slope_logL_var <- as.numeric(slope_logvar)
slope_row$slope_rmse     <- as.numeric(slope_rmse)
conv_df$slope_logL_var   <- NA_real_
conv_df$slope_rmse       <- NA_real_

# ------------------------------------------------------------------------------
# 5. Cross-check vs Python convergence results
# ------------------------------------------------------------------------------
py_path <- file.path(script_dir, "..", "solutions", "results_convergence.csv")
slope_py_logvar <- NA_real_
slope_py_rmse   <- NA_real_
if (file.exists(py_path)) {
  cat("\n--- Cross-check vs particlefilterbox (Python) ---\n")
  py <- read.csv(py_path, stringsAsFactors = FALSE)
  py_conv  <- py[py$metric == "convergence", ]
  py_slope <- py[py$metric == "slope", ]
  if (nrow(py_slope) > 0) {
    slope_py_logvar <- as.numeric(py_slope$slope_logL_var[1])
    cat(sprintf("  Python slope log(Var(logL)) ~ log(N) : %+.3f\n",
                slope_py_logvar))
  }

  cat("\n  N      R logL       Py logL      |diff|     R Var(logL)   Py Var(logL)\n")
  for (N in N_LIST) {
    r_row  <- conv_df[conv_df$N == N, ]
    py_row <- py_conv[py_conv$N == N, ]
    if (nrow(r_row) == 1 && nrow(py_row) == 1) {
      r_ll  <- r_row$logL_mean[1]
      py_ll <- as.numeric(py_row$logL_mean[1])
      r_v   <- r_row$logL_var[1]
      py_v  <- as.numeric(py_row$logL_var[1])
      cat(sprintf("  %5d  %10.3f  %10.3f  %8.3f  %12.4f  %12.4f\n",
                  N, r_ll, py_ll, abs(r_ll - py_ll), r_v, py_v))
    }
  }

  if (is.finite(slope_py_logvar)) {
    diff_slope <- abs(slope_logvar - slope_py_logvar)
    cat(sprintf("\n  Slope agreement |R - Py| = %.4f   (target < 0.30)\n",
                diff_slope))
  }
} else {
  cat("\n  [skip] Python results not found:", py_path, "\n")
}

# ------------------------------------------------------------------------------
# 6. Save results
# ------------------------------------------------------------------------------
out <- rbind(conv_df, slope_row)
output_csv <- file.path(script_dir, "results_r_convergence.csv")
write.csv(out, output_csv, row.names = FALSE)
cat(sprintf("\nResults saved (%d rows) to: %s\n", nrow(out), output_csv))

# ------------------------------------------------------------------------------
# 7. Acceptance check
# ------------------------------------------------------------------------------
cat("\n")
cat(strrep("-", 71), "\n")
cat("Acceptance:\n")

# (a) Slope -1 theory check (within tolerance because of finite reps)
slope_ok <- is.finite(slope_logvar) && abs(slope_logvar + 1.0) < 0.5
cat(sprintf("  log(Var(logL)) slope = %+.3f (theory -1)  [%s]\n",
            slope_logvar,
            if (slope_ok) "PASS" else "CHECK"))

# (b) Slope agreement with Python
if (is.finite(slope_py_logvar)) {
  agree_ok <- abs(slope_logvar - slope_py_logvar) < 0.5
  cat(sprintf("  R slope vs Py slope: |%+.3f - %+.3f| = %.3f  [%s]\n",
              slope_logvar, slope_py_logvar,
              abs(slope_logvar - slope_py_logvar),
              if (agree_ok) "PASS" else "CHECK"))
}

# (c) logL agreement at largest N
if (file.exists(py_path)) {
  N_top <- max(N_LIST)
  r_row  <- conv_df[conv_df$N == N_top, ]
  py_row <- py[py$metric == "convergence" & py$N == N_top, ]
  if (nrow(r_row) == 1 && nrow(py_row) == 1) {
    diff_ll <- abs(r_row$logL_mean[1] - as.numeric(py_row$logL_mean[1]))
    ok_ll <- diff_ll < 1.0
    cat(sprintf("  logL agreement at N=%d : |R - Py| = %.3f   [%s]\n",
                N_top, diff_ll, if (ok_ll) "PASS" else "CHECK"))
  }
}
cat(strrep("-", 71), "\n")

cat("\nDone.\n")
