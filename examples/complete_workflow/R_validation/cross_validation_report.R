#!/usr/bin/env Rscript
# ==============================================================================
# cross_validation_report.R
#
# Build the final Python-vs-R cross-validation table for FASE 9.4.
#
# Inputs (must already exist):
#   ../solutions/results_sv_workflow_filtering.csv   (Python SV workflow)
#   ../solutions/results_sv_workflow_params.csv      (Python posterior summary)
#   ../solutions/results_sv_workflow_forecast.csv    (Python forecast cone)
#   ../solutions/results_estimation_comparison.csv   (Python SMC/PMMH/PGAS)
#   ./results_r_sv_workflow.csv                      (R full_sv_workflow_r.R)
#   ./results_r_estimation.csv                       (R full_estimation_comparison_r.R)
#
# Output:
#   ./results_r_cross_validation.csv     -- one row per metric, fields:
#       category, metric, python_value, r_value, abs_diff, rel_diff,
#       passes_threshold, threshold, notes
#
# Categories covered:
#   - filtered_h:    correlation between R and Python filtered/smoothed h_t
#                    (decimal scale, sign-aligned)
#   - parameters:    posterior-mean / std comparison for (mu, phi, sigma)
#   - forecast:      median/quantile comparison of vol forecast at h=1,5,20
#   - estimation:    bias / abs_bias / time per (method, param) on simulated data
#   - log_evidence:  R SMC log p(y) vs Python SMC log p(y) (when available)
#
# Usage: Rscript cross_validation_report.R
# ==============================================================================

cat(strrep("=", 78), "\n", sep = "")
cat("FASE 9.4 / cross_validation_report.R\n")
cat(strrep("=", 78), "\n\n", sep = "")

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", args, value = TRUE)
  if (length(fa) > 0) return(dirname(normalizePath(sub("^--file=", "", fa[1]))))
  getwd()
}
script_dir <- get_script_dir()

# Path to input CSVs
PY_DIR <- file.path(script_dir, "..", "solutions")
R_DIR  <- script_dir

py_filt_path     <- file.path(PY_DIR, "results_sv_workflow_filtering.csv")
py_params_path   <- file.path(PY_DIR, "results_sv_workflow_params.csv")
py_forecast_path <- file.path(PY_DIR, "results_sv_workflow_forecast.csv")
py_est_path      <- file.path(PY_DIR, "results_estimation_comparison.csv")
r_sv_path        <- file.path(R_DIR,  "results_r_sv_workflow.csv")
r_est_path       <- file.path(R_DIR,  "results_r_estimation.csv")

required <- c(py_filt_path, py_params_path, py_forecast_path, py_est_path,
              r_sv_path, r_est_path)
missing <- required[!file.exists(required)]
if (length(missing) > 0) {
  stop("Missing input CSVs:\n  ", paste(missing, collapse = "\n  "),
       "\nRun the upstream scripts first:\n",
       "  Rscript full_sv_workflow_r.R\n",
       "  Rscript full_estimation_comparison_r.R\n",
       "  python ../solutions/solution_01_sv_workflow.py\n",
       "  python ../solutions/solution_02_estimation_comparison.py")
}

py_filt     <- read.csv(py_filt_path,     stringsAsFactors = FALSE)
py_params   <- read.csv(py_params_path,   stringsAsFactors = FALSE)
py_forecast <- read.csv(py_forecast_path, stringsAsFactors = FALSE)
py_est      <- read.csv(py_est_path,      stringsAsFactors = FALSE)
r_sv        <- read.csv(r_sv_path,        stringsAsFactors = FALSE)
r_est       <- read.csv(r_est_path,       stringsAsFactors = FALSE)

cat(sprintf("Loaded Python filtering rows = %d, Python params rows = %d, R sv rows = %d\n",
            nrow(py_filt), nrow(py_params), nrow(r_sv)))
cat(sprintf("Loaded Python estimation rows = %d, R estimation rows = %d\n\n",
            nrow(py_est), nrow(r_est)))

# ==============================================================================
# Helpers
# ==============================================================================
ROW_COLS <- c("category", "metric", "python_value", "r_value",
              "abs_diff", "rel_diff", "passes_threshold", "threshold", "notes")

mk_row <- function(category, metric, py_val, r_val,
                   threshold = NA_real_, passes = NA, notes = "") {
  abs_diff <- if (is.na(py_val) || is.na(r_val)) NA_real_ else abs(py_val - r_val)
  rel_diff <- if (is.na(py_val) || is.na(r_val) || abs(py_val) < 1e-12) {
    NA_real_
  } else abs_diff / abs(py_val)
  data.frame(
    category = category, metric = metric,
    python_value = py_val, r_value = r_val,
    abs_diff = abs_diff, rel_diff = rel_diff,
    passes_threshold = passes, threshold = threshold,
    notes = notes, stringsAsFactors = FALSE
  )
}

rows <- list()

# ==============================================================================
# 1. Filtered / smoothed log-volatility h_t  (SV workflow on SP500)
# ==============================================================================
cat(strrep("-", 78), "\n", sep = "")
cat("1. Filtered / smoothed h_t correlations (decimal scale)\n")
cat(strrep("-", 78), "\n", sep = "")

r_filt <- r_sv[r_sv$section == "filtered", ]
r_bpf  <- r_sv[r_sv$section == "filtered_bpf_plugin", ]
n_match <- min(nrow(py_filt), nrow(r_filt))
if (n_match > 5) {
  py_h_plug   <- as.numeric(py_filt$filtered_h_plugin[1:n_match])
  py_vol_plug <- as.numeric(py_filt$filtered_vol_plugin[1:n_match])
  py_h_filt   <- as.numeric(py_filt$filtered_h_postmean[1:n_match])
  py_h_smooth <- as.numeric(py_filt$smoothed_h[1:n_match])
  r_h_smooth  <- as.numeric(r_filt$h_mean_dec[1:n_match])
  r_h_bpf     <- if (nrow(r_bpf) >= n_match) as.numeric(r_bpf$h_mean_dec[1:n_match]) else NULL
  r_vol_bpf   <- if (!is.null(r_h_bpf)) as.numeric(r_bpf$vol_mean_dec[1:n_match]) else NULL

  # Primary correlation: R Bootstrap PF (at plug-in theta) vs Python BPF (at
  # plug-in theta). Same algorithm, same theta, same data -- the *pure*
  # algorithmic cross-check. Should yield > 0.95.
  cor_bpf <- if (!is.null(r_h_bpf)) suppressWarnings(cor(r_h_bpf, py_h_plug)) else NA_real_
  cor_vol_bpf <- if (!is.null(r_vol_bpf)) suppressWarnings(cor(r_vol_bpf, py_vol_plug)) else NA_real_
  rmse_bpf <- if (!is.null(r_h_bpf)) sqrt(mean((r_h_bpf - py_h_plug)^2)) else NA_real_

  # Secondary correlations: R smoother (stochvol) vs Python posterior-mean filter
  # and Python FFBSi smoother. These compare two *different* algorithms / posteriors,
  # so they're informative but expected to be lower than the BPF-vs-BPF check.
  cor_filt    <- suppressWarnings(cor(r_h_smooth, py_h_filt))
  cor_smooth  <- suppressWarnings(cor(r_h_smooth, py_h_smooth))
  cor_pf_sm   <- suppressWarnings(cor(py_h_filt, py_h_smooth))

  rmse_filt   <- sqrt(mean((r_h_smooth - py_h_filt)^2))

  cat(sprintf("  [primary, BPF@plugin]\n"))
  cat(sprintf("    cor(R h_BPF,  Py h_plugin)  = %.4f   (n=%d)\n", cor_bpf, n_match))
  cat(sprintf("    cor(R vol_BPF,Py vol_plugin)= %.4f\n", cor_vol_bpf))
  cat(sprintf("    RMSE(R h_BPF, Py h_plugin)  = %.4f\n", rmse_bpf))
  cat(sprintf("  [secondary, R smoother vs Py posterior mean]\n"))
  cat(sprintf("    cor(R h_smooth, Py h_filt) = %.4f\n", cor_filt))
  cat(sprintf("    cor(R h_smooth, Py h_smoo) = %.4f\n", cor_smooth))
  cat(sprintf("    cor(Py h_filt, h_smoothed) = %.4f  (Python internal sanity)\n", cor_pf_sm))

  rows[[length(rows) + 1]] <- mk_row(
    "filtered_h", "cor(h_R_BPF, h_Py_plugin)",
    py_val = 1.0, r_val = cor_bpf, threshold = 0.85,
    passes = !is.na(cor_bpf) && cor_bpf > 0.85,
    notes = sprintf("PRIMARY: bootstrap PF cross-check at plug-in theta, n=%d", n_match)
  )
  rows[[length(rows) + 1]] <- mk_row(
    "filtered_h", "cor(vol_R_BPF, vol_Py_plugin)",
    py_val = 1.0, r_val = cor_vol_bpf, threshold = 0.85,
    passes = !is.na(cor_vol_bpf) && cor_vol_bpf > 0.85,
    notes = "PRIMARY: vol_t = exp(h_t/2) cross-check at plug-in theta"
  )
  rows[[length(rows) + 1]] <- mk_row(
    "filtered_h", "rmse(h_R_BPF, h_Py_plugin)",
    py_val = 0.0, r_val = rmse_bpf, threshold = 0.30,
    passes = !is.na(rmse_bpf) && rmse_bpf < 0.30,
    notes = "PRIMARY: RMSE on log-vol scale at plug-in theta"
  )
  rows[[length(rows) + 1]] <- mk_row(
    "filtered_h", "cor(h_R_smooth, h_Py_filt)",
    py_val = 1.0, r_val = cor_filt, threshold = NA_real_, passes = NA,
    notes = "Secondary: R stochvol smoother vs Py PGAS filter (expected < BPF check)"
  )
  rows[[length(rows) + 1]] <- mk_row(
    "filtered_h", "cor(h_R_smooth, h_Py_smooth)",
    py_val = 1.0, r_val = cor_smooth, threshold = NA_real_, passes = NA,
    notes = "Secondary: R stochvol smoother vs Py FFBSi smoother"
  )
} else {
  cat("  (skipped: insufficient overlapping observations)\n")
}

# ==============================================================================
# 2. Posterior parameter summary (mu, phi, sigma) on SV workflow
# ==============================================================================
cat("\n", strrep("-", 78), "\n", sep = "")
cat("2. Posterior parameter summary (Python PGAS  vs  R stochvol)\n")
cat(strrep("-", 78), "\n", sep = "")

r_par <- r_sv[r_sv$section == "parameters", ]
# Python reports sigma_h, R reports sigma -- align names
param_map <- c(mu = "mu", phi = "phi", sigma_h = "sigma")
for (py_p in names(param_map)) {
  r_p <- param_map[[py_p]]
  py_row <- py_params[py_params$param == py_p, ]
  r_row  <- r_par[r_par$parameter == r_p, ]
  if (nrow(py_row) != 1 || nrow(r_row) != 1) next

  py_mean <- as.numeric(py_row$posterior_mean)
  r_mean  <- as.numeric(r_row$post_mean_dec)   # decimal-scale equivalent
  py_std  <- as.numeric(py_row$posterior_std)
  r_std   <- as.numeric(r_row$post_sd)

  cat(sprintf("  %-7s  Py: mean=%+.4f std=%.4f   R: mean=%+.4f std=%.4f   |dmean|=%.4f\n",
              py_p, py_mean, py_std, r_mean, r_std, abs(py_mean - r_mean)))

  # Threshold: |bias_difference| should be small relative to the parameter scale.
  # We use 2 * Py std as a generous tolerance (within 2-sigma of Python posterior).
  thr <- max(2 * py_std, 0.05)
  rows[[length(rows) + 1]] <- mk_row(
    "parameters", sprintf("posterior_mean[%s]", py_p),
    py_val = py_mean, r_val = r_mean, threshold = thr,
    passes = abs(py_mean - r_mean) <= thr,
    notes = sprintf("threshold = max(2*Py_std=%.4f, 0.05)", 2 * py_std)
  )
  rows[[length(rows) + 1]] <- mk_row(
    "parameters", sprintf("posterior_std[%s]", py_p),
    py_val = py_std, r_val = r_std, threshold = NA_real_,
    passes = NA,
    notes = "ratio reported in rel_diff; small ratio = similar uncertainty"
  )
}

# ==============================================================================
# 3. Forecast comparison (median + quantiles at h=1, 5, 20)
# ==============================================================================
cat("\n", strrep("-", 78), "\n", sep = "")
cat("3. Volatility forecast (decimal scale)\n")
cat(strrep("-", 78), "\n", sep = "")

r_fore <- r_sv[r_sv$section == "forecast", ]
H_targets <- intersect(c(1L, 5L, 20L), as.integer(r_fore$horizon))
H_targets <- intersect(H_targets, as.integer(py_forecast$horizon))
for (h in H_targets) {
  py_row <- py_forecast[as.integer(py_forecast$horizon) == h, ]
  r_row  <- r_fore[as.integer(r_fore$horizon) == h, ]
  if (nrow(py_row) != 1 || nrow(r_row) != 1) next
  for (q in c("vol_q05", "vol_q50", "vol_q95")) {
    qcol_r <- sub("^vol_", "vol_", q)             # vol_q05 etc.
    qcol_r <- sub("^vol_q05$", "vol_q05_dec", qcol_r)
    qcol_r <- sub("^vol_q50$", "vol_mean_dec", qcol_r)
    qcol_r <- sub("^vol_q95$", "vol_q95_dec", qcol_r)
    py_v <- as.numeric(py_row[[q]])
    r_v  <- as.numeric(r_row[[qcol_r]])
    cat(sprintf("  h=%2d %-8s  Py=%.5f  R=%.5f  |diff|=%.5f\n",
                h, q, py_v, r_v, abs(py_v - r_v)))
    rows[[length(rows) + 1]] <- mk_row(
      "forecast", sprintf("h=%d/%s", h, q),
      py_val = py_v, r_val = r_v, threshold = NA_real_,
      passes = NA,
      notes = "decimal-scale forecast cone"
    )
  }
}

# ==============================================================================
# 4. Estimation comparison (simulated data, methods)
# Python methods: SMC, PMMH, PGAS.   R methods: SMC, pomp_pmmh, stochvol.
# ==============================================================================
cat("\n", strrep("-", 78), "\n", sep = "")
cat("4. Estimation methods comparison (simulated data, T=200)\n")
cat(strrep("-", 78), "\n", sep = "")

method_pairs <- list(
  list(py = "SMC",  r = "SMC",        label = "SMC"),
  list(py = "PMMH", r = "pomp_pmmh",  label = "PMMH"),
  list(py = "PGAS", r = "stochvol",   label = "PGAS<->stochvol")
)

for (mp in method_pairs) {
  cat(sprintf("\n  ** %s  (Py=%s  vs  R=%s) **\n", mp$label, mp$py, mp$r))
  for (param in c("mu", "phi", "sigma_h")) {
    py_row <- py_est[py_est$method == mp$py & py_est$param == param, ]
    r_row  <- r_est [r_est$method  == mp$r  & r_est$param  == param, ]
    if (nrow(py_row) != 1 || nrow(r_row) != 1) next
    py_mean <- as.numeric(py_row$posterior_mean)
    r_mean  <- as.numeric(r_row$posterior_mean)
    py_bias <- as.numeric(py_row$bias)
    r_bias  <- as.numeric(r_row$bias)
    py_abs  <- as.numeric(py_row$abs_bias)
    r_abs   <- as.numeric(r_row$abs_bias)
    cat(sprintf("    %-7s  Py mean=%+.4f bias=%+.4f   R mean=%+.4f bias=%+.4f   |dmean|=%.4f\n",
                param, py_mean, py_bias, r_mean, r_bias,
                ifelse(is.na(r_mean) || is.na(py_mean), NA, abs(py_mean - r_mean))))
    rows[[length(rows) + 1]] <- mk_row(
      "estimation", sprintf("%s/%s/posterior_mean", mp$label, param),
      py_val = py_mean, r_val = r_mean, threshold = 0.20,
      passes = !is.na(r_mean) && !is.na(py_mean) && abs(py_mean - r_mean) < 0.20,
      notes = "absolute difference of posterior means on simulated SV"
    )
    rows[[length(rows) + 1]] <- mk_row(
      "estimation", sprintf("%s/%s/abs_bias", mp$label, param),
      py_val = py_abs, r_val = r_abs, threshold = NA_real_,
      passes = NA, notes = "abs_bias relative to true theta"
    )
  }
  # Per-method aggregate row
  py_all <- py_est[py_est$method == mp$py & py_est$param == "all", ]
  r_all  <- r_est [r_est$method  == mp$r  & r_est$param  == "all", ]
  if (nrow(py_all) == 1 && nrow(r_all) == 1) {
    py_t <- as.numeric(py_all$time_seconds)
    r_t  <- as.numeric(r_all$time_seconds)
    py_e <- as.numeric(py_all$ess)
    r_e  <- as.numeric(r_all$ess)
    py_le <- suppressWarnings(as.numeric(py_all$log_evidence))
    r_le  <- suppressWarnings(as.numeric(r_all$log_evidence))
    cat(sprintf("    [all]    time:   Py=%.2fs  R=%.2fs   ess_min: Py=%.1f R=%.1f\n",
                py_t, r_t, py_e, r_e))
    rows[[length(rows) + 1]] <- mk_row(
      "estimation", sprintf("%s/all/time_seconds", mp$label),
      py_val = py_t, r_val = r_t, threshold = NA_real_, passes = NA,
      notes = "wall time per method"
    )
    rows[[length(rows) + 1]] <- mk_row(
      "estimation", sprintf("%s/all/ess_min", mp$label),
      py_val = py_e, r_val = r_e, threshold = NA_real_, passes = NA,
      notes = "minimum ESS across parameters"
    )
    if (!is.na(py_le) || !is.na(r_le)) {
      rows[[length(rows) + 1]] <- mk_row(
        "log_evidence", sprintf("%s/log_p_y", mp$label),
        py_val = py_le, r_val = r_le, threshold = 5.0,
        passes = (!is.na(py_le) && !is.na(r_le) && abs(py_le - r_le) < 5.0),
        notes = "absolute difference of log-evidence (where reported)"
      )
    }
  }
}

# ==============================================================================
# Persist + print summary table
# ==============================================================================
combined <- do.call(rbind, rows)
combined <- combined[, ROW_COLS]
out_path <- file.path(R_DIR, "results_r_cross_validation.csv")
write.csv(combined, out_path, row.names = FALSE)

cat("\n", strrep("=", 78), "\n", sep = "")
cat(sprintf("Saved %d cross-validation rows -> %s\n", nrow(combined), out_path))
cat(strrep("=", 78), "\n", sep = "")

# Compact summary: how many threshold checks passed
checks <- combined[!is.na(combined$passes_threshold), ]
n_checks <- nrow(checks)
n_pass   <- sum(checks$passes_threshold == TRUE, na.rm = TRUE)
cat(sprintf("\nThreshold checks: %d / %d passed.\n", n_pass, n_checks))
if (n_checks > 0) {
  print(checks[, c("category", "metric", "python_value", "r_value",
                   "abs_diff", "passes_threshold")],
        row.names = FALSE, digits = 4)
}
cat("\nDone.\n")
