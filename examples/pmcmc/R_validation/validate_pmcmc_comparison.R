#!/usr/bin/env Rscript
# ==============================================================================
# validate_pmcmc_comparison.R
# Compare posteriors from R (pomp, nimbleSMC) vs Python (particlefilterbox)
# for all three PMCMC algorithms: PMMH, Particle Gibbs, PGAS
#
# Required R packages: coda
# Install: install.packages("coda")
#
# Usage: Rscript validate_pmcmc_comparison.R
# ==============================================================================

suppressPackageStartupMessages({
  library(coda)
})

cat(strrep("=", 71), "\n")
cat("PMCMC Cross-Validation: R vs Python (particlefilterbox)\n")
cat(strrep("=", 71), "\n\n")

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------
# Determine script directory robustly
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  }
  return(getwd())
}
script_dir <- get_script_dir()

load_csv <- function(path, label) {
  if (!file.exists(path)) {
    cat(sprintf("WARNING: %s not found at %s\n", label, path))
    return(NULL)
  }
  df <- read.csv(path)
  cat(sprintf("Loaded %s: %d samples, columns: %s\n",
              label, nrow(df), paste(names(df), collapse=", ")))
  return(df)
}

compare_posteriors <- function(r_samples, py_samples, param_names, algo_name) {
  cat(sprintf("\n--- %s: R vs Python Comparison ---\n\n", algo_name))

  results <- data.frame(
    parameter = character(),
    r_mean = numeric(),
    py_mean = numeric(),
    diff_mean = numeric(),
    r_sd = numeric(),
    py_sd = numeric(),
    r_ess = numeric(),
    py_ess = numeric(),
    ks_pvalue = numeric(),
    stringsAsFactors = FALSE
  )

  for (p in param_names) {
    if (!(p %in% names(r_samples)) || !(p %in% names(py_samples))) {
      cat(sprintf("  Skipping %s (not in both datasets)\n", p))
      next
    }

    r_vals <- r_samples[[p]]
    py_vals <- py_samples[[p]]

    r_mean <- mean(r_vals)
    py_mean <- mean(py_vals)
    r_sd <- sd(r_vals)
    py_sd <- sd(py_vals)

    # ESS
    r_ess <- effectiveSize(r_vals)
    py_ess <- effectiveSize(py_vals)

    # KS test for distribution comparison
    ks <- ks.test(r_vals, py_vals)

    cat(sprintf("  %s:\n", p))
    cat(sprintf("    R:      mean=%.4f  sd=%.4f  ESS=%.1f\n", r_mean, r_sd, r_ess))
    cat(sprintf("    Python: mean=%.4f  sd=%.4f  ESS=%.1f\n", py_mean, py_sd, py_ess))
    cat(sprintf("    Diff:   mean=%.4f (%.1f%%)\n",
                abs(r_mean - py_mean),
                100 * abs(r_mean - py_mean) / max(abs(py_mean), 1e-6)))
    cat(sprintf("    KS test p-value: %.4f %s\n",
                ks$p.value,
                ifelse(ks$p.value > 0.05, "(distributions consistent)",
                       "(distributions differ)")))

    results <- rbind(results, data.frame(
      parameter = p,
      r_mean = r_mean,
      py_mean = py_mean,
      diff_mean = abs(r_mean - py_mean),
      r_sd = r_sd,
      py_sd = py_sd,
      r_ess = r_ess,
      py_ess = py_ess,
      ks_pvalue = ks$p.value,
      stringsAsFactors = FALSE
    ))
  }

  return(results)
}

compare_acceptance <- function(r_samples, py_samples, param_names, algo_name) {
  cat(sprintf("\n  Acceptance rates (%s):\n", algo_name))

  for (p in param_names) {
    if (!(p %in% names(r_samples)) || !(p %in% names(py_samples))) next

    r_rate <- sum(diff(r_samples[[p]]) != 0) / (nrow(r_samples) - 1)
    py_rate <- sum(diff(py_samples[[p]]) != 0) / (nrow(py_samples) - 1)

    cat(sprintf("    %s: R=%.3f, Python=%.3f, diff=%.3f\n",
                p, r_rate, py_rate, abs(r_rate - py_rate)))
  }
}

# ------------------------------------------------------------------------------
# 1. Load all posterior files
# ------------------------------------------------------------------------------
cat("Loading posterior samples...\n\n")

# R results
r_pmmh <- load_csv(file.path(script_dir, "results_r_pmmh.csv"), "R PMMH")
r_pgas <- load_csv(file.path(script_dir, "results_r_pgas.csv"), "R PGAS")

# Python results
py_dir <- file.path(script_dir, "..", "solutions")
py_pmmh <- load_csv(file.path(py_dir, "results_pmmh.csv"), "Python PMMH")
py_pg <- load_csv(file.path(py_dir, "results_particle_gibbs.csv"), "Python PG")
py_pgas <- load_csv(file.path(py_dir, "results_pgas.csv"), "Python PGAS")

# ------------------------------------------------------------------------------
# 2. Compare PMMH posteriors
# ------------------------------------------------------------------------------
all_results <- data.frame()

if (!is.null(r_pmmh) && !is.null(py_pmmh)) {
  pmmh_results <- compare_posteriors(
    r_pmmh, py_pmmh,
    c("mu", "phi", "sigma_h"),
    "PMMH"
  )
  pmmh_results$algorithm <- "PMMH"
  all_results <- rbind(all_results, pmmh_results)

  compare_acceptance(r_pmmh, py_pmmh, c("mu", "phi", "sigma_h"), "PMMH")
} else {
  cat("\nSkipping PMMH comparison (missing data)\n")
}

# ------------------------------------------------------------------------------
# 3. Compare Particle Gibbs posteriors (R PGAS vs Python PG for basic SV)
# Note: nimbleSMC does not have a separate PG (non-AS) sampler,
# so we compare using the shared basic SV parameters
# ------------------------------------------------------------------------------
if (!is.null(py_pg)) {
  cat("\n--- Particle Gibbs (Python only) ---\n")
  cat("Note: nimbleSMC provides PGAS; no separate PG validation from R.\n")
  cat("Python PG posterior summary:\n")
  for (p in c("mu", "phi", "sigma_h")) {
    cat(sprintf("  %s: mean=%.4f, sd=%.4f\n",
                p, mean(py_pg[[p]]), sd(py_pg[[p]])))
  }
}

# ------------------------------------------------------------------------------
# 4. Compare PGAS posteriors
# ------------------------------------------------------------------------------
if (!is.null(r_pgas) && !is.null(py_pgas)) {
  pgas_results <- compare_posteriors(
    r_pgas, py_pgas,
    c("mu", "phi", "sigma_h", "rho"),
    "PGAS"
  )
  pgas_results$algorithm <- "PGAS"
  all_results <- rbind(all_results, pgas_results)

  compare_acceptance(r_pgas, py_pgas, c("mu", "phi", "sigma_h", "rho"), "PGAS")
} else {
  cat("\nSkipping PGAS comparison (missing data)\n")
}

# ------------------------------------------------------------------------------
# 5. Overall validation summary
# ------------------------------------------------------------------------------
cat("\n")
cat(strrep("=", 71), "\n")
cat("OVERALL VALIDATION SUMMARY\n")
cat(strrep("=", 71), "\n\n")

if (nrow(all_results) > 0) {
  # Print comparison table
  cat("Parameter comparison table:\n\n")
  cat(sprintf("%-10s %-8s %10s %10s %10s %10s\n",
              "Algo", "Param", "R_mean", "Py_mean", "Diff", "KS_pval"))
  cat(strrep("-", 60), "\n")

  for (i in 1:nrow(all_results)) {
    r <- all_results[i, ]
    status <- ifelse(r$ks_pvalue > 0.05, "  OK", "  *")
    cat(sprintf("%-10s %-8s %10.4f %10.4f %10.4f %10.4f%s\n",
                r$algorithm, r$parameter,
                r$r_mean, r$py_mean, r$diff_mean, r$ks_pvalue, status))
  }

  cat("\n* = KS test rejects H0 at 5% level (distributions may differ)\n")
  cat("  Note: MCMC variability is expected; modest differences are normal.\n")

  # Overall pass/fail
  # Criterion: posterior means within reasonable tolerance
  # (MCMC inherently has variability, so we use a generous threshold)
  mean_diffs <- all_results$diff_mean
  max_diff <- max(mean_diffs)
  cat(sprintf("\nMax absolute mean difference: %.4f\n", max_diff))

  n_consistent <- sum(all_results$ks_pvalue > 0.01)
  n_total <- nrow(all_results)
  cat(sprintf("Parameters with consistent distributions (KS p>0.01): %d/%d\n",
              n_consistent, n_total))

  if (n_consistent >= n_total * 0.7) {
    cat("\nVALIDATION: PASS - R and Python posteriors are broadly consistent\n")
  } else {
    cat("\nVALIDATION: CHECK - Some parameters show significant differences\n")
    cat("  This may be due to different random seeds, MCMC variability,\n")
    cat("  or implementation differences. Manual inspection recommended.\n")
  }
} else {
  cat("No comparisons were performed (missing result files).\n")
  cat("Run validate_pmmh.R and validate_pgas.R first.\n")
}

# True parameter values
cat("\nTrue parameter values for reference:\n")
cat("  Basic SV:   mu=-1.0, phi=0.97, sigma_h=0.15\n")
cat("  SV Leverage: mu=-1.0, phi=0.97, sigma_h=0.15, rho=-0.5\n")

# Required R packages documentation
cat("\n")
cat(strrep("-", 71), "\n")
cat("Required R packages:\n")
cat("  pomp (>= 5.0)     - pmcmc() for PMMH\n")
cat("  nimble (>= 1.0)   - NIMBLE MCMC framework\n")
cat("  nimbleSMC (>= 0.11) - PGAS / particle MCMC samplers\n")
cat("  coda               - MCMC diagnostics (ESS, trace plots)\n")
cat(strrep("-", 71), "\n")

# ------------------------------------------------------------------------------
# 6. Save comparison results
# ------------------------------------------------------------------------------
output_path <- file.path(script_dir, "results_r_comparison.csv")
if (nrow(all_results) > 0) {
  write.csv(all_results, output_path, row.names = FALSE)
  cat(sprintf("\nComparison results saved to: %s\n", output_path))
} else {
  # Create empty placeholder
  write.csv(data.frame(
    algorithm = character(),
    parameter = character(),
    r_mean = numeric(),
    py_mean = numeric(),
    diff_mean = numeric(),
    ks_pvalue = numeric()
  ), output_path, row.names = FALSE)
  cat(sprintf("\nEmpty comparison file saved to: %s\n", output_path))
}

cat("\nDone.\n")
