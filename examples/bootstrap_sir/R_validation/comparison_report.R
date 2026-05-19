#!/usr/bin/env Rscript
# ===========================================================================
# Comparison Report: particlefilterbox (Python) vs R validation
#
# Le os resultados CSV do Python e do R e gera tabelas de comparacao.
# Verifica o criterio: |loglik_R - loglik_Python| / |loglik_Python| < 5%
#
# Pacotes requeridos: nenhum (base R apenas)
# ===========================================================================

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) > 0) {
  script_dir <- dirname(normalizePath(script_path))
} else {
  script_dir <- getwd()
}
solutions_dir <- file.path(script_dir, "..", "solutions")

read_if_exists <- function(path, label) {
  if (file.exists(path)) {
    cat(sprintf("[OK] %s encontrado\n", label))
    read.csv(path)
  } else {
    cat(sprintf("[WARN] %s nao encontrado: %s\n", label, path))
    NULL
  }
}

pct_diff <- function(a, b) {
  abs(a - b) / abs(b) * 100
}

# ---------------------------------------------------------------------------
# 1. Carregar resultados
# ---------------------------------------------------------------------------
cat(strrep("=", 62), "\n")
cat("  Comparison Report: particlefilterbox vs R validation\n")
cat(strrep("=", 62), "\n\n")

# Python results
py_bootstrap <- read_if_exists(
  file.path(solutions_dir, "results_bootstrap_pf.csv"), "Python Bootstrap PF")
py_sir <- read_if_exists(
  file.path(solutions_dir, "results_sir.csv"), "Python SIR")
py_resampling <- read_if_exists(
  file.path(solutions_dir, "results_resampling.csv"), "Python Resampling")

# R results
r_bootstrap <- read_if_exists(
  file.path(script_dir, "results_r_bootstrap.csv"), "R Bootstrap PF")
r_sir <- read_if_exists(
  file.path(script_dir, "results_r_sir.csv"), "R SIR")
r_resampling <- read_if_exists(
  file.path(script_dir, "results_r_resampling.csv"), "R Resampling")

# ---------------------------------------------------------------------------
# 2. Comparacao: Bootstrap PF (Linear-Gaussian)
# ---------------------------------------------------------------------------
cat("\n")
cat(strrep("-", 62), "\n")
cat("  1. Bootstrap PF - Linear-Gaussian Model\n")
cat(strrep("-", 62), "\n\n")

if (!is.null(py_bootstrap) && !is.null(r_bootstrap)) {
  # Alinhar por N
  common_N <- intersect(py_bootstrap$N, r_bootstrap$N)

  cat(sprintf("%-8s  %12s  %12s  %10s  %6s\n",
              "N", "Python logL", "R logL", "Diff(%)", "Pass?"))
  cat(strrep("-", 55), "\n")

  all_pass <- TRUE
  for (n in common_N) {
    py_ll <- py_bootstrap$log_likelihood[py_bootstrap$N == n]
    r_ll <- r_bootstrap$loglik[r_bootstrap$N == n]
    diff_pct <- pct_diff(r_ll, py_ll)
    pass <- diff_pct < 5
    if (!pass) all_pass <- FALSE
    cat(sprintf("%-8d  %12.2f  %12.2f  %9.2f%%  %6s\n",
                n, py_ll, r_ll, diff_pct, ifelse(pass, "OK", "FAIL")))
  }

  # Kalman reference
  if ("kalman_loglik" %in% names(r_bootstrap)) {
    cat(sprintf("\nKalman filter logLik (KFAS): %.4f\n", r_bootstrap$kalman_loglik[1]))
  }

  cat(sprintf("\nBootstrap PF validation: %s\n",
              ifelse(all_pass, "PASSED", "FAILED")))
} else {
  cat("Resultados insuficientes para comparacao.\n")
}

# ---------------------------------------------------------------------------
# 3. Comparacao: SIR (Stochastic Volatility)
# ---------------------------------------------------------------------------
cat("\n")
cat(strrep("-", 62), "\n")
cat("  2. SIR - Stochastic Volatility Model\n")
cat(strrep("-", 62), "\n\n")

if (!is.null(py_sir) && !is.null(r_sir)) {
  common_N <- intersect(py_sir$N, r_sir$N)

  cat(sprintf("%-8s  %12s  %12s  %10s  %6s\n",
              "N", "Python logL", "R logL", "Diff(%)", "Pass?"))
  cat(strrep("-", 55), "\n")

  all_pass <- TRUE
  for (n in common_N) {
    py_ll <- py_sir$log_likelihood[py_sir$N == n]
    r_ll <- r_sir$loglik[r_sir$N == n]
    diff_pct <- pct_diff(r_ll, py_ll)
    pass <- diff_pct < 5
    if (!pass) all_pass <- FALSE
    cat(sprintf("%-8d  %12.2f  %12.2f  %9.2f%%  %6s\n",
                n, py_ll, r_ll, diff_pct, ifelse(pass, "OK", "FAIL")))
  }

  cat(sprintf("\nSIR validation: %s\n",
              ifelse(all_pass, "PASSED", "FAILED")))
} else {
  cat("Resultados insuficientes para comparacao.\n")
}

# ---------------------------------------------------------------------------
# 4. Comparacao: Resampling methods
# ---------------------------------------------------------------------------
cat("\n")
cat(strrep("-", 62), "\n")
cat("  3. Resampling Methods Comparison\n")
cat(strrep("-", 62), "\n\n")

if (!is.null(py_resampling) && !is.null(r_resampling)) {
  # Python columns: method, RMSE_mean, RMSE_std, ESS_mean, ESS_N_pct, time_s
  # R columns: method, loglik_mean, loglik_sd, ESS_mean, ESS_N_pct

  common_methods <- intersect(
    tolower(py_resampling$method),
    tolower(r_resampling$method)
  )

  cat(sprintf("%-15s  %10s  %10s  %10s  %10s\n",
              "Method", "Py ESS(%)", "R ESS(%)", "Py RMSE", "R logL_sd"))
  cat(strrep("-", 60), "\n")

  for (m in common_methods) {
    py_row <- py_resampling[tolower(py_resampling$method) == m, ]
    r_row <- r_resampling[tolower(r_resampling$method) == m, ]
    cat(sprintf("%-15s  %10.1f  %10.1f  %10.4f  %10.2f\n",
                m,
                py_row$ESS_N_pct[1],
                r_row$ESS_N_pct[1],
                py_row$RMSE_mean[1],
                r_row$loglik_sd[1]))
  }
} else {
  cat("Resultados insuficientes para comparacao.\n")
}

# ---------------------------------------------------------------------------
# 5. Sumario final
# ---------------------------------------------------------------------------
cat("\n")
cat(strrep("=", 62), "\n")
cat("  Summary\n")
cat(strrep("=", 62), "\n\n")

cat("Pacotes R utilizados para validacao:\n")
cat("  - pomp: Particle filter (pfilter) para Bootstrap PF e SIR\n")
cat("  - KFAS: Kalman filter para benchmark exato (linear-Gaussian)\n")
cat("  - SMC:  Referencia para metodos de resampling\n")
cat("\nModelos validados:\n")
cat("  1. Linear-Gaussian (Bootstrap PF): Python vs R (pomp) vs Kalman (KFAS)\n")
cat("  2. Stochastic Volatility (SIR): Python vs R (pomp)\n")
cat("  3. Resampling: multinomial, systematic, stratified, residual\n")
cat("\nCriterio de aceite: |loglik_R - loglik_Python| / |loglik_Python| < 5%%\n")
cat("(Para N=5000, ambas implementacoes devem convergir para valores proximos)\n")

cat("\n--- Report generated", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "---\n")
