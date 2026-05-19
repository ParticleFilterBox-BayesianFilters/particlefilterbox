#!/usr/bin/env Rscript
# ===========================================================================
# Comparison Report: particlefilterbox (Python) vs R validation - SMC
#
# Le os resultados CSV do Python e do R e gera tabelas de comparacao.
# Criterio: |logev_R - logev_Python| / |logev_Python| < 10%
#
# Pacotes requeridos: nenhum (base R apenas)
# Pacotes R documentados: SMC, pomp
#
# Uso: Rscript validate_smc_comparison.R
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
    cat(sprintf("[OK]   %s encontrado\n", label))
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
cat(strrep("=", 66), "\n")
cat("  Comparison Report: particlefilterbox vs R - SMC Algorithms\n")
cat(strrep("=", 66), "\n\n")

# Python results
py_sampler <- read_if_exists(
  file.path(solutions_dir, "results_smc_sampler.csv"), "Python SMC Sampler")
py_smc2 <- read_if_exists(
  file.path(solutions_dir, "results_smc2.csv"), "Python SMC^2")

# R results
r_sampler <- read_if_exists(
  file.path(script_dir, "results_r_smc_sampler.csv"), "R SMC Sampler")
r_smc2 <- read_if_exists(
  file.path(script_dir, "results_r_smc2.csv"), "R SMC^2")

cat("\n")

# ---------------------------------------------------------------------------
# 2. Comparacao: SMC Sampler (Linear-Gaussian)
# ---------------------------------------------------------------------------
cat(strrep("-", 66), "\n")
cat("  1. SMC Sampler - Linear-Gaussian Model\n")
cat(strrep("-", 66), "\n\n")

comparison_rows <- list()
all_pass_sampler <- TRUE

if (!is.null(py_sampler) && !is.null(r_sampler)) {
  # Parameter comparison
  param_rows <- c("phi", "sigma_x", "sigma_y")

  cat(sprintf("%-12s  %10s  %10s  %10s  %10s  %6s\n",
              "Parameter", "True", "Python", "R", "Diff(%)", "Pass?"))
  cat(strrep("-", 66), "\n")

  for (p in param_rows) {
    py_row <- py_sampler[py_sampler$parameter == p, ]
    r_row  <- r_sampler[r_sampler$parameter == p, ]

    if (nrow(py_row) > 0 && nrow(r_row) > 0) {
      py_mean <- py_row$smc_mean[1]
      r_mean  <- r_row$smc_mean[1]
      true_v  <- py_row$true_value[1]
      diff_p  <- pct_diff(r_mean, py_mean)

      # For parameters, check if both are within reasonable range of true value
      pass <- diff_p < 30  # 30% tolerance for MCMC estimates
      if (!pass) all_pass_sampler <- FALSE

      cat(sprintf("%-12s  %10.4f  %10.4f  %10.4f  %9.1f%%  %6s\n",
                  p, true_v, py_mean, r_mean, diff_p,
                  ifelse(pass, "OK", "FAIL")))

      comparison_rows[[length(comparison_rows) + 1]] <- data.frame(
        method    = "smc_sampler",
        parameter = p,
        true_value = true_v,
        python_mean = py_mean,
        python_std  = py_row$smc_std[1],
        r_mean      = r_mean,
        r_std       = r_row$smc_std[1],
        diff_pct    = diff_p,
        pass        = pass,
        stringsAsFactors = FALSE
      )
    }
  }

  # Log-evidence comparison
  py_le <- py_sampler[py_sampler$parameter == "log_evidence", ]
  r_le  <- r_sampler[r_sampler$parameter == "log_evidence", ]

  if (nrow(py_le) > 0 && nrow(r_le) > 0) {
    py_ev <- py_le$smc_mean[1]
    r_ev  <- r_le$smc_mean[1]
    diff_ev <- pct_diff(r_ev, py_ev)
    pass_ev <- diff_ev < 10  # 10% tolerance for log-evidence
    if (!pass_ev) all_pass_sampler <- FALSE

    cat(sprintf("\n%-12s  %10s  %10.2f  %10.2f  %9.1f%%  %6s\n",
                "log_evid", "", py_ev, r_ev, diff_ev,
                ifelse(pass_ev, "OK", "FAIL")))

    comparison_rows[[length(comparison_rows) + 1]] <- data.frame(
      method = "smc_sampler", parameter = "log_evidence",
      true_value = NA, python_mean = py_ev, python_std = NA,
      r_mean = r_ev, r_std = NA, diff_pct = diff_ev, pass = pass_ev,
      stringsAsFactors = FALSE
    )
  }

  cat(sprintf("\nSMC Sampler validation: %s\n",
              ifelse(all_pass_sampler, "PASSED", "FAILED")))
} else {
  cat("Resultados insuficientes para comparacao.\n")
}

# ---------------------------------------------------------------------------
# 3. Comparacao: SMC^2 (Stochastic Volatility)
# ---------------------------------------------------------------------------
cat("\n")
cat(strrep("-", 66), "\n")
cat("  2. SMC^2 - Stochastic Volatility Model\n")
cat(strrep("-", 66), "\n\n")

all_pass_smc2 <- TRUE

if (!is.null(py_smc2) && !is.null(r_smc2)) {
  param_rows <- c("mu", "phi", "sigma_h")

  cat(sprintf("%-12s  %10s  %10s  %10s  %10s  %6s\n",
              "Parameter", "True", "Python", "R", "Diff(%)", "Pass?"))
  cat(strrep("-", 66), "\n")

  for (p in param_rows) {
    py_row <- py_smc2[py_smc2$parameter == p, ]
    r_row  <- r_smc2[r_smc2$parameter == p, ]

    if (nrow(py_row) > 0 && nrow(r_row) > 0) {
      py_mean <- py_row$smc2_mean[1]
      r_mean  <- r_row$smc2_mean[1]
      true_v  <- py_row$true_value[1]
      diff_p  <- pct_diff(r_mean, py_mean)

      pass <- diff_p < 50  # 50% tolerance for SMC^2 (high variance)
      if (!pass) all_pass_smc2 <- FALSE

      cat(sprintf("%-12s  %10.4f  %10.4f  %10.4f  %9.1f%%  %6s\n",
                  p, true_v, py_mean, r_mean, diff_p,
                  ifelse(pass, "OK", "FAIL")))

      comparison_rows[[length(comparison_rows) + 1]] <- data.frame(
        method = "smc2", parameter = p,
        true_value = true_v, python_mean = py_mean, python_std = py_row$smc2_std[1],
        r_mean = r_mean, r_std = r_row$smc2_std[1],
        diff_pct = diff_p, pass = pass,
        stringsAsFactors = FALSE
      )
    }
  }

  # Log-evidence comparison
  py_le <- py_smc2[py_smc2$parameter == "log_evidence", ]
  r_le  <- r_smc2[r_smc2$parameter == "log_evidence", ]

  if (nrow(py_le) > 0 && nrow(r_le) > 0) {
    py_ev <- py_le$smc2_mean[1]
    r_ev  <- r_le$smc2_mean[1]
    diff_ev <- pct_diff(r_ev, py_ev)
    pass_ev <- diff_ev < 10
    if (!pass_ev) all_pass_smc2 <- FALSE

    cat(sprintf("\n%-12s  %10s  %10.2f  %10.2f  %9.1f%%  %6s\n",
                "log_evid", "", py_ev, r_ev, diff_ev,
                ifelse(pass_ev, "OK", "FAIL")))

    comparison_rows[[length(comparison_rows) + 1]] <- data.frame(
      method = "smc2", parameter = "log_evidence",
      true_value = NA, python_mean = py_ev, python_std = NA,
      r_mean = r_ev, r_std = NA, diff_pct = diff_ev, pass = pass_ev,
      stringsAsFactors = FALSE
    )
  }

  cat(sprintf("\nSMC^2 validation: %s\n",
              ifelse(all_pass_smc2, "PASSED", "FAILED")))
} else {
  cat("Resultados insuficientes para comparacao.\n")
}

# ---------------------------------------------------------------------------
# 4. Salvar tabela de comparacao
# ---------------------------------------------------------------------------
if (length(comparison_rows) > 0) {
  comparison_df <- do.call(rbind, comparison_rows)
  out_file <- file.path(script_dir, "results_r_comparison.csv")
  write.csv(comparison_df, out_file, row.names = FALSE)
  cat(sprintf("\nTabela de comparacao salva em: %s\n", out_file))
} else {
  cat("\nNenhum resultado para salvar.\n")
}

# ---------------------------------------------------------------------------
# 5. Sumario final
# ---------------------------------------------------------------------------
cat("\n")
cat(strrep("=", 66), "\n")
cat("  Summary\n")
cat(strrep("=", 66), "\n\n")

cat("Pacotes R utilizados para validacao SMC:\n")
cat("  - KFAS: Kalman filter para log-likelihood exata (linear-Gaussian)\n")
cat("  - pomp: Particle filter de referencia (SV model)\n")
cat("  - SMC:  Pacote de referencia para SMC samplers (CRAN)\n")
cat("\nAlgoritmos validados:\n")
cat("  1. SMC Sampler (tempering adaptativo): Linear-Gaussian model\n")
cat("  2. SMC^2 (SMC dentro de SMC): Stochastic Volatility model\n")
cat("\nCriterios de aceite:\n")
cat("  - Log-evidencia: |R - Python| / |Python| < 10%%\n")
cat("  - Parametros: estimativas dentro de intervalos similares\n")

overall <- TRUE
if (!is.null(py_sampler) && !is.null(r_sampler)) {
  cat(sprintf("\n  SMC Sampler: %s\n", ifelse(all_pass_sampler, "PASSED", "FAILED")))
  if (!all_pass_sampler) overall <- FALSE
}
if (!is.null(py_smc2) && !is.null(r_smc2)) {
  cat(sprintf("  SMC^2:       %s\n", ifelse(all_pass_smc2, "PASSED", "FAILED")))
  if (!all_pass_smc2) overall <- FALSE
}

cat(sprintf("\n  Overall: %s\n", ifelse(overall, "PASSED", "FAILED")))
cat(sprintf("\n--- Report generated %s ---\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
