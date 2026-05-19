#!/usr/bin/env Rscript
# ===========================================================================
# Comparison Report: Smoothers - particlefilterbox (Python) vs R validation
#
# Le os resultados CSV do Python (FFBS) e do R (FFBS) e gera tabelas
# comparativas de filtered vs smoothed estimates.
# Quantifica a melhoria do smoother sobre o filter em ambas implementacoes.
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
out_dir <- script_dir

read_if_exists <- function(path, label) {
  if (file.exists(path)) {
    cat(sprintf("[OK] %s encontrado\n", label))
    read.csv(path)
  } else {
    cat(sprintf("[WARN] %s nao encontrado: %s\n", label, path))
    NULL
  }
}

rmse <- function(est, true_val) {
  sqrt(mean((est - true_val)^2))
}

mae <- function(est, true_val) {
  mean(abs(est - true_val))
}

mean_var <- function(v) {
  mean(v)
}

# ---------------------------------------------------------------------------
# 1. Carregar resultados
# ---------------------------------------------------------------------------
cat(strrep("=", 70), "\n")
cat("  Smoothers Comparison: particlefilterbox (Python) vs R (pomp/FFBS)\n")
cat(strrep("=", 70), "\n\n")

# Python FFBS results
py_ffbs <- read_if_exists(
  file.path(solutions_dir, "results_ffbs.csv"), "Python FFBS results")

# Python two-filter / fixed-lag results
py_twofilter <- read_if_exists(
  file.path(solutions_dir, "results_two_filter_fixed_lag.csv"),
  "Python Two-filter/Fixed-lag results")

# R smoothed results
r_smoothed <- read_if_exists(
  file.path(script_dir, "results_r_smoothed.csv"), "R FFBS results")

# ---------------------------------------------------------------------------
# 2. Comparacao: Filter vs Smoother no R
# ---------------------------------------------------------------------------
cat("\n")
cat(strrep("-", 70), "\n")
cat("  1. R: Filter vs Smoother (FFBS)\n")
cat(strrep("-", 70), "\n\n")

comparison <- data.frame()

if (!is.null(r_smoothed)) {
  h_true <- r_smoothed$h_true

  r_filter_rmse <- rmse(r_smoothed$filtered_mean, h_true)
  r_filter_mae <- mae(r_smoothed$filtered_mean, h_true)
  r_filter_var <- mean_var(r_smoothed$filtered_var)

  r_smooth_rmse <- rmse(r_smoothed$smoothed_mean, h_true)
  r_smooth_mae <- mae(r_smoothed$smoothed_mean, h_true)
  r_smooth_var <- mean_var(r_smoothed$smoothed_var)

  r_improvement <- (r_filter_rmse - r_smooth_rmse) / r_filter_rmse * 100

  cat(sprintf("%-25s  %12s  %12s  %12s\n",
              "Metric", "R Filter", "R Smoother", "Improvement"))
  cat(strrep("-", 65), "\n")
  cat(sprintf("%-25s  %12.6f  %12.6f  %11.2f%%\n",
              "RMSE", r_filter_rmse, r_smooth_rmse, r_improvement))
  cat(sprintf("%-25s  %12.6f  %12.6f  %11.2f%%\n",
              "MAE", r_filter_mae, r_smooth_mae,
              (r_filter_mae - r_smooth_mae) / r_filter_mae * 100))
  cat(sprintf("%-25s  %12.6f  %12.6f\n",
              "Mean Variance", r_filter_var, r_smooth_var))

  comparison <- rbind(comparison, data.frame(
    Source = "R",
    Method = "Filter (BPF)",
    RMSE = r_filter_rmse,
    MAE = r_filter_mae,
    Mean_Var = r_filter_var
  ))
  comparison <- rbind(comparison, data.frame(
    Source = "R",
    Method = "Smoother (FFBS)",
    RMSE = r_smooth_rmse,
    MAE = r_smooth_mae,
    Mean_Var = r_smooth_var
  ))

  cat(sprintf("\n[%s] Smoother RMSE %s Filter RMSE\n",
              ifelse(r_smooth_rmse <= r_filter_rmse, "OK", "WARN"),
              ifelse(r_smooth_rmse <= r_filter_rmse, "<=", ">")))
} else {
  cat("R smoothed results not available. Run validate_ffbs.R first.\n")
}

# ---------------------------------------------------------------------------
# 3. Comparacao: Python Filter vs Smoother (FFBS)
# ---------------------------------------------------------------------------
cat("\n")
cat(strrep("-", 70), "\n")
cat("  2. Python: Filter vs Smoother (FFBS)\n")
cat(strrep("-", 70), "\n\n")

if (!is.null(py_ffbs)) {
  h_true_py <- py_ffbs$h_true

  py_filter_rmse <- rmse(py_ffbs$filtered_mean, h_true_py)
  py_filter_mae <- mae(py_ffbs$filtered_mean, h_true_py)
  py_filter_var <- mean_var(py_ffbs$filtered_var)

  # FFBSm
  py_ffbsm_rmse <- rmse(py_ffbs$ffbsm_smoothed_mean, h_true_py)
  py_ffbsm_mae <- mae(py_ffbs$ffbsm_smoothed_mean, h_true_py)
  py_ffbsm_var <- mean_var(py_ffbs$ffbsm_smoothed_var)

  # FFBSi
  py_ffbsi_rmse <- rmse(py_ffbs$ffbsi_smoothed_mean, h_true_py)
  py_ffbsi_mae <- mae(py_ffbs$ffbsi_smoothed_mean, h_true_py)
  py_ffbsi_var <- mean_var(py_ffbs$ffbsi_smoothed_var)

  cat(sprintf("%-25s  %12s  %12s  %12s\n",
              "Metric", "Py Filter", "Py FFBSm", "Py FFBSi"))
  cat(strrep("-", 65), "\n")
  cat(sprintf("%-25s  %12.6f  %12.6f  %12.6f\n",
              "RMSE", py_filter_rmse, py_ffbsm_rmse, py_ffbsi_rmse))
  cat(sprintf("%-25s  %12.6f  %12.6f  %12.6f\n",
              "MAE", py_filter_mae, py_ffbsm_mae, py_ffbsi_mae))
  cat(sprintf("%-25s  %12.6f  %12.6f  %12.6f\n",
              "Mean Variance", py_filter_var, py_ffbsm_var, py_ffbsi_var))

  comparison <- rbind(comparison, data.frame(
    Source = "Python",
    Method = "Filter (BPF)",
    RMSE = py_filter_rmse,
    MAE = py_filter_mae,
    Mean_Var = py_filter_var
  ))
  comparison <- rbind(comparison, data.frame(
    Source = "Python",
    Method = "Smoother (FFBSm)",
    RMSE = py_ffbsm_rmse,
    MAE = py_ffbsm_mae,
    Mean_Var = py_ffbsm_var
  ))
  comparison <- rbind(comparison, data.frame(
    Source = "Python",
    Method = "Smoother (FFBSi)",
    RMSE = py_ffbsi_rmse,
    MAE = py_ffbsi_mae,
    Mean_Var = py_ffbsi_var
  ))
} else {
  cat("Python FFBS results not available.\n")
}

# ---------------------------------------------------------------------------
# 4. Cross-comparison: R vs Python smoothed RMSE
# ---------------------------------------------------------------------------
cat("\n")
cat(strrep("-", 70), "\n")
cat("  3. Cross-Comparison: R Smoother vs Python Smoother\n")
cat(strrep("-", 70), "\n\n")

if (!is.null(r_smoothed) && !is.null(py_ffbs)) {
  # Comparar RMSE dos smoothers
  cat(sprintf("%-25s  %12s  %12s  %12s  %6s\n",
              "Comparison", "R RMSE", "Python RMSE", "Diff(%)", "Pass?"))
  cat(strrep("-", 72), "\n")

  # R FFBS vs Python FFBSm
  diff_pct_m <- abs(r_smooth_rmse - py_ffbsm_rmse) / py_ffbsm_rmse * 100
  pass_m <- diff_pct_m < 30  # tolerancia de 30% (Monte Carlo variabilidade)
  cat(sprintf("%-25s  %12.6f  %12.6f  %11.2f%%  %6s\n",
              "R FFBS vs Py FFBSm", r_smooth_rmse, py_ffbsm_rmse,
              diff_pct_m, ifelse(pass_m, "OK", "WARN")))

  # R FFBS vs Python FFBSi
  diff_pct_i <- abs(r_smooth_rmse - py_ffbsi_rmse) / py_ffbsi_rmse * 100
  pass_i <- diff_pct_i < 30
  cat(sprintf("%-25s  %12.6f  %12.6f  %11.2f%%  %6s\n",
              "R FFBS vs Py FFBSi", r_smooth_rmse, py_ffbsi_rmse,
              diff_pct_i, ifelse(pass_i, "OK", "WARN")))

  # R Filter vs Python Filter
  diff_filt <- abs(r_filter_rmse - py_filter_rmse) / py_filter_rmse * 100
  pass_f <- diff_filt < 30
  cat(sprintf("%-25s  %12.6f  %12.6f  %11.2f%%  %6s\n",
              "R Filter vs Py Filter", r_filter_rmse, py_filter_rmse,
              diff_filt, ifelse(pass_f, "OK", "WARN")))

  cat(sprintf("\nNota: tolerancia de 30%% devida a variabilidade Monte Carlo\n"))
  cat(sprintf("(seeds diferentes entre R e Python)\n"))

  # Verificacao qualitativa: ambos mostram melhoria smoother > filter?
  cat(sprintf("\n--- Verificacao qualitativa ---\n"))
  r_shows_improvement <- r_smooth_rmse <= r_filter_rmse
  py_shows_improvement <- py_ffbsm_rmse <= py_filter_rmse

  cat(sprintf("R smoother melhora sobre filter:      %s\n",
              ifelse(r_shows_improvement, "SIM", "NAO")))
  cat(sprintf("Python smoother melhora sobre filter:  %s\n",
              ifelse(py_shows_improvement, "SIM", "NAO")))

  if (r_shows_improvement && py_shows_improvement) {
    cat("[OK] Ambas implementacoes mostram melhoria do smoother\n")
  }
} else {
  cat("Resultados insuficientes para comparacao cruzada.\n")
  cat("Execute validate_ffbs.R e os scripts Python primeiro.\n")
}

# ---------------------------------------------------------------------------
# 5. Tabela comparativa completa
# ---------------------------------------------------------------------------
cat("\n")
cat(strrep("-", 70), "\n")
cat("  4. Tabela Comparativa Completa\n")
cat(strrep("-", 70), "\n\n")

if (nrow(comparison) > 0) {
  cat(sprintf("%-10s  %-20s  %12s  %12s  %12s\n",
              "Source", "Method", "RMSE", "MAE", "Mean_Var"))
  cat(strrep("-", 70), "\n")
  for (i in seq_len(nrow(comparison))) {
    cat(sprintf("%-10s  %-20s  %12.6f  %12.6f  %12.6f\n",
                comparison$Source[i],
                comparison$Method[i],
                comparison$RMSE[i],
                comparison$MAE[i],
                comparison$Mean_Var[i]))
  }
}

# ---------------------------------------------------------------------------
# 6. Salvar resultados de comparacao
# ---------------------------------------------------------------------------
write.csv(comparison, file.path(out_dir, "results_r_comparison.csv"),
          row.names = FALSE)
cat(sprintf("\nResultados de comparacao salvos em: %s\n",
            file.path(out_dir, "results_r_comparison.csv")))

# ---------------------------------------------------------------------------
# 7. Sumario final
# ---------------------------------------------------------------------------
cat("\n")
cat(strrep("=", 70), "\n")
cat("  Summary\n")
cat(strrep("=", 70), "\n\n")

cat("Pacotes R utilizados para validacao:\n")
cat("  - pomp (>= 5.0): Particle filter (pfilter) para forward pass\n")
cat("  - FFBS backward sampling: implementacao manual\n")
cat("\nModelo validado:\n")
cat("  - Stochastic Volatility: h_t = mu + phi*(h_{t-1}-mu) + sigma_h*w_t\n")
cat("  - Parametros: mu=-1.0, phi=0.97, sigma_h=0.15\n")
cat("\nMetodos comparados:\n")
cat("  - R: BPF (filter) vs FFBS (smoother)\n")
cat("  - Python: BPF (filter) vs FFBSm vs FFBSi\n")
cat("\nCriterio de aceite:\n")
cat("  - Smoother RMSE <= Filter RMSE (ambas implementacoes)\n")
cat("  - RMSE cross-implementation: diferenca < 30%% (Monte Carlo variabilidade)\n")

cat(sprintf("\n--- Report generated %s ---\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
