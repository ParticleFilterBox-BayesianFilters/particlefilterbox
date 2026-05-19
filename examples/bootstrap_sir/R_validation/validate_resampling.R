#!/usr/bin/env Rscript
# ===========================================================================
# Validacao: Comparacao de metodos de resampling
# Usa o modelo SV com pomp para comparar multinomial vs systematic
#
# Nota sobre metodos de resampling no pomp:
#   pomp usa systematic resampling por padrao. Para testar multinomial,
#   usamos uma implementacao manual. O pacote SMC oferece funcoes auxiliares
#   de resampling que tambem sao comparadas aqui.
#
# Pacotes requeridos: pomp (>= 5.0)
# Instalar: install.packages("pomp")
# ===========================================================================

suppressPackageStartupMessages({
  library(pomp)
})

set.seed(42)

# ---------------------------------------------------------------------------
# 1. Carregar dados SV
# ---------------------------------------------------------------------------
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) > 0) {
  script_dir <- dirname(normalizePath(script_path))
} else {
  script_dir <- getwd()
}
data_dir <- file.path(script_dir, "..", "data")
out_dir <- script_dir

df <- read.csv(file.path(data_dir, "simulated_sv.csv"))
cat(sprintf("Dados SV carregados: %d observacoes\n", nrow(df)))

# Parametros do modelo SV
mu <- -1.0
phi_sv <- 0.97
sigma_h <- 0.15

# ---------------------------------------------------------------------------
# 2. Definir modelo SV no pomp
# ---------------------------------------------------------------------------
sv_model <- pomp(
  data = data.frame(time = df$t, y = df$y_obs),
  times = "time",
  t0 = -1,
  rprocess = discrete_time(
    step.fun = function(h, ..., mu = -1.0, phi = 0.97, sigma_h = 0.15) {
      c(h = rnorm(1,
                   mean = mu + phi * (h - mu),
                   sd = sigma_h))
    },
    delta.t = 1
  ),
  dmeasure = function(y, h, ..., log) {
    dnorm(y, mean = 0, sd = exp(h / 2), log = log)
  },
  rinit = function(..., mu = -1.0, phi = 0.97, sigma_h = 0.15) {
    c(h = rnorm(1,
                mean = mu,
                sd = sigma_h / sqrt(1 - phi^2)))
  },
  statenames = "h",
  paramnames = c("mu", "phi", "sigma_h")
)

params_sv <- c(mu = mu, phi = phi_sv, sigma_h = sigma_h)

# ---------------------------------------------------------------------------
# 3. Implementacoes de resampling para comparacao
# ---------------------------------------------------------------------------

# Multinomial resampling
resample_multinomial <- function(weights) {
  n <- length(weights)
  sample.int(n, size = n, replace = TRUE, prob = weights)
}

# Systematic resampling
resample_systematic <- function(weights) {
  n <- length(weights)
  cw <- cumsum(weights)
  u <- (runif(1) + seq(0, n - 1)) / n
  indices <- integer(n)
  j <- 1
  for (i in seq_len(n)) {
    while (cw[j] < u[i]) j <- j + 1
    indices[i] <- j
  }
  indices
}

# Stratified resampling
resample_stratified <- function(weights) {
  n <- length(weights)
  cw <- cumsum(weights)
  u <- (runif(n) + seq(0, n - 1)) / n
  indices <- integer(n)
  j <- 1
  for (i in seq_len(n)) {
    while (cw[j] < u[i]) j <- j + 1
    indices[i] <- j
  }
  indices
}

# Residual resampling
resample_residual <- function(weights) {
  n <- length(weights)
  nw <- weights * n
  det_counts <- floor(nw)
  residuals <- nw - det_counts
  n_det <- sum(det_counts)
  n_res <- n - n_det

  # Deterministic part
  indices <- rep(seq_len(n), times = det_counts)

  # Stochastic part (multinomial on residuals)
  if (n_res > 0) {
    res_probs <- residuals / sum(residuals)
    extra <- sample.int(n, size = n_res, replace = TRUE, prob = res_probs)
    indices <- c(indices, extra)
  }
  indices
}

# ---------------------------------------------------------------------------
# 4. Testar uniformidade dos metodos de resampling
# ---------------------------------------------------------------------------
cat("\n=== Teste de uniformidade dos metodos de resampling ===\n")
cat("(Pesos uniformes => indices devem ser uniformes)\n\n")

n_test <- 1000
n_reps <- 500
uniform_weights <- rep(1 / n_test, n_test)

methods <- list(
  multinomial = resample_multinomial,
  systematic = resample_systematic,
  stratified = resample_stratified,
  residual = resample_residual
)

for (name in names(methods)) {
  counts <- integer(n_test)
  for (r in seq_len(n_reps)) {
    idx <- methods[[name]](uniform_weights)
    counts <- counts + tabulate(idx, nbins = n_test)
  }
  freq <- counts / sum(counts)
  chi_sq <- sum((freq - 1 / n_test)^2 / (1 / n_test)) * sum(counts)
  cat(sprintf("%-15s: chi-sq=%.2f (df=%d), max_dev=%.6f\n",
              name, chi_sq, n_test - 1, max(abs(freq - 1 / n_test))))
}

# ---------------------------------------------------------------------------
# 5. Comparar metodos usando pomp pfilter (N=1000, varias repeticoes)
# ---------------------------------------------------------------------------
cat("\n=== Comparacao de resampling no modelo SV (pomp pfilter) ===\n")
cat("Nota: pomp usa systematic resampling internamente.\n")
cat("Rodamos multiplas repeticoes para estimar variancia do log-likelihood.\n\n")

N_particles <- 1000
n_rep <- 10

results <- data.frame()

# pomp pfilter (systematic resampling, padrao do pomp)
logliks <- numeric(n_rep)
ess_means <- numeric(n_rep)
for (r in seq_len(n_rep)) {
  pf <- pfilter(sv_model, Np = N_particles, params = params_sv)
  logliks[r] <- logLik(pf)
  ess_means[r] <- mean(eff_sample_size(pf))
}

# Relatar resultado do systematic (pomp default)
results <- rbind(results, data.frame(
  method = "systematic",
  loglik_mean = mean(logliks),
  loglik_sd = sd(logliks),
  ESS_mean = mean(ess_means),
  ESS_N_pct = mean(ess_means) / N_particles * 100
))
cat(sprintf("systematic (pomp): logLik=%.2f +/- %.2f, ESS=%.1f (%.1f%%)\n",
            mean(logliks), sd(logliks), mean(ess_means),
            mean(ess_means) / N_particles * 100))

# Para os outros metodos, reportamos o mesmo resultado do pomp como referencia,
# pois pomp nao permite trocar o metodo de resampling diretamente.
# A comparacao real entre metodos e feita no particlefilterbox (Python).
# Aqui documentamos que o pomp usa systematic e fornecemos a referencia.
cat("\nNota: pomp usa systematic resampling fixo.\n")
cat("Comparacao entre metodos (multinomial, stratified, residual) e feita\n")
cat("usando as implementacoes R puras acima para validar as estatisticas.\n\n")

# Testar variancia do log-likelihood estimado com cada metodo de resampling
# usando implementacao manual do bootstrap PF
cat("=== Bootstrap PF manual com diferentes resampling (SV model) ===\n\n")

y_obs <- df$y_obs
T_len <- length(y_obs)

run_manual_bpf <- function(resample_fn, N, y, mu, phi, sigma_h) {
  # Inicializacao
  particles <- rnorm(N, mean = mu, sd = sigma_h / sqrt(1 - phi^2))
  log_lik <- 0

  ess_vec <- numeric(T_len)

  for (tt in seq_along(y)) {
    # Propagacao (prior)
    if (tt > 1) {
      particles <- rnorm(N, mean = mu + phi * (particles - mu), sd = sigma_h)
    }

    # Pesos (log)
    log_w <- dnorm(y[tt], mean = 0, sd = exp(particles / 2), log = TRUE)
    max_lw <- max(log_w)
    w <- exp(log_w - max_lw)
    w_norm <- w / sum(w)

    # Log-likelihood incremental
    log_lik <- log_lik + max_lw + log(mean(w))

    # ESS
    ess_vec[tt] <- 1 / sum(w_norm^2)

    # Resampling
    idx <- resample_fn(w_norm)
    particles <- particles[idx]
  }

  list(loglik = log_lik, ess_mean = mean(ess_vec))
}

for (method_name in names(methods)) {
  logliks_m <- numeric(n_rep)
  ess_m <- numeric(n_rep)
  for (r in seq_len(n_rep)) {
    res <- run_manual_bpf(methods[[method_name]], N_particles, y_obs,
                          mu, phi_sv, sigma_h)
    logliks_m[r] <- res$loglik
    ess_m[r] <- res$ess_mean
  }
  results <- rbind(results, data.frame(
    method = method_name,
    loglik_mean = mean(logliks_m),
    loglik_sd = sd(logliks_m),
    ESS_mean = mean(ess_m),
    ESS_N_pct = mean(ess_m) / N_particles * 100
  ))
  cat(sprintf("%-15s: logLik=%.2f +/- %.2f, ESS=%.1f (%.1f%%)\n",
              method_name, mean(logliks_m), sd(logliks_m),
              mean(ess_m), mean(ess_m) / N_particles * 100))
}

# ---------------------------------------------------------------------------
# 6. Salvar resultados
# ---------------------------------------------------------------------------
write.csv(results, file.path(out_dir, "results_r_resampling.csv"), row.names = FALSE)
cat(sprintf("\nResultados salvos em: %s\n",
            file.path(out_dir, "results_r_resampling.csv")))
