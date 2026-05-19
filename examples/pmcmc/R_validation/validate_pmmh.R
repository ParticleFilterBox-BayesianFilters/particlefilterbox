#!/usr/bin/env Rscript
# ==============================================================================
# validate_pmmh.R
# PMMH (Particle Marginal Metropolis-Hastings) via pomp::pmcmc
# Cross-validation of particlefilterbox PMMH on the stochastic volatility model
#
# Required R packages: pomp (>= 5.0), coda
# Install: install.packages(c("pomp", "coda"))
#
# Usage: Rscript validate_pmmh.R
# ==============================================================================

suppressPackageStartupMessages({
  library(pomp)
  library(coda)
})

cat(strrep("=", 71), "\n")
cat("PMMH Validation via pomp::pmcmc\n")
cat("Model: Stochastic Volatility (basic)\n")
cat("True parameters: mu=-1.0, phi=0.97, sigma_h=0.15\n")
cat(strrep("=", 71), "\n\n")

# ------------------------------------------------------------------------------
# 1. Load data
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

data_path <- file.path(script_dir, "..", "data", "simulated_sv.csv")
if (!file.exists(data_path)) {
  data_path <- file.path("examples", "pmcmc", "data", "simulated_sv.csv")
}

cat("Loading data from:", data_path, "\n")
sv_data <- read.csv(data_path)

# Use first 200 observations (same as Python)
n_obs <- min(200, nrow(sv_data))
sv_data <- sv_data[1:n_obs, ]
cat("Using", n_obs, "observations\n\n")

# ------------------------------------------------------------------------------
# 2. Define SV model in pomp
# ------------------------------------------------------------------------------
# Model:
#   h_t = mu + phi * (h_{t-1} - mu) + sigma_h * eta_t,  eta_t ~ N(0,1)
#   y_t = exp(h_t / 2) * eps_t,                          eps_t ~ N(0,1)
#
# Parameterization: we estimate (mu, logit_phi, log_sigma_h) on unconstrained
# scale to ensure phi in (0,1) and sigma_h > 0.

sv_rinit <- Csnippet("
  double phi = 1.0 / (1.0 + exp(-logit_phi));
  double sigma_h = exp(log_sigma_h);
  h = mu;
")

sv_rprocess <- Csnippet("
  double phi = 1.0 / (1.0 + exp(-logit_phi));
  double sigma_h = exp(log_sigma_h);
  h = mu + phi * (h - mu) + sigma_h * rnorm(0, 1);
")

sv_dmeasure <- Csnippet("
  double sd = exp(h / 2.0);
  lik = dnorm(y_obs, 0.0, sd, give_log);
")

sv_rmeasure <- Csnippet("
  double phi = 1.0 / (1.0 + exp(-logit_phi));
  double sigma_h = exp(log_sigma_h);
  y_obs = rnorm(0, exp(h / 2.0));
")

# Create pomp object (no partrans needed - we estimate on unconstrained scale)
sv_pomp <- pomp(
  data = data.frame(time = sv_data$t, y_obs = sv_data$y_obs),
  times = "time",
  t0 = 0,
  rinit = sv_rinit,
  rprocess = discrete_time(sv_rprocess, delta.t = 1),
  dmeasure = sv_dmeasure,
  rmeasure = sv_rmeasure,
  statenames = "h",
  paramnames = c("mu", "logit_phi", "log_sigma_h")
)

# ------------------------------------------------------------------------------
# 3. Run PMMH via pomp::pmcmc
# ------------------------------------------------------------------------------
n_particles <- 200
n_mcmc <- 2000
n_burnin <- 500

cat("PMMH configuration:\n")
cat("  Particles:", n_particles, "\n")
cat("  MCMC iterations:", n_mcmc, "\n")
cat("  Burn-in:", n_burnin, "\n\n")

# Starting values on unconstrained scale
# phi=0.97 -> logit(0.97) = log(0.97/0.03) = 3.476
# sigma_h=0.15 -> log(0.15) = -1.897
start_params <- c(mu = -1.0,
                  logit_phi = log(0.97 / 0.03),
                  log_sigma_h = log(0.15))

# Proposal standard deviations (random walk on unconstrained scale)
# Tuned for ~0.20-0.30 acceptance rate
proposal_sd <- c(mu = 0.3, logit_phi = 0.5, log_sigma_h = 0.5)

set.seed(42)

cat("Running PMMH... (this may take several minutes)\n")
t_start <- proc.time()

pmcmc_out <- pmcmc(
  sv_pomp,
  Nmcmc = n_mcmc,
  Np = n_particles,
  params = start_params,
  proposal = mvn_diag_rw(rw.sd = proposal_sd)
)

t_elapsed <- (proc.time() - t_start)[3]
cat(sprintf("PMMH completed in %.1f seconds\n\n", t_elapsed))

# ------------------------------------------------------------------------------
# 4. Extract posterior samples
# ------------------------------------------------------------------------------
chain_raw <- as.data.frame(traces(pmcmc_out))

# Remove burn-in
chain_raw <- chain_raw[(n_burnin + 1):nrow(chain_raw), ]

# Back-transform to natural scale
chain_post <- data.frame(
  mu = chain_raw$mu,
  phi = 1.0 / (1.0 + exp(-chain_raw$logit_phi)),
  sigma_h = exp(chain_raw$log_sigma_h)
)
cat("Posterior samples (after burn-in):", nrow(chain_post), "\n\n")

# Posterior summaries
cat("Posterior summaries:\n")
cat(sprintf("  mu:      mean=%.4f, sd=%.4f, 95%% CI=[%.4f, %.4f]\n",
            mean(chain_post$mu), sd(chain_post$mu),
            quantile(chain_post$mu, 0.025), quantile(chain_post$mu, 0.975)))
cat(sprintf("  phi:     mean=%.4f, sd=%.4f, 95%% CI=[%.4f, %.4f]\n",
            mean(chain_post$phi), sd(chain_post$phi),
            quantile(chain_post$phi, 0.025), quantile(chain_post$phi, 0.975)))
cat(sprintf("  sigma_h: mean=%.4f, sd=%.4f, 95%% CI=[%.4f, %.4f]\n",
            mean(chain_post$sigma_h), sd(chain_post$sigma_h),
            quantile(chain_post$sigma_h, 0.025), quantile(chain_post$sigma_h, 0.975)))

# True values
cat("\nTrue values: mu=-1.0, phi=0.97, sigma_h=0.15\n")

# Acceptance rate
n_unique <- sum(diff(chain_raw$mu) != 0)
acceptance_rate <- n_unique / (nrow(chain_raw) - 1)
cat(sprintf("Acceptance rate: %.3f\n", acceptance_rate))

# Effective sample size
mcmc_obj <- mcmc(as.matrix(chain_post[, c("mu", "phi", "sigma_h")]))
ess <- effectiveSize(mcmc_obj)
cat(sprintf("ESS: mu=%.1f, phi=%.1f, sigma_h=%.1f\n\n", ess["mu"], ess["phi"], ess["sigma_h"]))

# ------------------------------------------------------------------------------
# 5. Save results
# ------------------------------------------------------------------------------
output_path <- file.path(script_dir, "results_r_pmmh.csv")
write.csv(chain_post, output_path, row.names = FALSE)
cat("Posterior samples saved to:", output_path, "\n")

# Save summary statistics
summary_path <- file.path(script_dir, "summary_r_pmmh.txt")
sink(summary_path)
cat("PMMH Validation Summary (pomp::pmcmc)\n")
cat(strrep("=", 50), "\n\n")
cat("Configuration:\n")
cat("  Particles:", n_particles, "\n")
cat("  MCMC iterations:", n_mcmc, "\n")
cat("  Burn-in:", n_burnin, "\n")
cat("  Posterior samples:", nrow(chain_post), "\n\n")
cat("True parameters: mu=-1.0, phi=0.97, sigma_h=0.15\n\n")
cat("Posterior summary:\n")
print(summary(chain_post[, c("mu", "phi", "sigma_h")]))
cat(sprintf("\nAcceptance rate: %.3f\n", acceptance_rate))
cat(sprintf("ESS: mu=%.1f, phi=%.1f, sigma_h=%.1f\n", ess["mu"], ess["phi"], ess["sigma_h"]))
cat(sprintf("\nElapsed time: %.1f seconds\n", t_elapsed))
sink()
cat("Summary saved to:", summary_path, "\n")

cat("\nDone.\n")
