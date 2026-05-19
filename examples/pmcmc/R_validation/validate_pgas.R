#!/usr/bin/env Rscript
# ==============================================================================
# validate_pgas.R
# Particle MCMC via nimbleSMC for SV model with leverage
# Cross-validation of particlefilterbox PGAS on SV model with leverage
#
# Uses NIMBLE's MCMC framework with nimbleSMC extensions.
# The sampler alternates between updating latent states and parameters,
# mirroring the structure of Particle Gibbs / PGAS algorithms.
#
# Required R packages: nimble (>= 1.0), nimbleSMC (>= 0.11), coda
# Install: install.packages(c("nimble", "nimbleSMC", "coda"))
#
# Usage: Rscript validate_pgas.R
# ==============================================================================

suppressPackageStartupMessages({
  library(nimble)
  library(nimbleSMC)
  library(coda)
})

cat(strrep("=", 71), "\n")
cat("PGAS Validation via nimbleSMC\n")
cat("Model: Stochastic Volatility with Leverage\n")
cat("True parameters: mu=-1.0, phi=0.97, sigma_h=0.15, rho=-0.5\n")
cat(strrep("=", 71), "\n\n")

# ------------------------------------------------------------------------------
# 1. Load data
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

data_path <- file.path(script_dir, "..", "data", "simulated_sv_leverage.csv")
if (!file.exists(data_path)) {
  data_path <- file.path("examples", "pmcmc", "data", "simulated_sv_leverage.csv")
}

cat("Loading data from:", data_path, "\n")
sv_data <- read.csv(data_path)

# Use first 100 observations (same as Python PGAS)
n_obs <- min(100, nrow(sv_data))
sv_data <- sv_data[1:n_obs, ]
cat("Using", n_obs, "observations\n\n")

y <- sv_data$y_obs

# ------------------------------------------------------------------------------
# 2. Define SV model with leverage in NIMBLE
# ------------------------------------------------------------------------------
# Model:
#   h_t ~ N(mu + phi * (h_{t-1} - mu), sigma_h^2)
#   eta_t = (h_t - mu - phi*(h_{t-1} - mu)) / sigma_h  (implied innovation)
#   y_t | h_t, h_{t-1} ~ N(rho * exp(h_t/2) * eta_t,
#                           exp(h_t) * (1 - rho^2))
#
# h is stochastic latent state. Leverage enters through the observation
# equation via eta_t, the state innovation.

svLeverageCode <- nimbleCode({
  # Priors
  mu ~ dnorm(-1.0, sd = 2.0)
  phi ~ dunif(0.5, 0.999)
  sigma_h ~ dgamma(2.0, rate = 10.0)
  rho ~ dunif(-0.99, 0.99)

  # Initial state (stationary distribution)
  h[1] ~ dnorm(mu, sd = sigma_h / sqrt(1.0 - phi * phi))
  y[1] ~ dnorm(0, sd = exp(h[1] / 2.0))

  # State evolution and observation with leverage
  for (t in 2:T) {
    h[t] ~ dnorm(mu + phi * (h[t-1] - mu), sd = sigma_h)
    eta[t] <- (h[t] - mu - phi * (h[t-1] - mu)) / sigma_h
    y[t] ~ dnorm(rho * exp(h[t] / 2.0) * eta[t],
                 sd = exp(h[t] / 2.0) * sqrt(1.0 - rho * rho))
  }
})

# Constants and data
svConstants <- list(T = n_obs)
svData <- list(y = y)

# Initial values
set.seed(42)
svInits <- list(
  mu = -1.0,
  phi = 0.95,
  sigma_h = 0.2,
  rho = -0.3,
  h = rep(-1.0, n_obs)
)

# ------------------------------------------------------------------------------
# 3. Build NIMBLE model
# ------------------------------------------------------------------------------
cat("Building NIMBLE model...\n")
svModel <- nimbleModel(
  code = svLeverageCode,
  constants = svConstants,
  data = svData,
  inits = svInits,
  check = FALSE
)
cat("Model built successfully.\n\n")

# Compile model
cat("Compiling model...\n")
cSvModel <- compileNimble(svModel)
cat("Model compiled.\n\n")

# ------------------------------------------------------------------------------
# 4. Configure and run MCMC
# ------------------------------------------------------------------------------
n_mcmc <- 2000
n_burnin <- 500

cat("MCMC configuration:\n")
cat("  MCMC iterations:", n_mcmc, "\n")
cat("  Burn-in:", n_burnin, "\n\n")

# Use default MCMC configuration - NIMBLE will assign appropriate samplers
# for both parameters and latent states (h nodes)
svMCMCconf <- configureMCMC(svModel)

# Monitor parameters
svMCMCconf$addMonitors(c("mu", "phi", "sigma_h", "rho"))

cat("Assigned samplers:\n")
print(svMCMCconf)

# Build and compile MCMC
cat("\nBuilding MCMC...\n")
svMCMC <- buildMCMC(svMCMCconf)
cat("Compiling MCMC...\n")
cSvMCMC <- compileNimble(svMCMC, project = svModel)
cat("MCMC compiled.\n\n")

# Run MCMC
set.seed(42)

cat("Running MCMC... (this may take several minutes)\n")
t_start <- proc.time()

samples <- runMCMC(
  cSvMCMC,
  niter = n_mcmc + n_burnin,
  nburnin = n_burnin,
  nchains = 1,
  setSeed = 42
)

t_elapsed <- (proc.time() - t_start)[3]
cat(sprintf("MCMC completed in %.1f seconds\n\n", t_elapsed))

# ------------------------------------------------------------------------------
# 5. Extract and summarize posterior
# ------------------------------------------------------------------------------
param_names <- c("mu", "phi", "sigma_h", "rho")
chain_post <- as.data.frame(samples[, param_names])

cat("Posterior samples:", nrow(chain_post), "\n\n")

cat("Posterior summaries:\n")
for (p in param_names) {
  vals <- chain_post[[p]]
  cat(sprintf("  %s: mean=%.4f, sd=%.4f, 95%% CI=[%.4f, %.4f]\n",
              p, mean(vals), sd(vals),
              quantile(vals, 0.025), quantile(vals, 0.975)))
}

cat("\nTrue values: mu=-1.0, phi=0.97, sigma_h=0.15, rho=-0.5\n")

# Check leverage sign
if (mean(chain_post$rho) < 0) {
  cat("PASS: Posterior mean of rho is negative (leverage effect detected)\n")
} else {
  cat("WARNING: Posterior mean of rho is not negative\n")
}

# ESS
mcmc_obj <- mcmc(as.matrix(chain_post))
ess <- effectiveSize(mcmc_obj)
cat(sprintf("ESS: mu=%.1f, phi=%.1f, sigma_h=%.1f, rho=%.1f\n\n",
            ess["mu"], ess["phi"], ess["sigma_h"], ess["rho"]))

# Acceptance rates (approximate from chain)
for (p in param_names) {
  n_unique <- sum(diff(chain_post[[p]]) != 0)
  rate <- n_unique / (nrow(chain_post) - 1)
  cat(sprintf("  %s acceptance rate: %.3f\n", p, rate))
}

# ------------------------------------------------------------------------------
# 6. Save results
# ------------------------------------------------------------------------------
output_path <- file.path(script_dir, "results_r_pgas.csv")
write.csv(chain_post, output_path, row.names = FALSE)
cat("\nPosterior samples saved to:", output_path, "\n")

# Save summary
summary_path <- file.path(script_dir, "summary_r_pgas.txt")
sink(summary_path)
cat("PGAS Validation Summary (nimbleSMC)\n")
cat(strrep("=", 50), "\n\n")
cat("Configuration:\n")
cat("  MCMC iterations:", n_mcmc, "\n")
cat("  Burn-in:", n_burnin, "\n")
cat("  Posterior samples:", nrow(chain_post), "\n\n")
cat("True parameters: mu=-1.0, phi=0.97, sigma_h=0.15, rho=-0.5\n\n")
cat("Posterior summary:\n")
print(summary(chain_post))
cat("\nESS:\n")
print(ess)
cat(sprintf("\nElapsed time: %.1f seconds\n", t_elapsed))
sink()
cat("Summary saved to:", summary_path, "\n")

cat("\nDone.\n")
