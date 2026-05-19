---
title: "CLI API"
description: "API reference for the particlefilterbox command-line interface — run filters, estimate parameters, compare methods, benchmark, and generate reports"
---

# CLI API Reference

!!! info "Entry Point"
    **Command**: `particlefilterbox` (short alias: `pfbox`)
    **Source**: `particlefilterbox/cli/`

## Overview

`particlefilterbox` provides a command-line interface for running particle filters, estimating parameters, comparing methods, benchmarking, and generating reports without writing Python code.

```bash
particlefilterbox <command> [options]
```

| Command | Description |
|---------|-------------|
| `particlefilterbox filter` | Run a particle filter on a data file |
| `particlefilterbox estimate` | Estimate parameters via PMCMC or SMC² |
| `particlefilterbox compare` | Compare several filters on the same data |
| `particlefilterbox report` | Render a saved result to HTML/LaTeX/Markdown |
| `particlefilterbox benchmark` | Sweep particle counts and measure wall-time |

Run `particlefilterbox <command> --help` for full help on any command.

---

## `particlefilterbox filter`

Run a particle filter over an observation sequence.

```bash
particlefilterbox filter \
  --model MODEL \
  --data FILE \
  --filter FILTER \
  --n-particles N \
  [options]
```

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--model NAME` | Built-in model (`sv`, `dsge`, `jump`, `regime`) | `--model sv` |
| `--data FILE` | CSV/Parquet observations | `--data returns.csv` |
| `--filter NAME` | Filter implementation | `--filter bootstrap` |
| `--n-particles N` | Particle count $N$ | `--n-particles 5000` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--variant NAME` | `basic` | Model variant (e.g. `leverage`, `t`, `jump` for SV) |
| `--params FILE` | — | YAML/JSON file with model parameters |
| `--resampling NAME` | `systematic` | `systematic`, `multinomial`, `stratified`, `residual` |
| `--ess-threshold FRAC` | `0.5` | Resample when ESS/N falls below this |
| `--seed INT` | `42` | RNG seed |
| `--output FILE`, `-o` | — | Save `ParticleFilterResults` (pickle) |
| `--format FORMAT` | `pickle` | `pickle`, `npz`, `json` |
| `--smooth` | off | Also run FFBSm smoother |
| `--backend NAME` | `numpy` | `numpy`, `numba`, `cupy`, `jax` |
| `--verbose`, `-v` | off | Verbose logging |

### Filter Names

| Value | Class |
|-------|-------|
| `bootstrap` | `BootstrapPF` |
| `sir` | `SIR` |
| `auxiliary`, `apf` | `AuxiliaryPF` |
| `rbpf`, `rao-blackwell` | `RaoBlackwellizedPF` |
| `upf`, `unscented` | `UnscentedPF` |
| `regularized` | `RegularizedPF` |
| `ensemble`, `enkf-pf` | `EnsemblePF` |
| `guided` | `GuidedPF` |
| `locally-optimal` | `LocallyOptimalPF` |

### Examples

```bash
# Bootstrap filter on S&P 500 returns with SV model
particlefilterbox filter \
  --model sv \
  --variant basic \
  --data sp500.csv \
  --filter bootstrap \
  --n-particles 5000 \
  --output sv_results.pkl

# Auxiliary PF with GPU acceleration and smoothing
particlefilterbox filter \
  --model sv \
  --data returns.csv \
  --filter auxiliary \
  --n-particles 20000 \
  --backend cupy \
  --smooth \
  -o auxiliary_smoothed.pkl
```

---

## `particlefilterbox estimate`

Estimate model parameters via PMCMC (PMMH / Particle Gibbs / PGAS) or SMC².

```bash
particlefilterbox estimate \
  --model MODEL \
  --data FILE \
  --method METHOD \
  --n-iterations N \
  [options]
```

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--model NAME` | Model identifier | `--model sv` |
| `--data FILE` | Observation file | `--data sp500.csv` |
| `--method NAME` | `pmmh`, `pgibbs`, `pgas`, `smc2` | `--method pmmh` |
| `--n-iterations N` | MCMC iterations (or SMC steps) | `--n-iterations 10000` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-particles N` | `500` | Inner-loop particles |
| `--priors FILE` | — | YAML/JSON prior specification |
| `--proposal NAME` | `adaptive` | `rw`, `adaptive`, `hmc`, `mala` |
| `--burn-in N` | `n_iter / 5` | Burn-in length |
| `--thin K` | `1` | Thinning factor |
| `--chains N` | `1` | Parallel chains |
| `--seed INT` | `42` | RNG seed |
| `--output FILE`, `-o` | — | Save `PMCMCResults` (pickle) |
| `--backend NAME` | `numpy` | Acceleration backend |
| `--verbose`, `-v` | off | Verbose logging |

### Examples

```bash
# PMMH on SV model
particlefilterbox estimate \
  --model sv \
  --data sp500.csv \
  --method pmmh \
  --n-iterations 10000 \
  --n-particles 500 \
  --priors sv_priors.yaml \
  -o pmmh_chain.pkl

# SMC² with 4 chains
particlefilterbox estimate \
  --model sv \
  --data sp500.csv \
  --method smc2 \
  --n-iterations 5000 \
  --chains 4 \
  --backend numba \
  -o smc2.pkl
```

**Prior file example (`sv_priors.yaml`):**

```yaml
mu:    { dist: normal,      mean: 0.0,   std: 1.0 }
phi:   { dist: uniform,     low: -1.0,   high: 1.0 }
sigma: { dist: inv_gamma,   shape: 2.5,  scale: 0.025 }
```

---

## `particlefilterbox compare`

Compare several filters on the same data.

```bash
particlefilterbox compare \
  --model MODEL \
  --data FILE \
  --filters LIST \
  [options]
```

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--model NAME` | Model | `--model sv` |
| `--data FILE` | Observations | `--data returns.csv` |
| `--filters LIST` | Comma-separated filter names | `--filters bootstrap,auxiliary,rbpf` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-particles N` | `1000` | Particle count per filter |
| `--n-repeats N` | `30` | Monte Carlo replications |
| `--metrics LIST` | `log_likelihood,rmse,wall_time` | Metrics to record |
| `--n-jobs N` | `1` | Parallel workers |
| `--seed INT` | `42` | RNG seed |
| `--output FILE`, `-o` | — | Save `ComparisonResult` (pickle) |
| `--report FILE` | — | Also emit HTML comparison report |

### Examples

```bash
# Compare three filters, 30 replications each
particlefilterbox compare \
  --model sv \
  --data sp500.csv \
  --filters bootstrap,auxiliary,rbpf \
  --n-particles 2000 \
  --n-repeats 30 \
  --metrics log_likelihood,rmse,ess_mean,wall_time \
  --n-jobs 8 \
  -o comparison.pkl \
  --report comparison.html
```

---

## `particlefilterbox report`

Render a saved result object to HTML/LaTeX/Markdown.

```bash
particlefilterbox report \
  --result FILE \
  --output FILE \
  [options]
```

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--result FILE` | Saved result (`.pkl`) — filter, PMCMC, comparison, or experiment | `--result pmmh.pkl` |
| `--output FILE`, `-o` | Output path. Format inferred from extension | `-o report.html` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--format FORMAT` | *from extension* | `html`, `latex`, `md` |
| `--template NAME` | `default` | Template directory or bundled name |
| `--theme NAME` | `default` | `default`, `academic`, `dark`, `minimal` |
| `--title STR` | — | Override report title |
| `--sections LIST` | *all* | Comma-separated section names |
| `--standalone` | on | Emit self-contained file (HTML/LaTeX) |
| `--verbose`, `-v` | off | Verbose logging |

### Examples

```bash
# HTML filter report
particlefilterbox report --result sv_results.pkl -o sv.html

# LaTeX fragment for inclusion in a paper
particlefilterbox report \
  --result pmmh.pkl \
  -o tables/posterior.tex \
  --format latex \
  --theme academic \
  --no-standalone

# Markdown summary of a comparison
particlefilterbox report --result comparison.pkl -o comparison.md
```

---

## `particlefilterbox benchmark`

Sweep particle counts and measure wall-time and key metrics.

```bash
particlefilterbox benchmark \
  --model MODEL \
  --n-particles LIST \
  [options]
```

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--model NAME` | Model | `--model sv` |
| `--n-particles LIST` | Comma-separated counts to sweep | `--n-particles 100,500,1000,5000` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data FILE` | *simulated* | Use provided data; otherwise simulate |
| `--T N` | `500` | Simulated horizon (if no `--data`) |
| `--filters LIST` | `bootstrap` | Filters to benchmark |
| `--n-repeats N` | `10` | Replications per cell |
| `--backends LIST` | `numpy` | Backends to test: `numpy,numba,cupy,jax` |
| `--metrics LIST` | `wall_time,log_likelihood,ess_mean` | Metrics to record |
| `--n-jobs N` | `1` | Parallel workers |
| `--seed INT` | `42` | RNG seed |
| `--output FILE`, `-o` | — | Save `ExperimentResult` (pickle) |
| `--report FILE` | — | Also emit HTML benchmark report |

### Examples

```bash
# Particle-count sweep for bootstrap
particlefilterbox benchmark \
  --model sv \
  --n-particles 100,500,1000,5000,10000 \
  --n-repeats 20 \
  -o bench.pkl \
  --report bench.html

# Compare CPU vs. Numba vs. CuPy
particlefilterbox benchmark \
  --model sv \
  --n-particles 1000,10000,100000 \
  --backends numpy,numba,cupy \
  --filters bootstrap,auxiliary \
  --n-repeats 5 \
  -o backend_bench.pkl
```

---

## Integration with Python API

The CLI is designed for quick analyses and batch jobs. For full control, use the Python API:

=== "CLI"

    ```bash
    particlefilterbox filter \
      --model sv \
      --data sp500.csv \
      --filter bootstrap \
      --n-particles 5000 \
      -o result.pkl
    ```

=== "Python"

    ```python
    import pandas as pd
    import particlefilterbox as pfb

    y = pd.read_csv("sp500.csv")["return"].values
    model = pfb.models.StochasticVolatility(variant="basic")
    config = pfb.PFConfig(n_particles=5000, resampling="systematic")

    pf = pfb.BootstrapPF(model, config)
    result = pf.filter(y)
    result.save("result.pkl")
    ```

The Python API provides access to:

- All filter, smoother, SMC, and PMCMC implementations
- Custom model definition
- Fine-grained configuration (proposals, jitter, adaptive $N$)
- Full report customization and composition with visualizations

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Invalid arguments |
| `2` | Data file not found or unreadable |
| `3` | Model/filter name not recognized |
| `4` | Filter divergence (all weights zero) |
| `5` | Backend unavailable (Numba/CuPy/JAX missing) |
| `99` | Unexpected error — rerun with `--verbose` |

---

## See Also

- [Getting Started](../getting-started/index.md) — installation and first steps
- [Filters API](filters.md) — full Python filter API
- [PMCMC API](pmcmc.md) — PMMH / PGibbs / PGAS / SMC² in Python
- [Reports API](reports.md) — template and theme customization
