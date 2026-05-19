---
title: "Experiment API"
description: "API reference for particlefilterbox.experiment — reproducible experiment configuration, runners, and results for multi-filter comparisons"
---

# Experiment API Reference

!!! info "Module"
    **Import**: `from particlefilterbox.experiment import ExperimentConfig, ExperimentRunner, ExperimentResult`
    **Source**: `particlefilterbox/experiment/`

## Overview

The experiment module formalizes reproducible comparisons between filters, models, and configurations. An experiment is defined declaratively via `ExperimentConfig`, executed by `ExperimentRunner`, and produces an `ExperimentResult` that can be serialized, tabulated, and fed into reports.

| Class | Role |
|-------|------|
| `ExperimentConfig` | Declarative specification of the experiment |
| `ExperimentRunner` | Executes the experiment (parallel-capable) |
| `ExperimentResult` | Result container: summaries, raw data, timings |

```python
from particlefilterbox.experiment import ExperimentConfig, ExperimentRunner
from particlefilterbox.filters import BootstrapPF, AuxiliaryPF

config = ExperimentConfig(
    model=sv_model,
    filters=[BootstrapPF, AuxiliaryPF],
    metrics=["log_likelihood", "rmse", "wall_time"],
    n_repeats=50,
    T=500,
)
result = ExperimentRunner(config, n_jobs=8).run()
result.to_dataframe().to_csv("experiment.csv")
```

---

## ExperimentConfig

Declarative specification of the experiment.

### Constructor

```python
@dataclass
class ExperimentConfig:
    model: ParticleFilterModel | list[ParticleFilterModel]
    filters: list[type[BaseParticleFilter]] | list[BaseParticleFilter]
    metrics: list[str] = field(default_factory=lambda: ["log_likelihood", "rmse"])
    n_repeats: int = 30
    T: int = 500
    n_particles: int | list[int] = 1000
    resampling: str = "systematic"
    seed: int | None = 42
    true_state: NDArray[np.float64] | None = None
    observations: NDArray[np.float64] | None = None
    extra_filter_kwargs: dict[str, Any] | None = None
    save_trajectories: bool = False
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ParticleFilterModel \| list[...]` | — | One or more state-space models |
| `filters` | `list[type[BaseParticleFilter]]` | — | Filter classes or pre-built instances |
| `metrics` | `list[str]` | `["log_likelihood", "rmse"]` | Metrics to record per run |
| `n_repeats` | `int` | `30` | Monte Carlo replications |
| `T` | `int` | `500` | Horizon when simulating observations |
| `n_particles` | `int \| list[int]` | `1000` | Particle counts to sweep |
| `resampling` | `str` | `"systematic"` | Default resampling scheme |
| `seed` | `int \| None` | `42` | Root seed for reproducibility |
| `true_state` | `NDArray \| None` | `None` | Ground-truth states (for RMSE) |
| `observations` | `NDArray \| None` | `None` | Fixed observations. `None` simulates per-repeat |
| `extra_filter_kwargs` | `dict \| None` | `None` | Per-filter keyword arguments |
| `save_trajectories` | `bool` | `False` | Persist full particle trajectories (heavy) |

**Supported metrics:**

| Metric | Description |
|--------|-------------|
| `log_likelihood` | $\log \hat{p}(y_{1:T})$ |
| `rmse` | Root-mean-square error against `true_state` |
| `mae` | Mean absolute error against `true_state` |
| `ess_mean` | Mean ESS over the horizon |
| `ess_min` | Minimum ESS observed |
| `wall_time` | Seconds to run the filter |
| `memory` | Peak RSS (MB) |
| `max_weight` | Worst-case maximum normalized weight |

---

## ExperimentRunner

Executes an `ExperimentConfig`.

### Constructor

```python
class ExperimentRunner(
    config: ExperimentConfig,
    n_jobs: int = 1,
    backend: str = "loky",
    verbose: int = 0,
    progress: bool = True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `ExperimentConfig` | — | Experiment specification |
| `n_jobs` | `int` | `1` | Workers (`-1` = all cores) |
| `backend` | `str` | `"loky"` | Joblib backend |
| `verbose` | `int` | `0` | Progress verbosity |
| `progress` | `bool` | `True` | Show `tqdm` progress bar |

### Methods

#### `run()`

Execute all (model × filter × particle-count × repeat) combinations.

```python
def run(self) -> ExperimentResult
```

**Returns**: `ExperimentResult` with aggregated and raw results.

**Raises**:

- `ValueError` if `config.metrics` contains an unknown metric.
- `RuntimeError` if all replications of a filter fail.

### Example

```python
runner = ExperimentRunner(config, n_jobs=-1, progress=True)
result = runner.run()
```

---

## ExperimentResult

Container holding aggregated summaries and raw per-replication results.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `summary` | `pd.DataFrame` | Mean, std, min, max per (filter × metric) |
| `raw_results` | `pd.DataFrame` | One row per replication (long form) |
| `timings` | `pd.DataFrame` | Wall-time by filter and particle count |
| `config` | `ExperimentConfig` | The config that produced the result |
| `failures` | `pd.DataFrame` | Failed replications (if any) |

### Methods

#### `to_dataframe()`

```python
def to_dataframe(
    self,
    form: str = "summary",
) -> pd.DataFrame
```

`form` ∈ {`"summary"`, `"raw"`, `"timings"`}.

#### `to_latex()`

```python
def to_latex(
    self,
    path: str | Path | None = None,
    metric: str = "log_likelihood",
    booktabs: bool = True,
    float_format: str = "%.3f",
) -> str
```

Returns the LaTeX string; writes to `path` when given.

#### `statistical_tests()`

Pairwise tests between filters for a chosen metric.

```python
def statistical_tests(
    self,
    metric: str = "log_likelihood",
    test: str = "welch",
    alpha: float = 0.05,
    correction: str = "holm",
) -> pd.DataFrame
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `str` | `"log_likelihood"` | Metric under test |
| `test` | `str` | `"welch"` | `"welch"`, `"t"`, `"wilcoxon"`, `"friedman"` |
| `alpha` | `float` | `0.05` | Significance level |
| `correction` | `str` | `"holm"` | `"holm"`, `"bonferroni"`, `"fdr_bh"`, `"none"` |

**Returns**: A DataFrame with columns `filter_a`, `filter_b`, `statistic`, `p_value`, `p_adjusted`, `reject`.

#### `save()` / `load()`

```python
def save(self, path: str | Path) -> Path

@classmethod
def load(cls, path: str | Path) -> ExperimentResult
```

Pickle-based round-trip. Use `.json()` for a portable (lossy) dump.

### Example

```python
result = runner.run()

print(result.summary)
result.to_latex("tables/sv_benchmark.tex", metric="log_likelihood")
tests = result.statistical_tests(metric="rmse", test="wilcoxon")
print(tests.query("reject"))
```

---

## End-to-End Example

```python
import numpy as np
from particlefilterbox.experiment import ExperimentConfig, ExperimentRunner
from particlefilterbox.filters import BootstrapPF, AuxiliaryPF, RaoBlackwellizedPF
from particlefilterbox.models import StochasticVolatility
from particlefilterbox.reports import ExperimentReport

model = StochasticVolatility(variant="basic", mu=-0.5, phi=0.95, sigma=0.3)

config = ExperimentConfig(
    model=model,
    filters=[BootstrapPF, AuxiliaryPF, RaoBlackwellizedPF],
    metrics=["log_likelihood", "rmse", "ess_mean", "wall_time"],
    n_repeats=100,
    T=1000,
    n_particles=[500, 1000, 5000],
    seed=2026,
)

runner = ExperimentRunner(config, n_jobs=-1, progress=True)
result = runner.run()

ExperimentReport(result, title="SV filter benchmark").to_html("sv_bench.html")
result.save("sv_bench.pkl")
```

---

## See Also

- [Acceleration API](acceleration.md) — `ParallelRunner` used internally
- [Reports API](reports.md) — `ExperimentReport` for publication output
- [Diagnostics API](diagnostics.md) — metrics computed per run
