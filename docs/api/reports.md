---
title: "Reports API"
description: "API reference for particlefilterbox.reports — HTML, LaTeX, and Markdown report generators for filters, PMCMC, comparisons, and experiments"
---

# Reports API Reference

!!! info "Module"
    **Import**: `from particlefilterbox.reports import FilterReport, PMCMCReport, ComparisonReport, ExperimentReport`
    **Source**: `particlefilterbox/reports/`

## Overview

The reports module turns result objects into publication-ready artifacts. Four report classes are provided, one per result type, each with a consistent set of `to_html`, `to_latex`, and `to_markdown` methods (as applicable). Reports are templated (Jinja2) and fully themeable.

| Report | Consumes | Outputs |
|--------|----------|---------|
| `FilterReport` | `ParticleFilterResults` | HTML, LaTeX, Markdown |
| `PMCMCReport` | `PMCMCResults` | HTML, summary table, diagnostics table |
| `ComparisonReport` | `ComparisonResult` | HTML, LaTeX, Markdown |
| `ExperimentReport` | `ExperimentResult` | HTML, LaTeX, Markdown |

```python
from particlefilterbox.reports import FilterReport

report = FilterReport(result)
report.to_html("output/filter.html")
```

---

## FilterReport

Report for a single particle-filter run.

### Constructor

```python
class FilterReport(
    result: ParticleFilterResults,
    title: str | None = None,
    template: str = "default",
    theme: str = "default",
    include_sections: list[str] | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `ParticleFilterResults` | — | Output of `filter.filter(...)` |
| `title` | `str \| None` | `None` | Report title (falls back to model name) |
| `template` | `str` | `"default"` | Template directory name or absolute path |
| `theme` | `str` | `"default"` | CSS theme for HTML output |
| `include_sections` | `list[str] \| None` | `None` | Subset of sections to render |

**Available sections:** `"summary"`, `"diagnostics"`, `"filtered_state"`, `"weights"`, `"ess"`, `"trajectories"`, `"predictive_check"`.

### Methods

#### `to_html()`

```python
def to_html(
    self,
    path: str | Path,
    embed_figures: bool = True,
    standalone: bool = True,
) -> Path
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str \| Path` | — | Output HTML path |
| `embed_figures` | `bool` | `True` | Inline figures as base64 PNG |
| `standalone` | `bool` | `True` | Produce a self-contained file (no external assets) |

#### `to_latex()`

```python
def to_latex(
    self,
    path: str | Path,
    standalone: bool = False,
) -> Path
```

`standalone=True` emits a compilable `\documentclass{article}` file; `False` emits a fragment suitable for `\input`.

#### `to_markdown()`

```python
def to_markdown(
    self,
    path: str | Path | None = None,
) -> str
```

**Returns**: the Markdown string (always) and writes it to `path` when provided.

### Example

```python
from particlefilterbox.reports import FilterReport

report = FilterReport(result, title="SV-AR1 — bootstrap filter", theme="academic")
report.to_html("reports/sv_bootstrap.html")
report.to_latex("reports/sv_bootstrap.tex", standalone=False)
```

---

## PMCMCReport

Report for a PMCMC posterior sample.

### Constructor

```python
class PMCMCReport(
    chain: PMCMCResults,
    title: str | None = None,
    template: str = "default",
    theme: str = "default",
    burn_in: int | None = None,
    thin: int = 1,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chain` | `PMCMCResults` | — | PMMH / PG / PGAS / SMC² sample |
| `burn_in` | `int \| None` | `None` | Samples discarded before summaries |
| `thin` | `int` | `1` | Keep every $k$-th sample |

### Methods

#### `to_html()`

```python
def to_html(
    self,
    path: str | Path,
    include_traces: bool = True,
    include_acf: bool = True,
    include_pair_plot: bool = True,
) -> Path
```

#### `summary_table()`

Posterior mean, std, HDI, ESS, and $\hat{R}$ per parameter.

```python
def summary_table(
    self,
    hdi: float = 0.95,
    as_dataframe: bool = True,
) -> pd.DataFrame | str
```

#### `diagnostics_table()`

Convergence diagnostics: ESS, $\hat{R}$, Geweke, acceptance, integrated autocorrelation.

```python
def diagnostics_table(
    self,
    as_dataframe: bool = True,
) -> pd.DataFrame | str
```

### Example

```python
from particlefilterbox.reports import PMCMCReport

report = PMCMCReport(chain, burn_in=2000, thin=5)
print(report.summary_table())
print(report.diagnostics_table())
report.to_html("reports/pmmh.html")
```

---

## ComparisonReport

Report for a multi-filter comparison.

### Constructor

```python
class ComparisonReport(
    comparison: ComparisonResult,
    title: str | None = None,
    template: str = "default",
    theme: str = "default",
    baseline: str | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `comparison` | `ComparisonResult` | — | Output of `compare_filters(...)` |
| `baseline` | `str \| None` | `None` | Filter used as the reference row |

### Methods

#### `to_html()`

```python
def to_html(
    self,
    path: str | Path,
    metrics: list[str] | None = None,
) -> Path
```

#### `to_latex()`

```python
def to_latex(
    self,
    path: str | Path,
    booktabs: bool = True,
) -> Path
```

#### `to_markdown()`

```python
def to_markdown(
    self,
    path: str | Path | None = None,
) -> str
```

### Example

```python
from particlefilterbox import compare_filters
from particlefilterbox.reports import ComparisonReport

comparison = compare_filters(
    filters=[bootstrap, auxiliary, rbpf],
    observations=y,
    metrics=["log_likelihood", "rmse", "wall_time"],
    n_repeats=20,
)
ComparisonReport(comparison, baseline="bootstrap").to_html("cmp.html")
```

---

## ExperimentReport

Report for an `ExperimentResult` (combinations of model × filter × configuration).

### Constructor

```python
class ExperimentReport(
    experiment: ExperimentResult,
    title: str | None = None,
    template: str = "default",
    theme: str = "default",
)
```

### Methods

#### `to_html()`

```python
def to_html(
    self,
    path: str | Path,
    include_raw: bool = False,
) -> Path
```

#### `to_latex()`

```python
def to_latex(
    self,
    path: str | Path,
    booktabs: bool = True,
) -> Path
```

#### `to_markdown()`

```python
def to_markdown(
    self,
    path: str | Path | None = None,
) -> str
```

### Example

```python
from particlefilterbox.reports import ExperimentReport

report = ExperimentReport(experiment_result, title="SV filter benchmark")
report.to_html("reports/experiment.html", include_raw=True)
```

---

## Customization

All reports accept `template` and `theme` arguments. Customize either by pointing at your own directories.

### Templates

Templates are Jinja2 directories with a required `base.html.j2`, plus optional `base.tex.j2` and `base.md.j2`.

```text
my_templates/
├── base.html.j2
├── base.tex.j2
├── base.md.j2
└── partials/
    ├── summary.html.j2
    └── diagnostics.html.j2
```

Pass the directory path:

```python
FilterReport(result, template="/path/to/my_templates").to_html("out.html")
```

### Themes

Themes map to CSS files bundled under `particlefilterbox/reports/themes/`:

| Theme | Description |
|-------|-------------|
| `default` | Balanced light theme |
| `academic` | Serif fonts, muted palette, print-friendly |
| `dark` | Dark mode for screen reading |
| `minimal` | Black-and-white, PDF-targeted |

Register your own theme:

```python
from particlefilterbox.reports import register_theme

register_theme("corporate", css_path="/path/to/corporate.css")
FilterReport(result, theme="corporate").to_html("out.html")
```

### Logos

Embed a logo in the report header:

```python
FilterReport(
    result,
    title="Q2 2026 stability audit",
).to_html("audit.html", embed_figures=True)
```

Pass `logo=` via the template context variables (`extra_context={"logo": "path.png"}`) when calling `to_html`.

---

## See Also

- [Visualization API](visualization.md) — figures embedded by every report
- [Experiment API](experiment.md) — producer of `ExperimentResult`
- [PMCMC API](pmcmc.md) — producer of `PMCMCResults`
