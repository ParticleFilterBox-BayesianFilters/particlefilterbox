---
title: "Contributing Guide"
description: "How to contribute to particlefilterbox — development setup, coding standards, testing, documentation, and pull request process."
---

# Contributing to particlefilterbox

Thank you for your interest in contributing to **particlefilterbox**! Whether you are reporting a bug, proposing a new filter, improving documentation, or submitting code, your help is welcome. This library aims to be the most comprehensive Python toolkit for particle filtering, Sequential Monte Carlo (SMC), and Particle MCMC (PMCMC) methods — and community contributions are central to reaching that goal.

---

## Types of Contributions

| Type | Where | Description |
|------|-------|-------------|
| Bug reports | [GitHub Issues](https://github.com/nodesecon/particlefilterbox/issues) | Reproducible problem with expected vs. actual behavior |
| Feature requests | [GitHub Issues](https://github.com/nodesecon/particlefilterbox/issues) | Proposals with `[Feature]` label |
| Code (PR) | [Pull Requests](https://github.com/nodesecon/particlefilterbox/pulls) | New filters, smoothers, SMC methods, tests, bug fixes |
| Documentation | `docs/` directory | Tutorials, API docs, theory pages, examples |
| Benchmarks | `benchmarks/` directory | Performance comparisons, regression tests |
| Test additions | `tests/` directory | Unit, integration, and validation tests |

!!! tip "Good first contributions"
    - Fixing typos or improving clarity in the documentation
    - Adding an example to an existing filter's docstring
    - Porting a benchmark model from `particles`, `pyfilter`, or `bayesloop`
    - Improving error messages in boundary cases

---

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/nodesecon/particlefilterbox.git
cd particlefilterbox
```

### 2. Create a Virtual Environment

=== "venv"

    ```bash
    python -m venv .venv
    source .venv/bin/activate   # Linux / macOS
    # .venv\Scripts\activate    # Windows
    ```

=== "conda"

    ```bash
    conda create -n pfbox python=3.11
    conda activate pfbox
    ```

=== "uv"

    ```bash
    uv venv
    source .venv/bin/activate
    ```

### 3. Install in Development Mode

```bash
pip install -e ".[dev,all]"
```

This installs `particlefilterbox` in editable mode together with every optional extra (`viz`, `cli`, `docs`, `accel`) and the full developer toolchain.

### 4. Verify the Setup

```bash
pytest tests/ -v --timeout=60
```

A full green run confirms your environment is ready for development.

### Development Dependencies

| Tool | Minimum Version | Purpose |
|------|-----------------|---------|
| `pytest` | 7.0 | Testing framework |
| `pytest-cov` | 4.0 | Coverage measurement |
| `ruff` | 0.4 | Linting and formatting |
| `pyright` | 1.1 | Static type checking |
| `numba` | 0.58 | JIT acceleration (optional) |
| `cupy` | 12.0 | GPU backend (optional) |
| `mkdocs-material` | 9.0 | Documentation rendering |

!!! info "kalmanbox dependency"
    Several estimators (the Rao-Blackwellized and Unscented particle filters, Kalman-based validation diagnostics, DSGE state-space utilities) depend on [`kalmanbox`](https://github.com/nodesecon/kalmanbox). When developing these modules, install `kalmanbox` in editable mode as well so your changes to the Kalman primitives propagate.

---

## Code Standards

### Style

- **Formatter**: `ruff format` (line length 100)
- **Linter**: `ruff check`
- **Type hints**: Required for all public API functions and methods
- **Docstrings**: NumPy-style for all public classes, methods, and functions
- **Python version**: 3.11+ compatibility

```bash
# Format and lint
ruff format particlefilterbox/
ruff check particlefilterbox/ --fix

# Type check
pyright particlefilterbox/
```

### Public vs. Private API

- Anything exported from `particlefilterbox/__init__.py` or a submodule's `__init__.py` is considered **public** and must be documented.
- Private helpers start with a single underscore and live inside the module where they are used.
- Do not export symbols that are not yet ready for users.

### Branch Naming

Use descriptive branch names that communicate scope:

- `feature/add-guided-pf` — New feature
- `fix/ess-underflow` — Bug fix
- `docs/improve-rbpf-theory` — Documentation change
- `test/add-pgas-validation` — Test additions
- `perf/numba-systematic-resample` — Performance improvement

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `style`, `chore`.

**Example**:

```text
feat(filters): Add Pitt-Shephard auxiliary particle filter

Implements the fully adapted APF of Pitt & Shephard (1999) with
pre-specified lookahead weights. Validates against the Kalman
solution on the linear Gaussian state-space benchmark to 1e-6.

Closes #42
```

---

## Adding a New Filter

Particle filters live in `particlefilterbox/filters/`. Every filter must:

1. Inherit from `particlefilterbox.filters.base.ParticleFilter`
2. Implement `step(y_t)` and `filter(y)` (the latter provided by the base class when `step` is implemented)
3. Produce a `FilterResult` container
4. Have unit tests in `tests/filters/`
5. Be exported from `particlefilterbox/filters/__init__.py`
6. Have a user-guide page in `docs/user-guide/filters/` and an entry in `docs/api/filters.md`

### Filter Template

```python
"""My particle filter.

Implements the Author (Year) particle filter for nonlinear
state-space models.
"""

from __future__ import annotations

import numpy as np

from particlefilterbox.core import ParticleCloud
from particlefilterbox.filters.base import ParticleFilter, FilterResult
from particlefilterbox.resampling import systematic


class MyParticleFilter(ParticleFilter):
    r"""Short description of the filter.

    Longer description explaining the algorithm, the proposal
    structure, its assumptions, and when to use it.

    Parameters
    ----------
    model : StateSpaceModel
        Model with ``transition``, ``observation``, and
        ``initial_state`` methods.
    n_particles : int, default=1000
        Number of particles :math:`N`.
    ess_threshold : float, default=0.5
        ESS threshold (as a fraction of :math:`N`) triggering
        resampling.

    Notes
    -----
    The importance weights are updated according to

    .. math::
        w_t^{(i)} \propto w_{t-1}^{(i)} \,
        \frac{p(y_t \mid x_t^{(i)}) \, p(x_t^{(i)} \mid x_{t-1}^{(i)})}
             {q(x_t^{(i)} \mid x_{t-1}^{(i)}, y_t)}

    References
    ----------
    .. [1] Author, A. (Year). Title. *Journal*, vol(issue), pages.
    """

    def __init__(
        self,
        model,
        n_particles: int = 1000,
        ess_threshold: float = 0.5,
    ) -> None:
        super().__init__(model=model, n_particles=n_particles)
        self.ess_threshold = ess_threshold

    def step(self, y_t: np.ndarray) -> None:
        """Advance the filter by one observation.

        Parameters
        ----------
        y_t : array_like
            Observation at time ``t``.
        """
        # 1. Propose particles from q(x_t | x_{t-1}, y_t)
        # 2. Compute incremental log-weights
        # 3. Normalize and update ESS
        # 4. Resample if ESS < threshold
        raise NotImplementedError
```

### Where to Place the Code

| Module family | Directory | Base class |
|---|---|---|
| Particle filters | `particlefilterbox/filters/` | `ParticleFilter` |
| Smoothers | `particlefilterbox/smoothers/` | `ParticleSmoother` |
| SMC samplers | `particlefilterbox/smc/` | `SMCSampler` |
| PMCMC | `particlefilterbox/pmcmc/` | `PMCMCSampler` |
| State-space models | `particlefilterbox/models/` | `StateSpaceModel` |
| Resampling schemes | `particlefilterbox/resampling/` | — (functions) |
| Diagnostics | `particlefilterbox/diagnostics/` | `Diagnostic` |

---

## Adding a New Diagnostic

Diagnostics return a `DiagnosticResult` with a consistent interface:

```python
"""My diagnostic for particle filter output."""

from __future__ import annotations

from particlefilterbox.diagnostics.base import Diagnostic, DiagnosticResult


class MyDiagnostic(Diagnostic):
    """Detect degeneracy via some novel metric.

    Parameters
    ----------
    result : FilterResult
        Output of a particle filter run.

    References
    ----------
    .. [1] Author, A. (Year). Title. *Journal*, vol(issue), pages.
    """

    def run(self, alpha: float = 0.05) -> DiagnosticResult:
        """Compute the diagnostic.

        Returns
        -------
        DiagnosticResult
            Structured result with ``name``, ``statistic``,
            ``threshold``, ``passed``, and ``message``.
        """
        ...
```

---

## Running Tests

```bash
# Full suite
pytest tests/ -v

# Module-level
pytest tests/filters/ -v
pytest tests/smc/ -v

# Single test
pytest tests/filters/test_bootstrap.py::test_linear_gaussian_converges -v

# With coverage
pytest tests/ --cov=particlefilterbox --cov-report=html --cov-branch

# In parallel (install pytest-xdist)
pytest tests/ -n auto
```

### Writing Tests

Tests live under `tests/`, mirroring the package layout. Use the shared fixtures in `tests/conftest.py` whenever possible.

```python
import numpy as np
import pytest

from particlefilterbox.filters import BootstrapParticleFilter
from particlefilterbox.datasets import load_linear_gaussian


class TestBootstrapFilter:
    """Tests for the bootstrap particle filter."""

    @pytest.fixture
    def data(self):
        return load_linear_gaussian(T=200, seed=0)

    def test_filter_runs(self, data):
        model, y = data["model"], data["y"]
        pf = BootstrapParticleFilter(model, n_particles=500, seed=0)
        result = pf.filter(y)
        assert result.filtered_mean.shape[0] == len(y)

    def test_matches_kalman(self, data):
        """Bootstrap PF must converge to the Kalman filter."""
        model, y, kf_mean = data["model"], data["y"], data["kalman_mean"]
        pf = BootstrapParticleFilter(model, n_particles=50_000, seed=0)
        result = pf.filter(y)
        np.testing.assert_allclose(
            result.filtered_mean, kf_mean, rtol=5e-2,
            err_msg="Particle filter diverges from Kalman benchmark",
        )
```

### Validation Against Kalman Benchmarks

For any linear-Gaussian special case, the particle filter's posterior mean must converge to the Kalman filter solution. We use `kalmanbox` as the authoritative reference:

```python
from kalmanbox import KalmanFilter

def test_lg_matches_kalman_within_mc_error():
    kf = KalmanFilter(F, Q, H, R).fit(y)
    pf = BootstrapParticleFilter(model, n_particles=100_000, seed=0)
    res = pf.filter(y)
    np.testing.assert_allclose(res.filtered_mean, kf.filtered_mean, atol=1e-2)
```

---

## Contributing to Documentation

Documentation is written in Markdown with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/). Source lives under `docs/`.

```bash
# Local preview with live reload
mkdocs serve

# Build static site
mkdocs build --strict
```

### Documentation Conventions

- **API pages** (`docs/api/`) use `mkdocstrings` to auto-generate reference from docstrings — edit the source docstring, not the Markdown.
- **Theory pages** use MathJax via `pymdownx.arithmatex`. Inline: `$x_t$`; display: `$$...$$`.
- **Code blocks** must always declare a language (`python`, `bash`, `text`, etc.).
- **Admonitions** use the Material syntax: `!!! note`, `!!! warning`, `!!! tip`.
- **Tabs** use `=== "Label"` blocks.
- **Internal links** use relative paths (`[text](../api/filters.md)`), not absolute URLs.

### Adding a Tutorial

1. Place the Markdown file in `docs/tutorials/`.
2. Include runnable code that imports from `particlefilterbox`.
3. Add a navigation entry in `mkdocs.yml`.
4. If the tutorial introduces a new concept, cross-link to the corresponding theory page.

---

## Pull Request Process

### Step-by-Step

1. **Create a feature branch**:

    ```bash
    git checkout -b feature/my-new-feature
    ```

2. **Make changes**: code, tests, and documentation.

3. **Run the local quality gate**:

    ```bash
    ruff format particlefilterbox/
    ruff check particlefilterbox/ --fix
    pyright particlefilterbox/
    pytest tests/ -v
    mkdocs build --strict
    ```

4. **Commit** with a Conventional Commit message.

5. **Push and open a PR**:

    ```bash
    git push origin feature/my-new-feature
    ```

6. **Fill out the PR template** and respond to review comments.

### PR Checklist

- [ ] Tests pass locally (`pytest tests/ -v`)
- [ ] Lint and type checks pass (`ruff check`, `pyright`)
- [ ] Documentation builds without warnings (`mkdocs build --strict`)
- [ ] New public API has NumPy-style docstrings
- [ ] New estimators validated against a reference (Kalman closed form, other library, or Monte Carlo ground truth)
- [ ] Exports updated in the relevant `__init__.py`
- [ ] Changelog entry added under `[Unreleased]`

### Review Guidelines

- Keep PRs focused — one feature or fix per PR.
- Expect at least one maintainer review before merge.
- Squash-and-merge is preferred to keep `main` history linear.

---

## Release Process

Releases follow [Semantic Versioning](https://semver.org/) and the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

1. Bump the version in `particlefilterbox/__version__.py`.
2. Move the `[Unreleased]` block in `docs/contributing/changelog.md` to a dated version heading.
3. Tag the commit: `git tag -a v0.2.0 -m "Release 0.2.0"`.
4. Push the tag: `git push origin v0.2.0`.
5. The PyPI publish workflow (`.github/workflows/publish.yml`) builds and uploads the wheel.
6. Publish a GitHub Release copying the changelog section.

---

## Areas That Need Contribution

The roadmap lists our near-term priorities. Particularly high-value contributions right now:

- **Proposal distributions**: locally optimal and guided proposals for specific model families.
- **GPU kernels**: CuPy/JAX implementations of systematic and residual resampling.
- **Benchmarks**: validated comparisons against `particles`, `pyfilter`, `bayesloop`.
- **Pre-built models**: regime-switching GARCH, jump-diffusion with stochastic jump intensity.
- **Tutorials**: reproducing a published paper end-to-end.

See the [Roadmap](roadmap.md) for the full list.

---

## Reporting Issues

File issues on [GitHub](https://github.com/nodesecon/particlefilterbox/issues) with:

1. A clear title describing the problem.
2. A **minimal reproducible example** (MRE).
3. Expected vs. actual behavior.
4. `particlefilterbox` version: `pip show particlefilterbox`.
5. Python version and OS.
6. Full traceback, if applicable.

!!! warning "Security issues"
    Do **not** open a public issue for security vulnerabilities. Email the maintainers directly (see the Code of Conduct for contact information).

---

## Recognition

Contributors are recognized in:

- The [Changelog](changelog.md) for each release.
- Release notes on GitHub.
- The `AUTHORS` file at the repository root.

Significant methodological contributions may result in co-authorship on associated papers.

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## Questions?

- **General questions**: [GitHub Discussions](https://github.com/nodesecon/particlefilterbox/discussions)
- **Bug reports**: [GitHub Issues](https://github.com/nodesecon/particlefilterbox/issues)
- **Feature requests**: [GitHub Issues](https://github.com/nodesecon/particlefilterbox/issues) with `[Feature]` label

---

## See Also

- [Code of Conduct](code-of-conduct.md) — Community standards
- [Changelog](changelog.md) — Version history
- [Roadmap](roadmap.md) — Planned features
- [API Reference](../api/index.md) — Full API documentation
