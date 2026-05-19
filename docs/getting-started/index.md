---
title: Getting Started
description: Install particlefilterbox and run your first particle filter in minutes
---

# Getting Started

Welcome to particlefilterbox! This section will get you from zero to running particle filters, SMC samplers, and PMCMC algorithms in minutes.

## Learning Path

Follow these steps for the fastest route to productive use:

1. **[Installation](installation.md)** -- Install the library and verify your setup
2. **[Quickstart](quickstart.md)** -- Run your first particle filter and compare with the Kalman filter
3. **[Core Concepts](core-concepts.md)** -- Understand particles, weights, resampling, and the SMC framework
4. **[Choosing a Filter](choosing-filter.md)** -- Decision guide for selecting the right algorithm

!!! tip "Prerequisites"
    - **Python 3.11+** (3.11 and 3.12 supported)
    - **NumPy >= 1.24**, **SciPy >= 1.10**, **Pandas >= 2.0** (installed automatically)
    - *Optional*: [kalmanbox](https://github.com/nodesecon/kalmanbox) for Rao-Blackwellized and Unscented Particle Filters
    - *Optional*: [CuPy](https://cupy.dev/) or [JAX](https://jax.readthedocs.io/) for GPU acceleration

<div class="grid cards" markdown>

- :material-download: **[Installation](installation.md)**

    Install particlefilterbox via pip, from source, or with optional extras

- :material-rocket-launch: **[Quickstart](quickstart.md)**

    Four progressive examples: Bootstrap PF, SIR, Auxiliary PF, and PMMH

- :material-book-open-variant: **[Core Concepts](core-concepts.md)**

    Particles, weights, resampling, ESS, and the predict-update-resample cycle

- :material-map-marker-path: **[Choosing a Filter](choosing-filter.md)**

    Decision guide covering all 10+ filters, smoothers, and PMCMC methods

</div>
