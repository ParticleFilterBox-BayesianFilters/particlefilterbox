# Particle Filters

## Overview

Particle filters (Sequential Monte Carlo methods for filtering) approximate the
filtering distribution p(x_t | y_{1:t}) using a set of weighted particles.

## Available Filters

- **BootstrapFilter**: Standard SIR filter (Gordon et al., 1993)
- **AuxiliaryParticleFilter**: APF (Pitt & Shephard, 1999)
- **ExtendedKalmanPF**: EKF proposal
- **UnscentedKalmanPF**: UKF proposal

## Usage

See [Getting Started](../getting_started.md) for basic usage.
