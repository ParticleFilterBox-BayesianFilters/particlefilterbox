# Stata Validation - Bootstrap PF

## Limitations

Stata does NOT have native particle filter implementations.
Validation is limited to:

1. **Linear-Gaussian model only**: via `sspace` (Kalman filter)
2. **No SV model support**: `sspace` requires linear-Gaussian dynamics
3. **Benchmark only**: Stata provides the Kalman filter exact solution,
   which the particle filter should approximate

## Scripts

- `validate_linear_gaussian.do` - Kalman filter via sspace for linear-Gaussian SSM

## Usage

```stata
do validate_linear_gaussian.do
```
