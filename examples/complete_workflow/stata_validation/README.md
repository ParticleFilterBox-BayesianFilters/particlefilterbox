# Validacao Stata - particlefilterbox

Este diretorio contem scripts Stata que servem como benchmarks complementares
para a validacao do `particlefilterbox`. O Stata e utilizado apenas onde sua
funcionalidade nativa permite comparacoes diretas com a saida da biblioteca.

## Conteudo

- `full_stata_comparison.do` - script consolidado com os dois benchmarks
  oficiais (Kalman exato + aproximacao SV linearizada).
- `results_stata_linear_gaussian.csv` - estados filtrados/suavizados pelo
  Kalman exato (`sspace`) sobre o dataset linear-Gaussian (gerado apos
  execucao).
- `results_stata_sv_approx.csv` - estimativa aproximada da log-volatilidade
  (sspace sobre o modelo linearizado de Harvey, Ruiz e Shephard, 1994),
  gerado apenas se `sspace` convergir.

## Como executar

```
cd /home/guhaase/projetos/particlefilterbox/examples/complete_workflow/stata_validation
stata -b do full_stata_comparison.do
```

Os arquivos CSV produzidos podem ser comparados com a saida correspondente
do `particlefilterbox` (Kalman filter exato e Bootstrap PF / SV).

## Capacidades do Stata cobertas pelo benchmark

| Modelo | Comando | Tipo de benchmark |
|---|---|---|
| Linear-Gaussian state space | `sspace` | Kalman filter/smoother **exato** |
| Stochastic Volatility (linearizado) | `sspace` sobre `log y_t^2` | Aproximado (Harvey-Ruiz-Shephard) |
| MLE de parametros (linear) | `sspace` (default) | Comparacao com posterior PMMH |

## Limitacoes do Stata para particle filtering

O Stata nao possui suporte nativo para os principais metodos cobertos pelo
`particlefilterbox`. As limitacoes sao documentadas tanto no script `.do`
quanto aqui:

### Filtros nao-lineares / nao-Gaussianos (NAO suportado)
- Bootstrap Particle Filter
- SIR (Sequential Importance Resampling) generico
- Auxiliary Particle Filter (APF)
- Rao-Blackwellized Particle Filter (RBPF)

### Smoothers (NAO suportado)
- Forward-Filtering Backward-Smoothing marginal (FFBSm)
- Forward-Filtering Backward-Sampling (FFBSi)
- Two-filter smoothers
- Fixed-lag smoothers
- Backward simulation smoothers

### SMC samplers (NAO suportado)
- SMC sampler para distribuicoes estaticas
- SMC^2 (Chopin et al., 2013)
- IBIS / Iterated Batch Importance Sampling
- Waste-free SMC (Dau e Chopin, 2022)

### PMCMC (NAO suportado)
- Particle Marginal Metropolis-Hastings (PMMH)
- Particle Gibbs
- Particle Gibbs com Ancestor Sampling (PGAS)

### Modelos especificos (NAO suportado nativamente)
- SV com leverage (correlacao entre choques de retorno e log-volatilidade)
- SV com jumps (Bates, SVJD)
- Factor stochastic volatility (multivariado)
- Jump-diffusion: Merton, Kou, Bates
- SIR epidemiologico (compartimental)

### Diagnosticos PF/SMC (NAO suportado)
- Effective Sample Size (ESS)
- Weight degeneracy diagnostics
- Resampling diagnostics
- Convergencia em N (numero de particulas)
- Path degeneracy diagnostics

## Sumario: Stata vs particlefilterbox

| Capacidade | Stata | particlefilterbox |
|---|---|---|
| Kalman filter/smoother | OK (sspace) | OK (referencia) |
| Bootstrap PF | NAO | OK |
| SIR/APF/RBPF | NAO | OK |
| FFBSm/FFBSi/two-filter | NAO | OK |
| SMC sampler | NAO | OK |
| SMC^2/IBIS/waste-free | NAO | OK |
| PMMH/PG/PGAS | NAO | OK |
| SV basico | Aproximado | OK (exato via PF) |
| SV com leverage/jumps | NAO | OK |
| Jump-diffusion | NAO | OK |
| SIR epidemiologico | NAO | OK |
| Diagnosticos PF/SMC | NAO | OK |

## Conclusao

O Stata fornece dois benchmarks valiosos:
1. **Linear-Gaussian (Kalman exato)**: padrao-ouro para validar a
   consistencia do PF quando o modelo admite solucao analitica.
2. **SV aproximado**: referencia rapida (porem aproximada) para
   stochastic volatility univariado.

Para todos os outros casos cobertos pelo `particlefilterbox`, a validacao
externa e feita exclusivamente via R (pacotes `SMC`, `pomp`, `nimbleSMC`,
`stochvol`, `bsvars`, `yuima`).
