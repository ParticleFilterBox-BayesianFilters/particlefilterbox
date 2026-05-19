# Stata Validation: Advanced Filters (Referencia Limitada)

## Visao Geral

Este diretorio contem o script Stata de referencia para validacao cruzada dos
filtros avancados implementados em `particlefilterbox`. Como o Stata **nao possui**
implementacoes nativas de filtros de particulas avancados, a validacao e limitada
ao benchmark Kalman filter via `sspace` para o modelo linear-Gaussiano.

## Limitacoes do Stata

O comando `sspace` do Stata implementa apenas o **Kalman filter** padrao, que e
aplicavel exclusivamente a modelos lineares-Gaussianos (state-space models).

Os seguintes filtros **nao estao disponiveis** no Stata:

| Filtro | Disponivel no Stata? | Alternativa |
|--------|---------------------|-------------|
| Auxiliary Particle Filter (APF) | Nao | Nenhuma |
| Rao-Blackwellized PF (RBPF) | Nao | Nenhuma |
| Unscented Particle Filter (UPF) | Nao | Nenhuma |
| Regularized Particle Filter (RPF) | Nao | Nenhuma |
| Kalman Filter | Sim (`sspace`) | - |

## O que este benchmark fornece

O script `benchmark_kalman_reference.do` executa o Kalman filter via `sspace` no
modelo linear-Gaussiano (o mesmo usado nos exemplos de Bootstrap/SIR PF). Isso
permite comparar:

- **RBPF vs Kalman**: Para o modelo linear-Gaussiano, o componente Kalman do RBPF
  deve convergir para a solucao exata do Kalman filter. O benchmark Stata fornece
  os valores de referencia (RMSE e log-likelihood) para essa comparacao.

## Arquivos

| Arquivo | Descricao |
|---------|-----------|
| `benchmark_kalman_reference.do` | Script Stata com `sspace` para Kalman filter |
| `results_stata_kalman_ref.csv` | Resultados exportados (gerado apos execucao do .do) |

## Como usar

1. Abra o Stata (versao 14+)
2. Execute: `do benchmark_kalman_reference.do`
3. Os resultados serao exibidos no console e exportados para `results_stata_kalman_ref.csv`

## Comparacao com particlefilterbox

Para comparar os resultados do Kalman (Stata) com o RBPF (particlefilterbox):

```python
import pandas as pd

# Resultados Stata
stata = pd.read_csv("results_stata_kalman_ref.csv")

# Resultados RBPF (particlefilterbox)
# rbpf_results = ...  # ver notebooks em examples/advanced_filters/notebooks/

# Comparar RMSE e filtered states
```

## Nota sobre validacao

Para modelos nao-lineares (ex: volatilidade estocastica), a validacao cruzada dos
filtros avancados deve ser feita com o **R** (pacotes `pomp` e `nimbleSMC`), que
possuem implementacoes completas de SMC/particle filters. Veja o diretorio
`../R_validation/` para os scripts R de referencia.
