# Stata Validation - Applications (Reference Limitada)

Este diretorio contem scripts Stata de referencia para validacao cruzada
das aplicacoes em `examples/applications/`. O suporte do Stata e
**limitado** - apenas o modelo DSGE linearizado pode ser estimado via
`sspace` (Kalman filter). Para jump-diffusion e SIR nao ha alternativa
nativa no Stata.

---

## Escopo da Validacao

| Aplicacao            | Suporte Stata | Comando         | Observacao                                |
|----------------------|---------------|-----------------|-------------------------------------------|
| DSGE (3 eq linear)   | Parcial       | `sspace`        | Kalman filter no modelo linearizado       |
| DSGE nao-linear      | Nao           | -               | Stata nao tem particle filter             |
| Jump-diffusion (SDE) | Nao           | -               | Sem suporte para jumps em tempo continuo  |
| SIR epidemiologico   | Nao           | -               | Sem suporte para modelos compartimentais  |

A validacao cruzada principal das aplicacoes deve ser feita com R
(`yuima` para jump-diffusion, `EpiEstim`/`pomp` para SIR). O Stata
e mantido aqui apenas como referencia para o caso DSGE.

---

## Arquivos

```
stata_validation/
├── README.md                       # Este arquivo
├── benchmark_dsge_kalman.do        # Estima DSGE linearizado via sspace
└── results_stata_dsge.csv          # Estados filtrados (gerado pelo .do)
```

### `benchmark_dsge_kalman.do`

Estima o modelo New Keynesian de 3 equacoes (output gap, inflacao, juros)
como state-space via `sspace`. Os estados filtrados servem como benchmark
para o particle filter aplicado ao mesmo conjunto de dados em
`examples/applications/notebooks/01_dsge_estimation.ipynb`.

**Entrada**: `examples/applications/data/treasury_yields.csv`

**Saida**: `results_stata_dsge.csv` com colunas
`time, y_x, y_pi, y_r, x_filt, pi_filt, r_filt`.

---

## Como Executar

Requer Stata 14 ou superior.

```bash
stata -b do benchmark_dsge_kalman.do
```

Ou no Stata interativo:

```stata
do benchmark_dsge_kalman.do
```

---

## Limitacoes

1. **Sem particle filter nativo** - Stata nao implementa SMC, bootstrap PF
   ou variantes. Toda a validacao nao-linear/nao-Gaussiana deve ser feita
   em R ou Python.
2. **Sem jump-diffusion** - O comando `arima` e relacionados nao suportam
   processos de salto. Use R `yuima` para validar
   `examples/applications/notebooks/02_jump_diffusion.ipynb`.
3. **Sem SIR** - Stata nao tem rotinas para sistemas EDO compartimentais
   estocasticos. Use R `pomp` ou `EpiEstim` para validar
   `examples/applications/notebooks/03_sir_epidemic.ipynb`.
4. **DSGE linearizado apenas** - O `sspace` requer linearidade e
   distribuicoes Gaussianas. O particle filter da particlefilterbox pode
   tratar nao-linearidades e ruido nao-Gaussiano que o Kalman filter do
   Stata nao consegue.

---

## Referencia Cruzada

- Validacao R completa: `examples/applications/R_validation/`
- Notebooks Python: `examples/applications/notebooks/`
- Dados: `examples/applications/data/`
