# Validacao Stata: SV via Aproximacao Linearizada (`sspace`)

Este diretorio contem scripts Stata que estimam o modelo de Volatilidade
Estocastica (SV) usando a **aproximacao linearizada** sobre `log(y_t^2)` e o
comando `sspace` (state-space).

> **AVISO**: Os resultados sao **APROXIMADOS** e enviesados. Para inferencia
> precisa, use `particlefilterbox` (Python) ou os pacotes R `stochvol` /
> `bsvars`. Este diretorio existe principalmente para **comparacao** e para
> documentar as limitacoes da abordagem QML linearizada em Stata.

---

## Arquivos

| Arquivo | Descricao |
| ------- | --------- |
| `estimate_sv_linearized.do` | Estima SV linearizado via `sspace` sobre `sp500_returns.csv` |
| `results_stata_sv_linearized.csv` | Saida: `time`, `returns`, `h_filtered`, `sigma_filtered` (gerado apos execucao) |

---

## Modelo SV e a Linearizacao QML

### Modelo SV Original (nao linear, nao Gaussiano)

```
y_t = exp(h_t / 2) * eps_t,         eps_t ~ N(0, 1)
h_t = mu + phi * (h_{t-1} - mu) + sigma_h * eta_t,    eta_t ~ N(0, 1)
```

`h_t` e a log-volatilidade latente. O modelo nao tem forma fechada para o
filtro de Kalman porque a observacao `y_t = exp(h_t/2) * eps_t` e nao linear
em `h_t`.

### Aproximacao Linearizada (Harvey, Ruiz & Shephard, 1994)

Tomando `log(y_t^2)`:

```
log(y_t^2) = h_t + log(eps_t^2)
```

O termo `log(eps_t^2)` segue uma distribuicao **log-chi^2(1)**, com:

- `E[log(eps_t^2)]   = -1.2704` (constante de Euler-Mascheroni + log 2)
- `Var[log(eps_t^2)] = pi^2 / 2 ≈ 4.9348`

A aproximacao **QML (Quasi-Maximum Likelihood)** trata esse termo como
**Gaussiano** com a mesma media e variancia. O modelo se torna entao
linear-Gaussiano e estimavel via filtro de Kalman / `sspace`:

```
log(y_t^2) + 1.2704 = h_t + xi_t,     xi_t ~ N(0, pi^2/2)  [aproximacao]
h_t = phi * h_{t-1} + sigma_h * eta_t
```

---

## Vieses Conhecidos da Aproximacao

A literatura (Harvey, Ruiz & Shephard 1994; Kim, Shephard & Chib 1998;
Jacquier, Polson & Rossi 1994) documenta varios vieses:

### 1. `phi` (persistencia) **subestimado**

A linearizacao introduz ruido adicional no canal de observacao, atenuando a
correlacao serial estimada. O `phi` aproximado tipicamente cai 0.02-0.10
abaixo do valor verdadeiro (que costuma ser muito proximo de 1, ~0.95-0.99).

### 2. `sigma_h` **superestimado**

Para compensar o ruido extra, a variancia do choque na equacao de estado e
inflada. Erros relativos de 20-50% sao comuns.

### 3. Leverage (`rho`) **nao estimavel**

A correlacao `corr(eps_t, eta_t) = rho` (efeito assimetrico onde retornos
negativos elevam volatilidade futura) **se perde** ao tomar `y_t^2`, pois o
sinal do retorno desaparece. A aproximacao QML simplesmente **nao consegue**
estimar leverage. Para SP500 (onde `rho ≈ -0.7`), isso e um defeito grave.

### 4. Distribuicao log-chi^2 e **fortemente nao Gaussiana**

A `log-chi^2(1)` e altamente assimetrica (skewness negativa, excesso de
curtose ~4). A aproximacao Gaussiana e particularmente ruim na cauda
inferior, onde retornos proximos de zero (`y_t ≈ 0`) geram `log(y_t^2)` muito
negativo. Esses outliers distorcem a verossimilhanca QML.

### 5. Erros padrao **nao confiaveis**

Como a verossimilhanca nao e a verdadeira, os intervalos de confianca
calculados sob a hipotese Gaussiana subestimam a incerteza real. Inferencia
robusta exigiria sandwich/bootstrap.

### 6. **Inadequado** para previsao de risco (VaR / ES)

Como a forma das caudas e mal capturada, quantis extremos (VaR a 99%,
Expected Shortfall) calculados a partir de `h_filtered` Stata podem subestimar
o risco em periodos de alta volatilidade.

---

## Comparativo Tipico (SP500, ~3000 obs)

| Parametro | particlefilterbox (PMMH) | R `stochvol` (MCMC) | Stata `sspace` (QML) |
| --------- | ------------------------ | -------------------- | -------------------- |
| `mu`      | ~ -9.5                   | ~ -9.5               | ~ -9.0 (vies)        |
| `phi`     | ~ 0.985                  | ~ 0.985              | ~ 0.95 (subestimado) |
| `sigma_h` | ~ 0.15                   | ~ 0.15               | ~ 0.25 (superestimado) |
| `rho`     | ~ -0.70                  | ~ -0.70              | **n/d** (nao estimavel) |

> Os valores Stata sao indicativos; podem variar conforme a amostra. O ponto
> e que **direcao e magnitude relativa dos vieses sao reproduziveis**.

---

## Execucao

```bash
cd /home/guhaase/projetos/particlefilterbox/examples/stochastic_volatility/stata_validation
stata -b do estimate_sv_linearized.do
```

Saida: `results_stata_sv_linearized.csv` com `time`, `returns`, `h_filtered`
(log-volatilidade filtrada) e `sigma_filtered = exp(h_filtered/2)`
(volatilidade condicional).

---

## Quando Usar a Aproximacao Stata?

A aproximacao linearizada via `sspace` e util como:

- **Baseline rapido** para verificar ordem de magnitude da volatilidade
- **Ponto de partida** (chute inicial) para algoritmos exatos
- **Material didatico** mostrando por que metodos exatos sao necessarios

Para qualquer aplicacao seria (precificacao de opcoes, risk management,
research academico), use os metodos exatos disponiveis em
`particlefilterbox` ou em R.

---

## Referencias

- Harvey, A., Ruiz, E. & Shephard, N. (1994). *Multivariate stochastic
  variance models*. Review of Economic Studies, 61(2), 247-264.
- Kim, S., Shephard, N. & Chib, S. (1998). *Stochastic volatility:
  likelihood inference and comparison with ARCH models*. Review of Economic
  Studies, 65(3), 361-393.
- Jacquier, E., Polson, N. G. & Rossi, P. E. (1994). *Bayesian analysis of
  stochastic volatility models*. JBES, 12(4), 371-389.
- Stata Manual: `[TS] sspace` - State-space models.
