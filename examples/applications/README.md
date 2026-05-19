# Applications - particlefilterbox

Notebooks de aplicacoes do `particlefilterbox` em dominios alem de
volatilidade financeira. Esta colecao demonstra a versatilidade da
biblioteca em tres areas distintas:

1. **DSGE** (macroeconomia)
2. **Jump-Diffusion** (pricing de ativos com saltos)
3. **SIR** (epidemiologia)

---

## Estrutura

```
applications/
|-- README.md              <- este arquivo
|-- data/                  <- datasets simulados e gerador
|   |-- generate_data.py
|   |-- simulated_jump_diff.csv
|   |-- simulated_sir.csv
|   `-- treasury_yields.csv
|-- notebooks/             <- notebooks das aplicacoes (F7.2-F7.5)
|-- solutions/             <- solucoes de referencia dos exercicios
|-- R_validation/          <- scripts R (yuima) para cross-check
`-- stata_validation/      <- scripts Stata (sspace) quando aplicavel
```

---

## Aplicacoes

### 1. DSGE - Dynamic Stochastic General Equilibrium

Modelo macroeconomico simples de 3 equacoes:

- **IS curve**: output gap depende de taxa de juros real esperada
- **Phillips curve**: inflacao depende do output gap
- **Taylor rule**: banco central ajusta juros em resposta a inflacao e
  output gap

Estados latentes: output gap, inflacao, taxa de juros neutra.
Observaveis: as mesmas variaveis, com ruido de medida.

Dataset: `treasury_yields.csv` (500 obs, AR(1) linear-Gaussian)

### 2. Jump-Diffusion (Merton / Kou / Bates)

Modelos de retornos com saltos descontinuos:

```
r_t = mu + sigma * eps_t + J_t * Z_t

onde J_t ~ Bernoulli(lambda) e Z_t ~ N(mu_j, sigma_j^2)
```

Particle filter estima:
- intensidade de saltos (lambda)
- tamanho medio e variancia dos saltos (mu_j, sigma_j)
- probabilidade posterior de salto em cada t

Dataset: `simulated_jump_diff.csv` (1000 obs, modelo Merton)

Parametros DGP:
- `mu = 0.0005`
- `sigma = 0.01`
- `lam = 0.05`
- `mu_j = -0.02`
- `sigma_j = 0.03`

### 3. SIR Epidemiologico

Modelo compartimental Susceptible-Infected-Recovered com observacao
parcial dos infectados:

```
S_{t+1} = S_t - beta * S_t * I_t / N
I_{t+1} = I_t + beta * S_t * I_t / N - gamma * I_t
R_{t+1} = R_t + gamma * I_t
y_t     ~ Poisson(obs_rate * I_t)
```

Particle filter recupera trajetorias latentes (S, I, R) a partir
apenas de `y_obs` (casos reportados).

Dataset: `simulated_sir.csv` (200 obs)

Parametros DGP:
- `beta = 0.3` (taxa de transmissao)
- `gamma = 0.1` (taxa de recuperacao)
- `obs_rate = 0.5` (fracao de casos reportados)
- `N_pop = 10000`
- R0 nominal = beta / gamma = 3.0

---

## Datasets

Todos os datasets sao **simulados com seed fixa (42)** e, portanto,
totalmente reprodutiveis. Todas as series sao livres de `NaN`/`Inf`.

| Arquivo                     | Linhas | Colunas                                                         |
|-----------------------------|-------:|-----------------------------------------------------------------|
| `simulated_jump_diff.csv`   |  1000  | `t, returns, jump_true, jump_size_true`                         |
| `simulated_sir.csv`         |   200  | `t, S_true, I_true, R_true, y_obs`                              |
| `treasury_yields.csv`       |   500  | `t, output_gap_obs, inflation_obs, interest_rate_obs, *_true`   |

Para regenerar:

```bash
cd data
python3 generate_data.py
```

---

## Roadmap (FASE 7)

| Subfase | Topico                         | Entregavel                     |
|---------|--------------------------------|--------------------------------|
| F7.1    | Setup (esta subfase)           | estrutura + datasets           |
| F7.2    | DSGE notebook                  | `notebooks/dsge_pf.ipynb`      |
| F7.3    | Jump-diffusion notebook        | `notebooks/jump_diffusion.ipynb` |
| F7.4    | SIR notebook                   | `notebooks/sir_epidemic.ipynb` |
| F7.5    | Validacao cruzada (R/Stata)    | scripts em `R_validation/`     |

---

## Referencias

- Fernandez-Villaverde & Rubio-Ramirez (2007), "Estimating
  macroeconomic models: A likelihood approach", *RES*.
- Johannes, Polson & Stroud (2009), "Optimal filtering of jump
  diffusions", *RFS*.
- Dukic, Lopes & Polson (2012), "Tracking epidemics with Google flu
  trends and particle filtering", *JASA*.
