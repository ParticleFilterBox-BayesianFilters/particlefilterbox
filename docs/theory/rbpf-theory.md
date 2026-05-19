---
title: Rao-Blackwellized Particle Filters - Teoria
description: Teorema de Rao-Blackwell, decomposicao linear/nao-linear, algoritmo RBPF com integracao kalmanbox, prova de reducao de variancia e extensoes.
---

# Rao-Blackwellized Particle Filters: Teoria

Esta pagina desenvolve a teoria de **Rao-Blackwellized Particle Filters (RBPF)**, que exploram a estrutura parcialmente linear de modelos de estado-espaco para reduzir a variancia de estimativas Monte Carlo via marginalizacao analitica de componentes tratados pelo **kalmanbox**.

---

## 1. Teorema de Rao-Blackwell

### 1.1 Enunciado Classico

O **Teorema de Rao-Blackwell** e o resultado fundamental que motiva toda a classe de algoritmos RBPF.

!!! abstract "Teorema: Rao-Blackwell"
    Seja $\hat{\theta}$ um estimador de $\theta$ e $T$ uma **estatistica suficiente** (ou, mais geralmente, qualquer estatistica). Defina o estimador Rao-Blackwellizado:

    $$
    \tilde{\theta} = \mathbb{E}[\hat{\theta} \mid T]
    $$

    Entao:

    $$
    \text{Var}(\tilde{\theta}) \leq \text{Var}(\hat{\theta})
    $$

    com igualdade se e somente se $\hat{\theta}$ ja e funcao de $T$ quase certamente.

??? note "Sketch da Prova"
    A prova segue da **decomposicao da variancia total** (lei de Eve):

    $$
    \text{Var}(\hat{\theta}) = \mathbb{E}\left[\text{Var}(\hat{\theta} \mid T)\right] + \text{Var}\left(\mathbb{E}[\hat{\theta} \mid T]\right)
    $$

    O primeiro termo e a **variancia intra-grupo** (variancia residual apos condicionar em $T$) e e sempre $\geq 0$. Portanto:

    $$
    \text{Var}(\hat{\theta}) = \underbrace{\mathbb{E}\left[\text{Var}(\hat{\theta} \mid T)\right]}_{\geq 0} + \text{Var}(\tilde{\theta})
    \geq \text{Var}(\tilde{\theta})
    $$

    A igualdade vale sse $\text{Var}(\hat{\theta} \mid T) = 0$ q.c., ou seja, $\hat{\theta}$ e funcao mensuravel de $T$. $\blacksquare$

### 1.2 Aplicacao a Particle Filters

No contexto de particle filters, o estimador padrao de uma funcional $\varphi$ e:

$$
\hat{I}_N = \sum_{i=1}^{N} W_t^{(i)} \, \varphi(x_t^{(i)})
$$

Suponha que o estado $x_t$ admita uma decomposicao $x_t = (z_t, s_t)$ tal que, condicionalmente em $s_{1:t}$, a distribuicao de $z_t$ possui forma **analitica** (e.g., Gaussiana). Entao podemos "condicionar" (Rao-Blackwellizar) em $s_t$:

$$
\tilde{I}_N = \sum_{i=1}^{N} W_t^{(i)} \, \mathbb{E}\left[\varphi(z_t, s_t^{(i)}) \mid s_{1:t}^{(i)}, y_{1:t}\right]
$$

!!! tip "Intuicao"
    Em vez de amostrar **todo** o estado $x_t = (z_t, s_t)$ por particulas, amostramos apenas o componente nao-linear $s_t$ e tratamos $z_t$ **analiticamente** via Kalman filter. Isso reduz a dimensao efetiva do espaco amostral, diminuindo a variancia.

### 1.3 Magnitude da Reducao

A reducao de variancia e proporcional a dimensao do componente marginalizado. Se $d_z = \dim(z_t)$ e $d_s = \dim(s_t)$:

$$
\frac{\text{Var}(\tilde{I}_N)}{\text{Var}(\hat{I}_N)} \leq C \cdot \left(\frac{d_s}{d_s + d_z}\right)^{\alpha}
$$

para constantes $C > 0$ e $\alpha \geq 1$ que dependem do modelo. Em termos qualitativos:

| Razao $d_z / (d_z + d_s)$ | Reducao de variancia | Exemplo tipico |
|---|---|---|
| $> 0.8$ | Dramatica (ordens de magnitude) | DSGE com poucos choques nao-lineares |
| $0.5 - 0.8$ | Substancial | Factor models com regime switching |
| $< 0.3$ | Moderada | Modelos com forte nao-linearidade |

---

## 2. Decomposicao Linear/Nao-Linear

### 2.1 Modelo com Estrutura Mista

Considere o modelo de estado-espaco onde o estado $x_t$ e particionado como $x_t = (z_t, s_t)$:

$$
\begin{aligned}
s_t &\sim f_s(s_t \mid s_{t-1}, z_{t-1}) & &\text{(componente nao-linear)} \\
z_t &= A(s_t) \, z_{t-1} + B(s_t) \, u_t + \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q(s_t)) & &\text{(componente linear condicional)} \\
y_t &= C(s_t) \, z_t + D(s_t) \, v_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, R(s_t)) & &\text{(observacao)}
\end{aligned}
$$

!!! info "Estrutura Chave"
    | Componente | Variavel | Dinamica | Tratamento |
    |---|---|---|---|
    | Nao-linear | $s_t$ | Arbitraria: $f_s(s_t \mid s_{t-1}, z_{t-1})$ | Particle filter |
    | Linear condicional | $z_t$ | Linear-Gaussiana dado $s_t$ | Kalman filter (**kalmanbox**) |
    | Observacao | $y_t$ | Linear em $z_t$ dado $s_t$ | Integrada no Kalman update |

A propriedade fundamental e que, **condicionalmente em $s_{1:t}$**, o sistema para $z_t$ e **linear-Gaussiano**, e portanto tratavel analiticamente pelo filtro de Kalman.

### 2.2 Distribuicao Condicional via Kalman

Condicionalmente na trajetoria $s_{1:t}^{(i)}$ da $i$-esima particula:

$$
p(z_t \mid s_{1:t}^{(i)}, y_{1:t}) = \mathcal{N}\left(z_t \mid \hat{z}_t^{(i)}, P_t^{(i)}\right)
$$

onde $\hat{z}_t^{(i)}$ e $P_t^{(i)}$ sao a media e covariancia do filtro de Kalman executado com as matrizes $A(s_t^{(i)})$, $C(s_t^{(i)})$, $Q(s_t^{(i)})$, $R(s_t^{(i)})$.

### 2.3 Integracao com kalmanbox

!!! note "Integracao kalmanbox: Arquitetura do RBPF"
    Cada particula $i$ no RBPF carrega **seu proprio filtro de Kalman** via kalmanbox. A estrutura e:

    **Particula $i$ no tempo $t$:**

    $$
    \text{Particula}^{(i)}_t = \left(s_t^{(i)}, \underbrace{\hat{z}_t^{(i)}, P_t^{(i)}}_{\text{kalmanbox: KalmanFilter}^{(i)}}, W_t^{(i)}\right)
    $$

    - $s_t^{(i)}$: estado nao-linear amostrado por particle filter
    - $\hat{z}_t^{(i)}, P_t^{(i)}$: media e covariancia do Kalman filter condicional
    - $W_t^{(i)}$: peso da particula (baseado na likelihood marginal)

    Isto significa que com $N$ particulas, mantemos $N$ instancias independentes de `KalmanFilter` do kalmanbox, cada uma condicionada em uma trajetoria diferente de $s_{1:t}$.

A distribuicao de filtragem completa e representada como uma **mistura de Gaussianas**:

$$
p(z_t, s_t \mid y_{1:t}) \approx \sum_{i=1}^{N} W_t^{(i)} \, \underbrace{\delta_{s_t^{(i)}}(s_t)}_{\text{particula}} \, \underbrace{\mathcal{N}(z_t \mid \hat{z}_t^{(i)}, P_t^{(i)})}_{\text{kalmanbox}}
$$

---

## 3. Algoritmo RBPF

### 3.1 Inicializacao

Para $i = 1, \ldots, N$:

$$
\begin{aligned}
s_0^{(i)} &\sim \mu_s(s_0) \\
\hat{z}_0^{(i)} &= \mathbb{E}[z_0] \\
P_0^{(i)} &= \text{Cov}(z_0)
\end{aligned}
$$

Cada particula inicializa uma instancia do kalmanbox com prior $\mathcal{N}(\hat{z}_0^{(i)}, P_0^{(i)})$.

### 3.2 Passo de Predicao

!!! note "Algoritmo: RBPF Predict"
    Para cada particula $i = 1, \ldots, N$:

    **Componente nao-linear (Particle Filter):**

    $$
    s_t^{(i)} \sim q(s_t \mid s_{t-1}^{(i)}, y_t)
    $$

    onde $q$ e a distribuicao proposal (e.g., prior: $q = f_s$).

    **Componente linear (kalmanbox predict):**

    $$
    \begin{aligned}
    \hat{z}_{t|t-1}^{(i)} &= A(s_t^{(i)}) \, \hat{z}_{t-1}^{(i)} + B(s_t^{(i)}) \, u_t \\
    P_{t|t-1}^{(i)} &= A(s_t^{(i)}) \, P_{t-1}^{(i)} \, A(s_t^{(i)})^\top + Q(s_t^{(i)})
    \end{aligned}
    $$

    Isto corresponde a chamar `kalmanbox.KalmanFilter.predict()` para cada particula com matrizes dependentes de $s_t^{(i)}$.

### 3.3 Passo de Atualizacao

!!! note "Algoritmo: RBPF Update"
    Para cada particula $i = 1, \ldots, N$:

    **Inovacao e Kalman Gain (kalmanbox update):**

    $$
    \begin{aligned}
    \nu_t^{(i)} &= y_t - C(s_t^{(i)}) \, \hat{z}_{t|t-1}^{(i)} - D(s_t^{(i)}) \, v_t \\
    S_t^{(i)} &= C(s_t^{(i)}) \, P_{t|t-1}^{(i)} \, C(s_t^{(i)})^\top + R(s_t^{(i)}) \\
    K_t^{(i)} &= P_{t|t-1}^{(i)} \, C(s_t^{(i)})^\top \, \left(S_t^{(i)}\right)^{-1}
    \end{aligned}
    $$

    **Estado e covariancia atualizada:**

    $$
    \begin{aligned}
    \hat{z}_t^{(i)} &= \hat{z}_{t|t-1}^{(i)} + K_t^{(i)} \, \nu_t^{(i)} \\
    P_t^{(i)} &= (I - K_t^{(i)} \, C(s_t^{(i)})) \, P_{t|t-1}^{(i)}
    \end{aligned}
    $$

    **Peso (likelihood marginal):**

    $$
    \tilde{w}_t^{(i)} = \frac{f_s(s_t^{(i)} \mid s_{t-1}^{(i)})}{q(s_t^{(i)} \mid s_{t-1}^{(i)}, y_t)} \cdot p(y_t \mid s_{1:t}^{(i)}, y_{1:t-1})
    $$

    onde a **likelihood marginal** e obtida do Kalman filter:

    $$
    p(y_t \mid s_{1:t}^{(i)}, y_{1:t-1}) = \mathcal{N}\left(\nu_t^{(i)} \mid 0, S_t^{(i)}\right)
    $$

!!! tip "Intuicao: Likelihood Marginal"
    O peso de cada particula nao depende de $z_t$ — ele e calculado a partir da **predictive likelihood** do Kalman filter (inovacao e sua covariancia). Isso e possivel porque $z_t$ foi marginalizado analiticamente. Esta e a essencia do Rao-Blackwellization: integrar fora o que se pode, pesar pelo que resta.

### 3.4 Resampling

Apos o update, aplicamos resampling padrao (e.g., multinomial, residual, systematic) nos pesos $\{W_t^{(i)}\}$. Quando uma particula $j$ e replicada, **todo o estado** deve ser copiado:

$$
\text{Replicar particula } j: \quad \left(s_t^{(j)}, \hat{z}_t^{(j)}, P_t^{(j)}\right)
$$

!!! warning "Copia do Estado Kalman"
    No resampling, cada particula replicada deve receber uma **copia independente** da media $\hat{z}_t$ e covariancia $P_t$ do kalmanbox. Compartilhar referencias entre particulas levaria a atualizacoes incorretas nos passos seguintes.

### 3.5 Complexidade Computacional

| Operacao | Bootstrap PF | RBPF |
|---|---|---|
| Predict (por particula) | $O(d_x)$ | $O(d_s + d_z^2)$ |
| Update (por particula) | $O(d_x)$ | $O(d_z^2 d_y + d_y^3)$ |
| Total por passo | $O(N \cdot d_x)$ | $O(N \cdot (d_z^2 d_y + d_y^3))$ |
| Particulas necessarias | $N_{\text{BPF}}$ | $N_{\text{RBPF}} \ll N_{\text{BPF}}$ |

!!! info "Trade-off Computacional"
    Cada particula RBPF e mais cara que uma particula Bootstrap ($O(d_z^2)$ vs $O(1)$ para a operacao Kalman). Porem, a reducao de variancia permite usar um numero **dramaticamente menor** de particulas: tipicamente $N_{\text{RBPF}} \sim 10$–$100$ vs $N_{\text{BPF}} \sim 1000$–$10000$. O custo total e frequentemente menor.

---

## 4. Prova Formal de Reducao de Variancia

### 4.1 Setup

Considere estimar $\mathbb{E}[\varphi(x_t) \mid y_{1:t}]$ onde $x_t = (z_t, s_t)$. Definimos dois estimadores:

**Estimador padrao (BPF):** amostramos $x_t^{(i)} = (z_t^{(i)}, s_t^{(i)})$ conjuntamente:

$$
\hat{I}_N^{\text{BPF}} = \sum_{i=1}^{N} W_t^{(i)} \, \varphi(z_t^{(i)}, s_t^{(i)})
$$

**Estimador RBPF:** amostramos apenas $s_t^{(i)}$ e marginalizamos $z_t$:

$$
\hat{I}_N^{\text{RBPF}} = \sum_{i=1}^{N} W_t^{(i)} \, \bar{\varphi}(s_t^{(i)})
$$

onde $\bar{\varphi}(s_t) = \mathbb{E}[\varphi(z_t, s_t) \mid s_{1:t}, y_{1:t}]$ e calculavel analiticamente via kalmanbox.

### 4.2 Resultado Principal

!!! abstract "Teorema: Reducao de Variancia do RBPF"
    Sob condicoes de regularidade (pesos limitados, $\varphi$ integravel), para o mesmo numero de particulas $N$:

    $$
    \text{Var}\left(\hat{I}_N^{\text{RBPF}}\right) \leq \text{Var}\left(\hat{I}_N^{\text{BPF}}\right)
    $$

    Mais precisamente, a reducao de variancia e dada por:

    $$
    \text{Var}\left(\hat{I}_N^{\text{BPF}}\right) - \text{Var}\left(\hat{I}_N^{\text{RBPF}}\right) = \frac{1}{N} \, \mathbb{E}_{q}\left[w^2(s_{1:t}) \, \text{Var}\left(\varphi(z_t, s_t) \mid s_{1:t}, y_{1:t}\right)\right]
    $$

    que e estritamente positiva exceto quando $\varphi$ nao depende de $z_t$.

??? note "Sketch da Prova"
    **Passo 1:** Decomposicao da variancia condicional.

    Para uma unica particula com peso $w^{(i)}$, considere o termo $w^{(i)} \varphi(z_t^{(i)}, s_t^{(i)})$:

    $$
    \text{Var}_q\left[w \, \varphi(z, s)\right] = \text{Var}_q\left[\mathbb{E}[w \, \varphi(z, s) \mid s]\right] + \mathbb{E}_q\left[\text{Var}[w \, \varphi(z, s) \mid s]\right]
    $$

    **Passo 2:** Note que os pesos do RBPF dependem apenas de $s_{1:t}$ (nao de $z_t$, que foi marginalizado):

    $$
    \mathbb{E}[w(s_{1:t}) \, \varphi(z_t, s_t) \mid s_{1:t}, y_{1:t}] = w(s_{1:t}) \, \bar{\varphi}(s_t)
    $$

    Portanto:

    $$
    \text{Var}_q[w \, \varphi(z, s)] = \underbrace{\text{Var}_q[w \, \bar{\varphi}(s)]}_{\text{variancia RBPF}} + \underbrace{\mathbb{E}_q\left[w^2 \, \text{Var}[\varphi \mid s, y]\right]}_{\geq 0}
    $$

    **Passo 3:** O segundo termo e estritamente positivo quando $\varphi$ depende de $z_t$ e $\text{Var}[\varphi \mid s, y] > 0$, provando a desigualdade estrita. $\blacksquare$

### 4.3 Quando o RBPF Domina

A magnitude da reducao depende de dois fatores:

1. **Proporcao linear**: Quanto maior $d_z / (d_z + d_s)$, maior a reducao
2. **Variancia condicional**: Quanto maior $\text{Var}[\varphi(z_t, s_t) \mid s_{1:t}, y_{1:t}]$, maior o ganho

!!! example "Aplicacoes onde RBPF e Superior"
    **Modelos DSGE:**
    A maioria dos estados e linear condicional a poucos parametros de regime ou choques nao-lineares. Tipicamente $d_z \sim 10$–$30$ vs $d_s \sim 1$–$5$.

    **Factor Models com Regime Switching:**
    Os fatores latentes seguem dinamica linear, enquanto o regime $s_t \in \{1, \ldots, K\}$ e discreto e nao-linear.

    **Switching Linear Models:**
    Todo o sistema e linear condicional ao regime: $d_z$ e a dimensao total do estado, $d_s$ e o regime discreto.

### 4.4 Convergencia Assintotica

!!! abstract "Teorema: CLT para RBPF"
    Sob condicoes de regularidade (Chopin, 2004; Chen & Liu, 2000), o estimador RBPF satisfaz:

    $$
    \sqrt{N}\left(\hat{I}_N^{\text{RBPF}} - I\right) \xrightarrow{d} \mathcal{N}\left(0, \sigma_{\text{RBPF}}^2\right)
    $$

    onde $\sigma_{\text{RBPF}}^2 \leq \sigma_{\text{BPF}}^2$, com a variancia assintotica:

    $$
    \sigma_{\text{RBPF}}^2 = \text{Var}_{\pi}\left[\frac{p(s_{1:t} \mid y_{1:t})}{q(s_{1:t} \mid y_{1:t})} \, \bar{\varphi}(s_t)\right]
    $$

---

## 5. Extensoes

### 5.1 RBPF com Extended Kalman Filter

Quando o componente "linear" apresenta **leve nao-linearidade**, podemos usar o Extended Kalman Filter (EKF) dentro do RBPF:

$$
\begin{aligned}
z_t &= f_z(z_{t-1}, s_t) + \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q(s_t)) \\
y_t &= h(z_t, s_t) + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, R(s_t))
\end{aligned}
$$

O EKF lineariza em torno do estado predito:

$$
\begin{aligned}
A_t^{(i)} &= \left.\frac{\partial f_z}{\partial z}\right|_{z = \hat{z}_{t-1}^{(i)}, s = s_t^{(i)}} \\
C_t^{(i)} &= \left.\frac{\partial h}{\partial z}\right|_{z = \hat{z}_{t|t-1}^{(i)}, s = s_t^{(i)}}
\end{aligned}
$$

!!! warning "Perda da Optimalidade"
    Com EKF, a distribuicao condicional $p(z_t \mid s_{1:t}^{(i)}, y_{1:t})$ ja nao e exatamente Gaussiana. A Rao-Blackwellization e **aproximada** — ainda reduz variancia, mas nao de forma otima. O erro de aproximacao cresce com a nao-linearidade de $f_z$ e $h$.

### 5.2 RBPF com Unscented Kalman Filter

Para nao-linearidades mais fortes, o **Unscented Kalman Filter (UKF)** oferece melhor aproximacao sem necessidade de Jacobianos:

$$
\mathcal{Z} = \left\{\hat{z}, \quad \hat{z} \pm \sqrt{(d_z + \lambda) P}\right\}
$$

Os sigma points sao propagados pela funcao nao-linear e recombinados para obter media e covariancia preditas. Vantagens:

- Nao requer derivadas analiticas de $f_z$ ou $h$
- Captura momentos ate segunda ordem exatamente para funcoes nao-lineares
- Custo adicional de $O(d_z)$ sigma points por particula

### 5.3 Mixture RBPF

Para modelos onde a componente "linear" tem **distribuicao condicional multimodal**, podemos representar:

$$
p(z_t \mid s_{1:t}^{(i)}, y_{1:t}) \approx \sum_{m=1}^{M} \alpha_m^{(i)} \, \mathcal{N}\left(z_t \mid \hat{z}_{t,m}^{(i)}, P_{t,m}^{(i)}\right)
$$

Cada particula carrega nao um, mas $M$ filtros de Kalman, formando uma **Gaussian Mixture**. Isso e relevante para:

- Modelos com observacao discreta (quantizada)
- Modelos com restricoes de desigualdade
- Likelihood multimodal condicionalmente em $s_t$

!!! danger "Crescimento Exponencial"
    Sem poda, o numero de componentes Gaussianas cresce exponencialmente com $t$. Estrategias de **merging** e **pruning** de componentes (e.g., minimizacao de KL-divergencia) sao essenciais para manter o custo computacional tratavel.

### 5.4 Comparacao de Extensoes

| Extensao | Linearidade necessaria | Custo por particula | Acuracia |
|---|---|---|---|
| RBPF + KF (padrao) | Exata | $O(d_z^2 d_y)$ | Exata (otima) |
| RBPF + EKF | Aproximada (leve) | $O(d_z^2 d_y)$ | Primeira ordem |
| RBPF + UKF | Aproximada (moderada) | $O(d_z^3)$ | Segunda ordem |
| Mixture RBPF | Multimodal | $O(M \cdot d_z^2 d_y)$ | Exata (com $M \to \infty$) |

---

## Referencias

- **Doucet, A., de Freitas, N., Murphy, K., & Russell, S.** (2000). Rao-Blackwellised Particle Filtering for Dynamic Bayesian Networks. *Proceedings of the 16th Conference on Uncertainty in Artificial Intelligence (UAI)*, 176–183.
- **Schon, T., Gustafsson, F., & Nordlund, P.-J.** (2005). Marginalized Particle Filters for Mixed Linear/Nonlinear State-Space Models. *IEEE Transactions on Signal Processing*, 53(7), 2168–2177.
- **Chen, R. & Liu, J. S.** (2000). Mixture Kalman Filters. *Journal of the Royal Statistical Society: Series B*, 62(3), 493–508.
- **Chopin, N.** (2004). Central Limit Theorem for Sequential Monte Carlo Methods and its Application to Bayesian Inference. *Annals of Statistics*, 32(6), 2385–2411.
- **Casella, G. & Robert, C. P.** (1996). Rao-Blackwellisation of Sampling Schemes. *Biometrika*, 83(1), 81–94.
- **Andrieu, C. & Doucet, A.** (2002). Particle Filtering for Partially Observed Gaussian State Space Models. *Journal of the Royal Statistical Society: Series B*, 64(4), 827–836.
- **Gustafsson, F.** (2010). Particle Filter Theory and Practice with Positioning Applications. *IEEE Aerospace and Electronic Systems Magazine*, 25(7), 53–82.
- **Lindsten, F. & Schon, T. B.** (2013). Backward Simulation Methods for Monte Carlo Statistical Inference. *Foundations and Trends in Machine Learning*, 6(1), 1–143.
