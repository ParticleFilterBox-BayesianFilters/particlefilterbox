---
title: Teoria de Particle Filters
description: Fundamentos teoricos de particle filters - Hidden Markov Models, Bootstrap PF, optimal proposal e resultados assintoticos.
---

# Teoria de Particle Filters

Esta pagina desenvolve a teoria de **particle filters** (filtros de particulas) como aproximacoes Monte Carlo da recursao de filtragem Bayesiana para Hidden Markov Models.

---

## 1. Hidden Markov Models e Filtragem

### 1.1 Modelo de Estado-Espaco

Um **Hidden Markov Model (HMM)** ou modelo de estado-espaco e definido por:

$$
\begin{aligned}
x_0 &\sim \mu(x_0) & &\text{(distribuicao inicial)} \\
x_t &\sim f(x_t | x_{t-1}) & &\text{(equacao de transicao de estado)} \\
y_t &\sim g(y_t | x_t) & &\text{(equacao de observacao)}
\end{aligned}
$$

onde $x_t \in \mathcal{X} \subseteq \mathbb{R}^{d_x}$ e o **estado latente** (nao-observado) e $y_t \in \mathcal{Y} \subseteq \mathbb{R}^{d_y}$ e a **observacao**.

!!! info "Componentes do Modelo"
    | Componente | Notacao | Descricao |
    |---|---|---|
    | Distribuicao inicial | $\mu(x_0)$ | Prior sobre o estado inicial |
    | Kernel de transicao | $f(x_t \mid x_{t-1})$ | Dinamica do estado (geralmente nao-linear) |
    | Likelihood | $g(y_t \mid x_t)$ | Modelo de observacao |
    | Filtragem | $p(x_t \mid y_{1:t})$ | Distribuicao de interesse |

### 1.2 Objetivo: Distribuicao de Filtragem

O objetivo central e calcular a **distribuicao de filtragem**:

$$
\pi_t(x_t) \triangleq p(x_t | y_{1:t})
$$

que representa nosso conhecimento sobre o estado $x_t$ dadas todas as observacoes ate o tempo $t$.

### 1.3 Recursao de Bayes Otima

A distribuicao de filtragem satisfaz a **recursao de Bayes otima** em dois passos:

**Passo de Predicao**:

$$
p(x_t | y_{1:t-1}) = \int f(x_t | x_{t-1}) \, p(x_{t-1} | y_{1:t-1}) \, dx_{t-1}
$$

**Passo de Atualizacao (Bayes)**:

$$
p(x_t | y_{1:t}) = \frac{g(y_t | x_t) \, p(x_t | y_{1:t-1})}{p(y_t | y_{1:t-1})}
$$

onde a constante de normalizacao e a **predictive likelihood**:

$$
p(y_t | y_{1:t-1}) = \int g(y_t | x_t) \, p(x_t | y_{1:t-1}) \, dx_t
$$

!!! warning "Intratabilidade"
    A recursao de Bayes otima e **analiticamente intratavel** para a maioria dos modelos nao-lineares e/ou nao-Gaussianos. As excecoes notaveis sao:

    - **Modelos lineares-Gaussianos**: Filtro de Kalman (solucao exata)
    - **Espacos de estado finitos**: Forward algorithm para HMMs discretos
    - **Modelos conjugados**: Alguns casos especiais com solucao fechada

    Para todos os demais casos, necessitamos de **aproximacoes** — e aqui entram os particle filters.

---

## 2. Bootstrap Particle Filter como Aproximacao

### 2.1 Algoritmo como Aproximacao Monte Carlo

O **Bootstrap Particle Filter (BPF)** (Gordon, Salmond & Smith, 1993) aproxima a distribuicao de filtragem por uma medida empirica ponderada:

$$
\hat{\pi}_t^N(dx_t) = \sum_{i=1}^{N} W_t^{(i)} \, \delta_{x_t^{(i)}}(dx_t)
$$

onde $\{x_t^{(i)}, W_t^{(i)}\}_{i=1}^{N}$ sao as **particulas** e seus **pesos normalizados**.

!!! note "Algoritmo: Bootstrap Particle Filter"
    **Inicializacao** ($t=0$):

    1. Para $i = 1, \ldots, N$: amostrar $x_0^{(i)} \sim \mu(x_0)$
    2. Definir $W_0^{(i)} = 1/N$

    **Para** $t = 1, 2, \ldots$:

    1. **Resampling**: amostrar indices $a_{t-1}^{(i)} \sim \text{Categorical}(W_{t-1}^{(1:N)})$
    2. **Propagacao**: amostrar $x_t^{(i)} \sim f(x_t | x_{t-1}^{a_{t-1}^{(i)}})$
    3. **Pesos**: calcular $\tilde{w}_t^{(i)} = g(y_t | x_t^{(i)})$
    4. **Normalizacao**: $W_t^{(i)} = \tilde{w}_t^{(i)} / \sum_{j=1}^{N} \tilde{w}_t^{(j)}$

O BPF usa a **transicao do estado** $f(x_t|x_{t-1})$ como distribuicao proposal. O peso incremental simplifica para a likelihood $g(y_t | x_t)$.

!!! tip "Intuicao"
    O BPF implementa a recursao de Bayes otima aproximadamente:

    - **Predicao**: propagacao via $f$ corresponde a amostragem da preditiva
    - **Atualizacao**: ponderacao por $g(y_t|x_t)$ corresponde ao update de Bayes
    - **Resampling**: evita degeneracao, focando particulas em regioes de alta probabilidade

### 2.2 Convergencia

!!! abstract "Teorema: Convergencia do BPF (Crisan & Doucet, 2002)"
    Seja $\varphi : \mathcal{X} \to \mathbb{R}$ uma funcao limitada. Entao o estimador BPF e **consistente**:

    $$
    \hat{\pi}_t^N(\varphi) = \sum_{i=1}^{N} W_t^{(i)} \varphi(x_t^{(i)}) \xrightarrow{a.s.} \pi_t(\varphi) = \mathbb{E}_{\pi_t}[\varphi(x_t)]
    $$

    quando $N \to \infty$, para todo $t \geq 0$.

??? note "Sketch da Prova"
    A prova usa inducao em $t$ e propriedades de importance sampling:

    1. **Base** ($t=0$): IS classico com proposta $\mu$ e target $\mu$ (pesos uniformes). Convergencia por LGN.

    2. **Passo indutivo**: Suponha $\hat{\pi}_{t-1}^N \to \pi_{t-1}$.

        - **Resampling**: Gera amostras i.i.d. aproximadas de $\hat{\pi}_{t-1}^N$. Como $\hat{\pi}_{t-1}^N \to \pi_{t-1}$, as amostras reamostradas convergem para amostras de $\pi_{t-1}$.

        - **Propagacao**: Aplicar $f(x_t|x_{t-1})$ as amostras reamostradas gera amostras aproximadas da preditiva $p(x_t|y_{1:t-1})$.

        - **Ponderacao**: Os pesos $g(y_t|x_t)$ implementam IS com proposta $p(x_t|y_{1:t-1})$ e target $p(x_t|y_{1:t})$.

    3. Pela consistencia de IS e continuidade da composicao, $\hat{\pi}_t^N(\varphi) \to \pi_t(\varphi)$. $\blacksquare$

### 2.3 Bound de Erro

!!! abstract "Teorema: Bound de Erro $L^p$ (Del Moral, 2004)"
    Para qualquer $p \geq 1$ e funcao $\varphi$ limitada:

    $$
    \left(\mathbb{E}\left[\left|\hat{\pi}_t^N(\varphi) - \pi_t(\varphi)\right|^p\right]\right)^{1/p} \leq \frac{C_{t,p} \, \|\varphi\|_{\infty}}{\sqrt{N}}
    $$

    onde $C_{t,p}$ e uma constante que depende de $t$ e $p$ mas **nao de $N$**.

!!! example "Implicacao Pratica"
    Para obter um erro de $\epsilon$ na estimativa de $\mathbb{E}[\varphi(x_t)]$:

    - Necessitamos $N = O(1/\epsilon^2)$ particulas
    - **Independente da dimensao** $d_x$ do espaco de estados (para $t$ fixo)
    - A constante $C_t$ pode crescer com $t$, mas nao com $d_x$

    Isto contrasta com IS puro, onde o numero de amostras cresce exponencialmente com $d_x$.

---

## 3. Optimal Proposal Theory

### 3.1 Proposal que Minimiza Variancia

O BPF usa $f(x_t|x_{t-1})$ como proposta, mas esta nao e necessariamente a melhor escolha. A **optimal proposal** minimiza a variancia dos pesos condicionalmente aos ancestrais.

!!! abstract "Teorema: Optimal Proposal (Doucet, Godsill & Andrieu, 2000)"
    A distribuicao proposal que **minimiza a variancia** dos importance weights $\text{Var}[\tilde{w}_t | x_{t-1}]$ e:

    $$
    q_{\text{opt}}(x_t | x_{t-1}, y_t) = p(x_t | x_{t-1}, y_t) = \frac{g(y_t | x_t) \, f(x_t | x_{t-1})}{p(y_t | x_{t-1})}
    $$

    Sob esta proposta, os pesos dependem apenas do ancestral:

    $$
    \tilde{w}_t^{(i)} = p(y_t | x_{t-1}^{(i)}) = \int g(y_t | x_t) \, f(x_t | x_{t-1}^{(i)}) \, dx_t
    $$

??? note "Prova: Derivacao da Optimal Proposal"
    **Objetivo**: Minimizar $\text{Var}_{q}[\tilde{w}_t \mid x_{t-1}]$ sobre todas as proposals $q(x_t|x_{t-1}, y_t)$.

    O peso incremental geral e:

    $$
    \tilde{w}_t = \frac{g(y_t|x_t) f(x_t|x_{t-1})}{q(x_t|x_{t-1}, y_t)}
    $$

    **Passo 1**: Note que $\mathbb{E}_q[\tilde{w}_t | x_{t-1}]$ e constante em $q$ (igual a $p(y_t|x_{t-1})$), pois:

    $$
    \mathbb{E}_q[\tilde{w}_t | x_{t-1}] = \int \frac{g(y_t|x_t) f(x_t|x_{t-1})}{q(x_t|x_{t-1}, y_t)} q(x_t|x_{t-1}, y_t) dx_t = p(y_t|x_{t-1})
    $$

    **Passo 2**: Minimizar $\text{Var}[\tilde{w}_t]$ com media fixa equivale a minimizar $\mathbb{E}[\tilde{w}_t^2]$, que por Jensen e minimizado quando $\tilde{w}_t$ e constante em $x_t$.

    **Passo 3**: $\tilde{w}_t$ e constante em $x_t$ se e somente se:

    $$
    q(x_t|x_{t-1}, y_t) \propto g(y_t|x_t) f(x_t|x_{t-1})
    $$

    que e exatamente $p(x_t|x_{t-1}, y_t)$. O peso resultante e:

    $$
    \tilde{w}_t = \frac{g(y_t|x_t)f(x_t|x_{t-1})}{p(x_t|x_{t-1},y_t)} = p(y_t|x_{t-1})
    $$

    que **nao depende de $x_t$**, confirmando variancia zero condicional. $\blacksquare$

### 3.2 Intratabilidade da Optimal Proposal

!!! warning "Por que a Optimal Proposal e Intratavel"
    A optimal proposal $p(x_t|x_{t-1}, y_t)$ requer:

    1. **Amostrar** de $p(x_t|x_{t-1}, y_t) \propto g(y_t|x_t) f(x_t|x_{t-1})$ — requer saber a forma funcional do produto e ter um metodo de amostragem
    2. **Calcular** $p(y_t|x_{t-1}) = \int g(y_t|x_t) f(x_t|x_{t-1}) dx_t$ — integral geralmente intratavel

    Para modelos nao-lineares/nao-Gaussianos, ambas as operacoes sao tipicamente **analiticamene intratveis**.

### 3.3 Aproximacoes da Optimal Proposal

Varias estrategias aproximam a optimal proposal:

=== "Aproximacao Gaussiana Local (EKF/UKF)"

    Linearizar o modelo em torno da estimativa atual e usar o update do filtro de Kalman:

    $$
    q(x_t | x_{t-1}, y_t) = \mathcal{N}(x_t; \hat{m}_t, \hat{P}_t)
    $$

    onde $\hat{m}_t$ e $\hat{P}_t$ sao obtidos do Extended Kalman Filter (EKF) ou Unscented Kalman Filter (UKF) local. Este e o fundamento dos filtros **Unscented Particle Filter (UPF)** e **Extended Particle Filter**.

=== "Aproximacao de Laplace"

    Encontrar o modo $x_t^*$ da posterior local e aproximar por Gaussiana:

    $$
    x_t^* = \arg\max_{x_t} \left[\log g(y_t|x_t) + \log f(x_t|x_{t-1})\right]
    $$

    $$
    q(x_t|x_{t-1}, y_t) \approx \mathcal{N}\left(x_t^*,\; \left[-\nabla^2 \log p(x_t|x_{t-1}, y_t)\big|_{x_t^*}\right]^{-1}\right)
    $$

=== "Modelos Lineares-Gaussianos (Exata)"

    Para $x_t = A x_{t-1} + \eta_t$ e $y_t = C x_t + \epsilon_t$ com ruidos Gaussianos:

    $$
    p(x_t|x_{t-1}, y_t) = \mathcal{N}(x_t; m_t, \Sigma_t)
    $$

    com formulas fechadas via Kalman update. Neste caso, a optimal proposal e **exata** e o peso e $p(y_t|x_{t-1})$ analitico.

=== "Mixture Proposals"

    Combinar a prior com a likelihood via mistura:

    $$
    q(x_t|x_{t-1}, y_t) = \alpha \, f(x_t|x_{t-1}) + (1-\alpha) \, \hat{q}(x_t; y_t)
    $$

    onde $\hat{q}$ e uma aproximacao centrada na observacao.

!!! tip "Regra Pratica"
    A melhor aproximacao depende do modelo:

    | Situacao | Aproximacao Recomendada |
    |---|---|
    | Modelo fracamente nao-linear | EKF-proposal |
    | Nao-linearidade moderada | UKF-proposal |
    | Multimodalidade local | Mixture proposals |
    | Likelihood muito informativa | Laplace ou guided proposals |

---

## 4. Resultados Assintoticos

### 4.1 Lei dos Grandes Numeros para Particle Filters

!!! abstract "Teorema: LGN para Particle Filters (Del Moral & Guionnet, 1999)"
    Para qualquer funcao teste $\varphi$ com $\|\varphi\|_{\infty} < \infty$ e qualquer $t \geq 0$:

    $$
    \hat{\pi}_t^N(\varphi) \xrightarrow{a.s.} \pi_t(\varphi) \quad \text{quando } N \to \infty
    $$

    Mais geralmente, para a distribuicao conjunta de filtragem (path measure):

    $$
    \hat{p}^N(x_{0:t} | y_{1:t}) \xrightarrow{w} p(x_{0:t} | y_{1:t})
    $$

### 4.2 Teorema Central do Limite

!!! abstract "Teorema: CLT para Particle Filters (Chopin, 2004)"
    Sob condicoes de regularidade (kernels de transicao mixing, likelihood limitada), para funcoes $\varphi$ com $\pi_t(\varphi^2) < \infty$:

    $$
    \sqrt{N}\left(\hat{\pi}_t^N(\varphi) - \pi_t(\varphi)\right) \xrightarrow{d} \mathcal{N}\left(0, \sigma_t^2(\varphi)\right)
    $$

    onde a variancia assintotica tem a decomposicao aditiva:

    $$
    \sigma_t^2(\varphi) = \sum_{s=0}^{t} V_s\left(Q_{s+1,t}\varphi - \pi_t(\varphi)\right)
    $$

    Aqui $Q_{s,t}$ sao os operadores de Feynman-Kac e $V_s(\cdot)$ sao as variancias condicionais do resampling + IS no passo $s$.

??? note "Sketch da Prova"
    A prova segue a abordagem de **decomposicao de martingais** (Chopin, 2004):

    1. **Decomposicao**: O erro $\sqrt{N}(\hat{\pi}_t^N(\varphi) - \pi_t(\varphi))$ e escrito como soma de incrementos de martingal:

        $$
        \sqrt{N}(\hat{\pi}_t^N - \pi_t)(\varphi) = \sum_{s=0}^{t} D_s^N
        $$

        onde cada $D_s^N$ e um incremento de martingal associado ao passo $s$.

    2. **CLT para martingais**: Pelo CLT de arrays triangulares de martingais, basta verificar:
        - $\sum_s \mathbb{E}[(D_s^N)^2 | \mathcal{F}_{s-1}] \to \sigma_t^2$ (convergencia da variancia condicional)
        - Condicao de Lindeberg (caudas negligiveis)

    3. **Calculo da variancia**: Cada $\mathbb{E}[(D_s^N)^2 | \mathcal{F}_{s-1}]$ converge para $V_s(Q_{s+1,t}\varphi - \pi_t(\varphi))$, e a soma da a variancia total. $\blacksquare$

### 4.3 Variancia Assintotica e Sua Estimacao

A variancia assintotica $\sigma_t^2(\varphi)$ pode ser **estimada consistentemente** a partir das proprias particulas:

$$
\hat{\sigma}_t^{2,N}(\varphi) = \sum_{i=1}^{N} \left(W_t^{(i)}\right)^2 \left(\varphi(x_t^{(i)}) - \hat{\pi}_t^N(\varphi)\right)^2 \cdot N
$$

!!! tip "Estimacao via Genealogia"
    Uma estimativa mais precisa utiliza a **genealogia das particulas** (Lee & Whiteley, 2018):

    $$
    \hat{V}_t^N = \frac{1}{N} \sum_{i=1}^{N} \left(\sum_{j: a_j = i} W_t^{(j)} \varphi(x_t^{(j)}) - W_{t-1}^{(i)} \hat{\pi}_t^N(\varphi)\right)^2
    $$

    que captura tanto a variancia do IS quanto a do resampling.

### 4.4 Propagacao de Erro ao Longo do Tempo

!!! abstract "Teorema: Estabilidade Temporal (Del Moral & Guionnet, 2001)"
    Se o kernel de transicao $f$ satisfaz uma condicao de **mixing forte** — existem constantes $0 < \epsilon \leq 1$ e uma medida de referencia $\lambda$ tal que:

    $$
    \epsilon \, \lambda(A) \leq f(A | x) \leq \epsilon^{-1} \, \lambda(A) \quad \forall x, A
    $$

    entao a constante $C_t$ no bound de erro **permanece limitada uniformemente em $t$**:

    $$
    \sup_{t \geq 0} \; \mathbb{E}\left[|\hat{\pi}_t^N(\varphi) - \pi_t(\varphi)|^2\right] \leq \frac{C \|\varphi\|_\infty^2}{N}
    $$

    com $C$ independente de $t$.

??? note "Intuicao da Prova"
    A estabilidade temporal vem do **esquecimento exponencial** do filtro otimo:

    1. O filtro Bayesiano $\pi_t$ depende exponencialmente menos de $\pi_0$ a medida que $t$ cresce
    2. Informacao nova (observacoes) "renova" a distribuicao a cada passo
    3. Erros de aproximacao do passado sao **amortecidos** pela dinamica mixing
    4. Formalmente: $\|\pi_t^{\mu} - \pi_t^{\nu}\|_{TV} \leq C \rho^t$ para $\rho < 1$, onde $\pi_t^{\mu}$ e $\pi_t^{\nu}$ sao filtros com priors diferentes $\mu, \nu$

    O resampling impede acumulacao de erro, e o mixing garante que erros do passado sao esquecidos. $\blacksquare$

!!! warning "Quando a Estabilidade Falha"
    A estabilidade temporal pode falhar quando:

    - O kernel de transicao tem mixing fraco (estados quase-absorventes)
    - A likelihood e muito concentrada (observacoes muito informativas relativas a dinamica)
    - O espaco de estados nao e compacto sem condicoes de momento
    - Modelos com regimes de alta persistencia

    Nestes casos, o numero de particulas necessario pode crescer com $T$, e tecnicas como **tempering** ou **block particle filters** podem ser necessarias.

!!! example "Comparacao: Com e Sem Estabilidade"
    | Propriedade | Mixing Forte | Mixing Fraco |
    |---|---|---|
    | $C_t$ | Limitado ($C_t \leq C$) | Pode crescer: $C_t \sim \rho^t$ |
    | $N$ necessario | Fixo em $T$ | Cresce com $T$ |
    | Erro em $T=1000$ | $O(1/\sqrt{N})$ | Pode ser $O(T/\sqrt{N})$ ou pior |
    | Exemplo tipico | Modelos ergo­dicos, SV | Modelos com regime persistente |

---

## Referencias

| Referencia | Contribuicao |
|---|---|
| Gordon, Salmond & Smith (1993). *Novel approach to nonlinear/non-Gaussian Bayesian state estimation* | Bootstrap particle filter original |
| Doucet, Godsill & Andrieu (2000). *On sequential Monte Carlo sampling methods for Bayesian filtering* | Optimal proposal e framework geral |
| Del Moral (2004). *Feynman-Kac Formulae* | Teoria rigorosa: convergencia, CLT, bounds |
| Crisan & Doucet (2002). *A survey of convergence results on particle filtering* | Survey de resultados de convergencia |
| Chopin (2004). *Central limit theorem for sequential Monte Carlo* | CLT para particle filters |
| Del Moral & Guionnet (2001). *On the stability of interacting processes* | Estabilidade temporal |
| Lee & Whiteley (2018). *Variance estimation in the particle filter* | Estimacao de variancia via genealogia |
| Pitt & Shephard (1999). *Filtering via simulation* | Auxiliary particle filter e proposals melhoradas |
| Cappe, Moulines & Ryden (2005). *Inference in Hidden Markov Models* | Referencia geral sobre HMMs |
| Douc & Moulines (2008). *Limit theorems for weighted samples* | Resultados assintoticos para IS ponderado |
