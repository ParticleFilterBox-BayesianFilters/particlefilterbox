---
title: Particle Smoothing - Teoria
description: Teoria de particle smoothing - problema de smoothing, decomposicao forward-backward, FFBSm, FFBSi, Two-Filter Smoother e Fixed-Lag Smoothing.
---

# Particle Smoothing: Teoria

Esta pagina desenvolve a teoria de **particle smoothing**, que estende os metodos de filtragem para estimar distribuicoes de estados passados condicionadas em **todas** as observacoes, incluindo as futuras. Cobrem-se os algoritmos FFBSm, FFBSi, Two-Filter Smoother e Fixed-Lag Smoothing.

---

## 1. Problema de Smoothing

### 1.1 Definicao

Enquanto **filtragem** calcula $p(x_t \mid y_{1:t})$ usando observacoes ate o tempo $t$, **smoothing** incorpora observacoes futuras para obter estimativas mais precisas.

!!! info "Tres Problemas de Smoothing"
    | Problema | Distribuicao | Descricao |
    |---|---|---|
    | **Joint smoothing** | $p(x_{0:T} \mid y_{1:T})$ | Distribuicao sobre trajetorias completas |
    | **Marginal smoothing** | $p(x_t \mid y_{1:T})$ para $t < T$ | Distribuicao marginal de um unico tempo |
    | **Fixed-interval** | $p(x_t \mid y_{1:T})$ para todo $t$ | Todas as marginais simultaneamente |

### 1.2 Relacao com Filtragem

A relacao entre filtragem e smoothing e dada pela incorporacao de **informacao futura**:

$$
p(x_t \mid y_{1:T}) \neq p(x_t \mid y_{1:t}) \quad \text{para } t < T
$$

De fato, o smoothing sempre refina a filtragem:

$$
\text{Var}[x_t \mid y_{1:T}] \leq \text{Var}[x_t \mid y_{1:t}]
$$

!!! tip "Intuicao"
    O filtro "olha para tras" — usa apenas informacao passada. O smoother "olha para ambos os lados" — incorpora observacoes futuras $y_{t+1:T}$ para corrigir as estimativas. Isso e especialmente util para:

    - **Estimacao de parametros** via EM ou gradient methods
    - **Analise retrospectiva** de series temporais
    - **Inicializacao** de algoritmos MCMC
    - **Deteccao de pontos de mudanca** apos observar dados completos

### 1.3 Motivacao Bayesiana

A distribuicao de smoothing **marginal** pode ser escrita como:

$$
p(x_t \mid y_{1:T}) = \int p(x_t, x_{t+1:T} \mid y_{1:T}) \, dx_{t+1:T}
$$

ou, usando a propriedade de Markov do modelo de estado-espaco:

$$
p(x_t \mid y_{1:T}) = p(x_t \mid y_{1:t}) \int \frac{f(x_{t+1} \mid x_t)}{p(x_{t+1} \mid y_{1:t})} \, p(x_{t+1} \mid y_{1:T}) \, dx_{t+1}
$$

Esta ultima forma e a base da **backward recursion** usada nos algoritmos de smoothing.

---

## 2. Decomposicao Forward-Backward

### 2.1 Resultado Fundamental

!!! abstract "Teorema: Decomposicao Forward-Backward"
    A distribuicao de smoothing marginal admite a decomposicao:

    $$
    p(x_t \mid y_{1:T}) = p(x_t \mid y_{1:t}) \cdot \frac{p(y_{t+1:T} \mid x_t)}{p(y_{t+1:T} \mid y_{1:t})}
    $$

    onde:

    - $p(x_t \mid y_{1:t})$ e a **distribuicao de filtragem** (forward pass)
    - $p(y_{t+1:T} \mid x_t)$ e a **backward likelihood** (informacao futura)
    - $p(y_{t+1:T} \mid y_{1:t})$ e a constante de normalizacao

??? note "Sketch da Prova"
    Aplicamos Bayes com respeito a $y_{t+1:T}$:

    $$
    p(x_t \mid y_{1:T}) = p(x_t \mid y_{1:t}, y_{t+1:T}) = \frac{p(y_{t+1:T} \mid x_t, y_{1:t}) \, p(x_t \mid y_{1:t})}{p(y_{t+1:T} \mid y_{1:t})}
    $$

    Pela propriedade de Markov do HMM: dado $x_t$, as observacoes futuras $y_{t+1:T}$ sao condicionalmente independentes de $y_{1:t}$:

    $$
    p(y_{t+1:T} \mid x_t, y_{1:t}) = p(y_{t+1:T} \mid x_t)
    $$

    Substituindo, obtemos o resultado. $\blacksquare$

### 2.2 Forma Recursiva (Backward Kernel)

Uma forma alternativa e mais util computacionalmente usa o **backward kernel**:

$$
p(x_t \mid y_{1:T}) = \int p(x_t \mid x_{t+1}, y_{1:t}) \, p(x_{t+1} \mid y_{1:T}) \, dx_{t+1}
$$

onde o backward kernel e:

$$
p(x_t \mid x_{t+1}, y_{1:t}) = \frac{f(x_{t+1} \mid x_t) \, p(x_t \mid y_{1:t})}{p(x_{t+1} \mid y_{1:t})}
$$

!!! info "Significado dos Termos"
    | Termo | Nome | Fonte |
    |---|---|---|
    | $p(x_t \mid y_{1:t})$ | Filtering distribution | Forward pass (particle filter) |
    | $f(x_{t+1} \mid x_t)$ | Transition density | Modelo de estado |
    | $p(x_{t+1} \mid y_{1:t})$ | Predictive distribution | Predicao do filtro |
    | $p(x_{t+1} \mid y_{1:T})$ | Smoothing distribution ($t{+}1$) | Recursao backward |

### 2.3 Algoritmo Generico Forward-Backward

A estrutura geral de qualquer particle smoother e:

1. **Forward pass**: Executar particle filter para obter $\{x_t^{(i)}, W_t^{(i)}\}_{i=1}^{N}$ para $t = 0, \ldots, T$
2. **Backward pass**: Combinar a saida do filtro com informacao futura para obter pesos ou trajetorias de smoothing

A diferenca entre os algoritmos (FFBSm, FFBSi, Two-Filter) reside na implementacao do backward pass.

---

## 3. Forward Filtering Backward Smoothing (FFBSm)

### 3.1 Algoritmo

O **FFBSm** (Doucet, Godsill & Andrieu, 2000) computa os pesos de smoothing diretamente pela backward recursion.

!!! note "Algoritmo: FFBSm"
    **Input:** Particulas de filtragem $\{x_t^{(i)}, W_t^{(i)}\}_{i=1}^{N}$ para $t = 0, \ldots, T$.

    **Inicializacao:** Para $t = T$, os pesos de smoothing sao iguais aos de filtragem:

    $$
    W_{T|T}^{(i)} = W_T^{(i)}, \quad i = 1, \ldots, N
    $$

    **Backward recursion:** Para $t = T-1, T-2, \ldots, 0$:

    $$
    W_{t|T}^{(i)} = \sum_{j=1}^{N} W_{t+1|T}^{(j)} \cdot \frac{W_t^{(i)} \, f(x_{t+1}^{(j)} \mid x_t^{(i)})}{\sum_{k=1}^{N} W_t^{(k)} \, f(x_{t+1}^{(j)} \mid x_t^{(k)})}
    $$

    **Output:** Estimativa smoothed:

    $$
    \mathbb{E}[\varphi(x_t) \mid y_{1:T}] \approx \sum_{i=1}^{N} W_{t|T}^{(i)} \, \varphi(x_t^{(i)})
    $$

### 3.2 Derivacao dos Pesos

Os pesos de smoothing sao derivados da decomposicao backward. Definindo:

$$
W_{t|T}^{(i)} \propto p(x_t = x_t^{(i)} \mid y_{1:T})
$$

e usando a backward recursion:

$$
p(x_t^{(i)} \mid y_{1:T}) = W_t^{(i)} \sum_{j=1}^{N} W_{t+1|T}^{(j)} \cdot \frac{f(x_{t+1}^{(j)} \mid x_t^{(i)})}{\sum_{k=1}^{N} W_t^{(k)} \, f(x_{t+1}^{(j)} \mid x_t^{(k)})}
$$

### 3.3 Complexidade

!!! warning "Complexidade Quadratica"
    A complexidade do FFBSm e **$O(T \cdot N^2)$**, pois para cada par $(i, j)$ no backward pass, devemos avaliar a transition density $f(x_{t+1}^{(j)} \mid x_t^{(i)})$.

    Para $N$ e $T$ grandes, isso pode ser proibitivo. Estrategias de reducao incluem:

    - **Rejection-based FFBSm** (Douc, Garivier, Moulines & Olsson, 2011): custo esperado $O(TN)$ sob condicoes de mixing
    - **Tree-based approximation**: kd-trees para busca de vizinhos proximos, custo $O(TN \log N)$
    - **Block FFBSm**: processar blocos de tempos em paralelo

### 3.4 Convergencia

!!! abstract "Teorema: CLT para FFBSm"
    Sob condicoes de regularidade (Del Moral, Doucet & Singh, 2010), o estimador FFBSm satisfaz:

    $$
    \sqrt{N}\left(\sum_{i=1}^{N} W_{t|T}^{(i)} \varphi(x_t^{(i)}) - \mathbb{E}[\varphi(x_t) \mid y_{1:T}]\right) \xrightarrow{d} \mathcal{N}\left(0, \sigma_{t|T}^2\right)
    $$

    onde $\sigma_{t|T}^2 < \infty$ e nao depende de $T$ (sob mixing).

??? note "Sketch da Prova"
    A prova combina o CLT para o filtro (forward pass) com um argumento de estabilidade para a backward recursion.

    **Passo 1:** O forward filter produz:
    $$
    \sqrt{N}\left(\hat{\pi}_t^N(\varphi) - \pi_t(\varphi)\right) \xrightarrow{d} \mathcal{N}(0, \sigma_t^2)
    $$

    **Passo 2:** A backward recursion e uma transformacao continua dos pesos de filtragem. Pelo delta method, erros $O(1/\sqrt{N})$ no forward pass propagam como $O(1/\sqrt{N})$ no backward pass.

    **Passo 3:** A estabilidade da variancia assintotica com respeito a $T$ segue da propriedade de **forgetting** da cadeia de Markov: a influencia de $x_0$ nas observacoes $y_T$ decai exponencialmente. $\blacksquare$

---

## 4. Forward Filtering Backward Simulation (FFBSi)

### 4.1 Idea Fundamental

Enquanto o FFBSm computa **pesos** de smoothing, o **FFBSi** (Godsill, Doucet & West, 2004) gera **trajetorias** completas $x_{0:T}$ da distribuicao de smoothing via **backward simulation**.

!!! tip "Intuicao"
    O FFBSi amostra trajetorias "de tras para frente": comeca com uma particula em $t = T$ e, para cada passo anterior, seleciona uma particula do filtro com probabilidade proporcional a transicao. Isso gera amostras de $p(x_{0:T} \mid y_{1:T})$.

### 4.2 Algoritmo

!!! note "Algoritmo: FFBSi"
    **Input:** Particulas de filtragem $\{x_t^{(i)}, W_t^{(i)}\}_{i=1}^{N}$ para $t = 0, \ldots, T$.

    **Para cada trajetoria** $m = 1, \ldots, M$:

    **Inicializacao:** Amostrar $\tilde{x}_T^{(m)} = x_T^{(B_T^{(m)})}$ onde $B_T^{(m)} \sim \text{Categorical}(W_T^{(1)}, \ldots, W_T^{(N)})$

    **Backward simulation:** Para $t = T-1, T-2, \ldots, 0$:

    Amostrar $B_t^{(m)} \sim \text{Categorical}(\tilde{W}_{t|t+1}^{(1)}, \ldots, \tilde{W}_{t|t+1}^{(N)})$ onde:

    $$
    \tilde{W}_{t|t+1}^{(i)} \propto W_t^{(i)} \cdot f(\tilde{x}_{t+1}^{(m)} \mid x_t^{(i)})
    $$

    Definir $\tilde{x}_t^{(m)} = x_t^{(B_t^{(m)})}$

    **Output:** $M$ trajetorias smoothed $\{\tilde{x}_{0:T}^{(m)}\}_{m=1}^{M}$

### 4.3 Interpretacao como Importance Sampling

O FFBSi pode ser visto como importance sampling sobre trajetorias. A distribuicao proposal e:

$$
q(x_{0:T}) = W_T^{(B_T)} \prod_{t=0}^{T-1} \tilde{W}_{t|t+1}^{(B_t)}
$$

e a target e a distribuicao de joint smoothing $p(x_{0:T} \mid y_{1:T})$. Os pesos resultantes sao uniformes (as trajetorias sao amostras aproximadas da target), com acuracia melhorando com $N$.

### 4.4 Complexidade

| Aspecto | FFBSm | FFBSi |
|---|---|---|
| Custo total | $O(T \cdot N^2)$ | $O(T \cdot M \cdot N)$ |
| Output | Pesos smoothed | $M$ trajetorias |
| Memoria | $O(T \cdot N)$ | $O(T \cdot N + T \cdot M)$ |
| Uso tipico | Expectativas marginais | Trajetorias, joint functionals |

### 4.5 Relacao com Gibbs Sampling

O FFBSi e componente fundamental de algoritmos **Particle MCMC**, especialmente o **Particle Gibbs** (Andrieu, Doucet & Holenstein, 2010):

$$
\text{Particle Gibbs:} \quad
\begin{cases}
\theta^{(k+1)} \sim p(\theta \mid x_{0:T}^{(k)}, y_{1:T}) \\
x_{0:T}^{(k+1)} \sim \text{FFBSi}\left(\cdot \mid \theta^{(k+1)}, y_{1:T}\right)
\end{cases}
$$

!!! info "Papel no Particle Gibbs"
    - O FFBSi gera **uma trajetoria** da joint smoothing distribution
    - Esta trajetoria condiciona a atualizacao dos parametros $\theta$
    - A alternancia constitui um Gibbs sampler que converge para $p(\theta, x_{0:T} \mid y_{1:T})$
    - A variante **conditional** (cPF-BS) garante validade teorica mantendo uma trajetoria de referencia

---

## 5. Two-Filter Smoother

### 5.1 Motivacao

O **Two-Filter Smoother** (Briers, Doucet & Maskell, 2010; Fearnhead, Wyncoll & Tawn, 2010) aborda o smoothing usando dois filtros independentes executados em direcoes opostas:

- **Forward filter**: $p(x_t \mid y_{1:t})$ — o filtro padrao
- **Backward information filter**: $p(y_{t+1:T} \mid x_t)$ — likelihood das observacoes futuras

### 5.2 Formulacao Teorica

!!! abstract "Teorema: Two-Filter Decomposition"
    A distribuicao de smoothing pode ser expressa como:

    $$
    p(x_t \mid y_{1:T}) \propto p(x_t \mid y_{1:t}) \cdot \gamma_t(x_t)
    $$

    onde $\gamma_t(x_t) \propto p(y_{t+1:T} \mid x_t)$ e a **backward information** obtida do filtro backward.

??? note "Sketch da Prova"
    Da decomposicao forward-backward (Secao 2.1):

    $$
    p(x_t \mid y_{1:T}) = \frac{p(x_t \mid y_{1:t}) \cdot p(y_{t+1:T} \mid x_t)}{p(y_{t+1:T} \mid y_{1:t})}
    $$

    Definindo $\gamma_t(x_t) = p(y_{t+1:T} \mid x_t)$ e notando que o denominador e constante em $x_t$:

    $$
    p(x_t \mid y_{1:T}) \propto p(x_t \mid y_{1:t}) \cdot \gamma_t(x_t)
    $$

    $\blacksquare$

### 5.3 Backward Information Filter

O **backward information filter** computa $\gamma_t(x_t) = p(y_{t+1:T} \mid x_t)$ recursivamente:

$$
\gamma_t(x_t) = \int g(y_{t+1} \mid x_{t+1}) \, \gamma_{t+1}(x_{t+1}) \, f(x_{t+1} \mid x_t) \, dx_{t+1}
$$

com inicializacao $\gamma_T(x_T) = 1$ (ou $g(y_T \mid x_T)$ dependendo da convencao).

Na pratica, $\gamma_t$ e representada por particulas $\{x_t^{B,(j)}, \gamma_t^{(j)}\}_{j=1}^{N_B}$ obtidas executando um particle filter no sentido **reverso** do tempo.

### 5.4 Combinacao dos Filtros

Os pesos de smoothing sao obtidos combinando os dois conjuntos de particulas:

$$
W_{t|T}^{(i)} \propto W_t^{(i)} \sum_{j=1}^{N_B} \gamma_t^{(j)} \, K\left(x_t^{(i)}, x_t^{B,(j)}\right)
$$

onde $K$ e um kernel de integracao (em casos especiais, pode ser uma avaliacao pontual).

!!! warning "Dificuldade do Filtro Backward"
    O backward information filter opera em **espaco de observacoes futuras**, nao em espaco de estados. Para modelos nao-lineares gerais, construir um filtro backward eficiente e nao-trivial. Abordagens incluem:

    - **Artificial backward dynamics**: definir um kernel backward auxiliar
    - **Information form**: trabalhar no espaco de informacao em vez de covariancia
    - **Twisted targets**: usar targets modificadas que incorporam informacao futura (Whiteley, 2010)

### 5.5 Vantagens Computacionais

!!! tip "Paralelizacao"
    O principal atrativo do Two-Filter Smoother e a **paralelizabilidade**:

    - Forward e backward filters sao **independentes** — podem ser executados em paralelo
    - A combinacao no passo final e $O(N_F \cdot N_B)$ para cada $t$, mas os $T$ tempos sao tambem independentes
    - Custo total: $O(T \cdot N_F + T \cdot N_B + T \cdot N_F \cdot N_B)$

    Em arquiteturas multi-core, o Two-Filter pode ser significativamente mais rapido que o FFBSm sequencial.

| Aspecto | FFBSm | FFBSi | Two-Filter |
|---|---|---|---|
| Custo | $O(TN^2)$ | $O(TMN)$ | $O(TN_F N_B)$ |
| Paralelizavel | Nao (backward sequencial) | Nao (backward sequencial) | Sim (forward $\parallel$ backward) |
| Output | Pesos marginais | Trajetorias | Pesos marginais |
| Dificuldade | Baixa | Baixa | Alta (filtro backward) |

---

## 6. Fixed-Lag Smoothing

### 6.1 Motivacao: Smoothing Online

Os metodos anteriores (FFBSm, FFBSi, Two-Filter) sao **offline** — requerem todas as observacoes $y_{1:T}$ antes de iniciar o backward pass. Em aplicacoes **online**, queremos estimativas smoothed com atraso fixo.

### 6.2 Aproximacao Fixed-Lag

!!! info "Definicao: Fixed-Lag Smoother"
    O **Fixed-Lag Smoother** com lag $L$ aproxima:

    $$
    p(x_t \mid y_{1:t+L}) \approx p(x_t \mid y_{1:T}) \quad \text{para } L \text{ suficientemente grande}
    $$

    A ideia e que observacoes muito distantes no futuro contribuem pouco para a estimativa de $x_t$ (decaimento exponencial da informacao sob mixing).

### 6.3 Algoritmo

!!! note "Algoritmo: Fixed-Lag Particle Smoother"
    **Input:** Lag $L$, observacoes sequenciais $y_1, y_2, \ldots$

    No tempo $t + L$:

    1. Executar particle filter ate $t + L$, produzindo $\{x_{t:t+L}^{(i)}, W_{t+L}^{(i)}\}_{i=1}^{N}$
    2. A estimativa smoothed de $x_t$ e:

    $$
    \hat{\mathbb{E}}[x_t \mid y_{1:t+L}] = \sum_{i=1}^{N} W_{t+L}^{(i)} \, x_t^{(i)}
    $$

    onde $x_t^{(i)}$ e o **ancestral** da particula $i$ no tempo $t$ (rastreado pelo historico de resampling).

### 6.4 Bias do Fixed-Lag

!!! abstract "Teorema: Bias do Fixed-Lag Smoother"
    Seja $\Delta_L(t) = p(x_t \mid y_{1:t+L}) - p(x_t \mid y_{1:T})$ o erro de aproximacao. Sob condicoes de mixing exponencial do HMM:

    $$
    \|\Delta_L(t)\|_{\text{TV}} \leq C \, \rho^L
    $$

    onde $0 < \rho < 1$ e a taxa de mixing do modelo e $C > 0$ e uma constante. Portanto, o bias decai **exponencialmente** com o lag $L$.

??? note "Sketch da Prova"
    **Passo 1:** Pela propriedade de Markov, a informacao que $y_{t+L+1:T}$ fornece sobre $x_t$ e mediada por $x_{t+L}$:

    $$
    p(x_t \mid y_{1:T}) = \int p(x_t \mid x_{t+L}, y_{1:t+L}) \, p(x_{t+L} \mid y_{1:T}) \, dx_{t+L}
    $$

    **Passo 2:** Sob mixing exponencial, a dependencia de $x_t$ em $x_{t+L}$ decai como $\rho^L$:

    $$
    \|p(x_t \mid x_{t+L} = a, y_{1:t+L}) - p(x_t \mid x_{t+L} = b, y_{1:t+L})\|_{\text{TV}} \leq C \, \rho^L
    $$

    para quaisquer $a, b$.

    **Passo 3:** Portanto, a marginal $p(x_t \mid y_{1:T})$ nao pode diferir muito de $p(x_t \mid y_{1:t+L})$, pois a informacao adicional $y_{t+L+1:T}$ so afeta $x_t$ via $x_{t+L}$, cuja influencia ja decaiu. $\blacksquare$

### 6.5 Escolha do Lag $L$

A escolha de $L$ envolve um trade-off entre **acuracia** e **custo/latencia**:

| $L$ | Bias | Path degeneracy | Latencia | Uso tipico |
|---|---|---|---|---|
| Pequeno ($1$–$5$) | Alto | Baixa | Minima | Tracking em tempo real |
| Moderado ($10$–$50$) | Baixo | Moderada | Aceitavel | Financas, meteorologia |
| Grande ($> 50$) | Negligivel | Alta | Significativa | Analise offline aproximada |

!!! warning "Degeneracao de Trajetorias"
    Um problema fundamental do fixed-lag smoother e a **degeneracao de trajetorias** (path degeneracy): apos $L$ passos de resampling, todas as particulas tipicamente compartilham o **mesmo ancestral** em $t$. Isso significa que a estimativa smoothed colapsa para um unico ponto.

    Formalmente, o numero de **ancestrais distintos** no tempo $t$ satisfaz:

    $$
    |\{x_t^{(i)} : i = 1, \ldots, N\}| \xrightarrow{L \to \infty} 1 \quad \text{para } N \text{ fixo}
    $$

    Para mitigar, usam-se:

    - **$N$ grande** relativo a $L$: garantir diversidade suficiente
    - **Resampling adaptativo**: resampling apenas quando ESS cai abaixo de um limiar
    - **Algoritmos rejuvenation**: e.g., MCMC moves nas particulas ancestrais

### 6.6 Resultados de Mixing e Criterios Praticos

A taxa de mixing $\rho$ determina o $L$ necessario. Pode ser estimada por:

**Criterio 1: Autocorrelacao do estado**

$$
L^* \approx \min\{L : |\text{Cor}(x_t, x_{t+L} \mid y)| < \epsilon\}
$$

**Criterio 2: Log-likelihood incremental**

$$
L^* \approx \min\left\{L : \left|\frac{\partial}{\partial L} \log p(y_{1:t+L} \mid x_t)\right| < \delta\right\}
$$

**Criterio 3: ESS-based**

Monitorar o ESS das particulas ancestrais no tempo $t$. Se $\text{ESS}_t^{(L)} < N_{\text{thr}}$, entao $L$ ja e grande demais para o $N$ disponivel.

!!! tip "Regra Pratica"
    Para a maioria dos modelos economicos e financeiros, $L \in [10, 30]$ oferece um bom compromisso. Modelos com alta persistencia (e.g., componente tendencial, unit root proximo) requerem $L$ maior. Modelos com switching rapido tipicamente precisam apenas de $L \sim 5$–$10$.

---

## Referencias

- **Doucet, A., Godsill, S. J., & Andrieu, C.** (2000). On Sequential Monte Carlo Sampling Methods for Bayesian Filtering. *Statistics and Computing*, 10(3), 197–208.
- **Godsill, S. J., Doucet, A., & West, M.** (2004). Monte Carlo Smoothing for Nonlinear Time Series. *Journal of the American Statistical Association*, 99(465), 156–168.
- **Briers, M., Doucet, A., & Maskell, S.** (2010). Smoothing Algorithms for State-Space Models. *Annals of the Institute of Statistical Mathematics*, 62(1), 61–89.
- **Fearnhead, P., Wyncoll, D., & Tawn, J.** (2010). A Sequential Smoothing Algorithm with Linear Computational Cost. *Biometrika*, 97(2), 447–464.
- **Del Moral, P., Doucet, A., & Singh, S. S.** (2010). Forward Smoothing Using Sequential Monte Carlo. *arXiv preprint arXiv:1012.5390*.
- **Douc, R., Garivier, A., Moulines, E., & Olsson, J.** (2011). Sequential Monte Carlo Smoothing for General State Space Hidden Markov Models. *Annals of Applied Probability*, 21(6), 2226–2252.
- **Lindsten, F. & Schon, T. B.** (2013). Backward Simulation Methods for Monte Carlo Statistical Inference. *Foundations and Trends in Machine Learning*, 6(1), 1–143.
- **Andrieu, C., Doucet, A., & Holenstein, R.** (2010). Particle Markov Chain Monte Carlo Methods. *Journal of the Royal Statistical Society: Series B*, 72(3), 269–342.
- **Kitagawa, G.** (1996). Monte Carlo Filter and Smoother for Non-Gaussian Nonlinear State Space Models. *Journal of Computational and Graphical Statistics*, 5(1), 1–25.
- **Whiteley, N.** (2010). Discussion of "Particle Markov Chain Monte Carlo Methods". *Journal of the Royal Statistical Society: Series B*, 72(3), 306–307.
- **Cappe, O., Moulines, E., & Ryden, T.** (2005). *Inference in Hidden Markov Models*. Springer.
