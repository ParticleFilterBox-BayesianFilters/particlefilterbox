---
title: Teoria de Resampling
description: Framework unificado de resampling para particle filters - motivacao formal, algoritmos, propriedades teoricas e resampling adaptativo.
---

# Teoria de Resampling

Esta pagina apresenta a teoria de **resampling** (reamostragem) para particle filters e SMC, incluindo motivacao formal, framework unificado, analise de algoritmos e estrategias adaptativas.

---

## 1. Motivacao Formal

### 1.1 Weight Degeneracy

O problema central que resampling resolve e a **degeneracao dos pesos**: sem resampling, apos varios passos do filtro, praticamente toda a massa de probabilidade concentra-se em uma unica particula.

!!! abstract "Teorema: Degeneracao do ESS (Kong et al., 1994)"
    Considere SIS sem resampling com $N$ particulas. Apos $T$ passos, o Effective Sample Size satisfaz:

    $$
    \frac{\text{ESS}_T}{N} \xrightarrow{p} 0 \quad \text{quando } T \to \infty
    $$

    para $N$ fixo. Mais precisamente, se os pesos incrementais tem variancia $\sigma^2 > 0$:

    $$
    \text{ESS}_T \approx \frac{N}{(1 + \sigma^2)^T}
    $$

    que decai **exponencialmente** em $T$.

??? note "Prova: ESS Converge para 1"
    **Proposicao**: Sem resampling, $\text{ESS}_T \to 1$ quase certamente quando $T \to \infty$.

    **Prova**:

    1. Os pesos nao-normalizados apos $T$ passos sao $\tilde{w}_T^{(i)} = \prod_{t=1}^{T} \alpha_t^{(i)}$, onde $\alpha_t^{(i)}$ sao os pesos incrementais.

    2. O peso normalizado da particula $i$ e:

        $$
        W_T^{(i)} = \frac{\tilde{w}_T^{(i)}}{\sum_{j=1}^{N} \tilde{w}_T^{(j)}}
        $$

    3. Pela Lei dos Grandes Numeros aplicada a $\log \tilde{w}_T^{(i)} = \sum_{t=1}^{T} \log \alpha_t^{(i)}$, todas as particulas tem $\log$-pesos concentrados em torno de $T \cdot \mathbb{E}[\log \alpha]$.

    4. Porem, pela **desigualdade de Jensen**: $\mathbb{E}[\log \alpha] < \log \mathbb{E}[\alpha]$. A particula com maior desvio positivo domina exponencialmente.

    5. Formalmente, defina $i^* = \arg\max_i \tilde{w}_T^{(i)}$. Entao:

        $$
        W_T^{(i^*)} = \frac{1}{1 + \sum_{j \neq i^*} \exp\left(\sum_{t=1}^{T}(\log\alpha_t^{(j)} - \log\alpha_t^{(i^*)})\right)} \xrightarrow{a.s.} 1
        $$

    6. Portanto:

        $$
        \text{ESS}_T = \frac{1}{\sum_{i=1}^{N} (W_T^{(i)})^2} \xrightarrow{a.s.} 1 \quad \blacksquare
        $$

### 1.2 Resampling como Antidoto

!!! info "Papel do Resampling"
    Resampling **restaura a diversidade** das particulas ao:

    1. **Eliminar** particulas com peso baixo (pouco representativas)
    2. **Duplicar** particulas com peso alto (muito representativas)
    3. **Resetar** os pesos para $1/N$ (uniformes)

    Apos resampling, as particulas sao amostras i.i.d. aproximadas da distribuicao de filtragem empirica.

O efeito pode ser visualizado como uma "poda" da populacao: particulas irrelevantes sao removidas e particulas promissoras sao replicadas.

---

## 2. Framework Unificado

### 2.1 Resampling como Operador

Formalmente, resampling e um **operador estocastico** que transforma um conjunto ponderado em um conjunto uniformemente ponderado:

$$
\mathcal{R}: \left\{x^{(i)}, W^{(i)}\right\}_{i=1}^{N} \mapsto \left\{\tilde{x}^{(j)}, \frac{1}{N}\right\}_{j=1}^{N}
$$

onde cada $\tilde{x}^{(j)}$ pertence ao conjunto original $\{x^{(1)}, \ldots, x^{(N)}\}$.

Equivalentemente, resampling produz um vetor de **contagens** $\mathbf{N} = (N_1, \ldots, N_N)$ onde $N_i$ e o numero de copias da particula $i$ no novo conjunto:

$$
\sum_{i=1}^{N} N_i = N
$$

### 2.2 Propriedade Fundamental: Unbiasedness

!!! abstract "Teorema: Unbiasedness do Resampling"
    Um esquema de resampling e **valido** (unbiased) se:

    $$
    \mathbb{E}[N_i] = N \cdot W^{(i)} \quad \forall i = 1, \ldots, N
    $$

    Isto garante que o estimador apos resampling permanece consistente:

    $$
    \mathbb{E}\left[\frac{1}{N}\sum_{j=1}^{N} \varphi(\tilde{x}^{(j)})\right] = \sum_{i=1}^{N} W^{(i)} \varphi(x^{(i)})
    $$

??? note "Prova"
    $$
    \mathbb{E}\left[\frac{1}{N}\sum_{j=1}^{N}\varphi(\tilde{x}^{(j)})\right]
    = \frac{1}{N}\sum_{i=1}^{N}\mathbb{E}[N_i] \, \varphi(x^{(i)})
    = \frac{1}{N}\sum_{i=1}^{N} N W^{(i)} \varphi(x^{(i)})
    = \sum_{i=1}^{N} W^{(i)} \varphi(x^{(i)}) \quad \blacksquare
    $$

### 2.3 Variancia do Resampling

O custo do resampling e a **variancia adicional** que ele introduz. Para uma funcao teste $\varphi$:

$$
\text{Var}_{\mathcal{R}}\left[\frac{1}{N}\sum_{j=1}^{N}\varphi(\tilde{x}^{(j)})\right] = \frac{1}{N^2}\sum_{i=1}^{N}\text{Var}[N_i] \, \varphi(x^{(i)})^2 + \text{termos de covariancia}
$$

!!! abstract "Proposicao: Variancia do Resampling"
    Para um esquema de resampling com contagens $(N_1, \ldots, N_N)$:

    $$
    \text{Var}_{\mathcal{R}}\left[\frac{1}{N}\sum_j \varphi(\tilde{x}^{(j)})\right]
    = \frac{1}{N^2}\sum_{i,j} \text{Cov}[N_i, N_j] \, \varphi(x^{(i)}) \varphi(x^{(j)})
    $$

    A **variancia total** do estimador SMC e a soma da variancia de IS e da variancia de resampling. Minimizar a variancia do resampling e crucial para a eficiencia do filtro.

---

## 3. Algoritmos e Propriedades

### 3.1 Resampling Multinomial

O esquema mais simples: amostrar $N$ vezes independentemente da distribuicao categorica definida pelos pesos.

!!! note "Algoritmo: Multinomial Resampling"
    Para $j = 1, \ldots, N$:

    1. Amostrar $i_j \sim \text{Categorical}(W^{(1)}, \ldots, W^{(N)})$
    2. Definir $\tilde{x}^{(j)} = x^{(i_j)}$

    Equivalentemente: $(N_1, \ldots, N_N) \sim \text{Multinomial}(N; W^{(1)}, \ldots, W^{(N)})$

**Propriedades**:

| Propriedade | Valor |
|---|---|
| Unbiased | Sim: $\mathbb{E}[N_i] = NW^{(i)}$ |
| $\text{Var}[N_i]$ | $NW^{(i)}(1 - W^{(i)})$ |
| Complexidade | $O(N \log N)$ com busca binaria |
| Correlacao | $\text{Cov}[N_i, N_j] = -NW^{(i)}W^{(j)}$ para $i \neq j$ |

!!! warning "Variancia Maxima"
    Multinomial resampling tem a **maior variancia** entre os esquemas comuns. Para uma particula com peso $W^{(i)}$:

    $$
    \text{Var}_{\text{mult}}[N_i] = NW^{(i)}(1 - W^{(i)})
    $$

    Esta e a variancia maxima para qualquer esquema unbiased (entre os sem correlacao negativa).

### 3.2 Resampling Stratified

Stratified resampling reduz a variancia particionando o espaco amostral em $N$ estratos iguais.

!!! note "Algoritmo: Stratified Resampling (Kitagawa, 1996)"
    1. Para $j = 1, \ldots, N$: amostrar $U^{(j)} \sim \text{Uniform}\left(\frac{j-1}{N}, \frac{j}{N}\right)$
    2. Para cada $U^{(j)}$, encontrar $i$ tal que $\sum_{k=1}^{i-1} W^{(k)} < U^{(j)} \leq \sum_{k=1}^{i} W^{(k)}$
    3. Definir $\tilde{x}^{(j)} = x^{(i)}$

!!! abstract "Teorema: Variancia do Stratified Resampling (Douc & Cappe, 2005)"
    Para stratified resampling, a variancia das contagens satisfaz:

    $$
    \text{Var}_{\text{strat}}[N_i] \leq \min\left(NW^{(i)}(1 - W^{(i)}),\; NW^{(i)}\right)
    $$

    Portanto:

    $$
    \text{Var}_{\text{strat}}[N_i] \leq \text{Var}_{\text{mult}}[N_i]
    $$

    com igualdade apenas nos casos degenerados $W^{(i)} \in \{0, 1\}$.

??? note "Sketch da Prova"
    1. No stratified resampling, $N_i = \#\{j : U^{(j)} \in I_i\}$ onde $I_i = [\sum_{k<i}W^{(k)}, \sum_{k \leq i}W^{(k)})$.

    2. Cada estrato $((j-1)/N, j/N)$ contem no maximo um ponto por construcao.

    3. A particula $i$ e selecionada pelo estrato $j$ se $I_i \cap ((j-1)/N, j/N) \neq \emptyset$.

    4. Para os estratos que estao inteiramente contidos em $I_i$, a selecao e **deterministica** ($N_i$ inclui estes com certeza).

    5. A aleatoriedade vem apenas dos estratos nas **bordas** de $I_i$ (no maximo 2). Logo a variancia e limitada pela probabilidade de selecao nestes estratos, que e menor que a multinomial. $\blacksquare$

### 3.3 Resampling Systematic

Systematic resampling usa um **unico** numero aleatorio para gerar todas as amostras.

!!! note "Algoritmo: Systematic Resampling (Carpenter et al., 1999)"
    1. Amostrar $U \sim \text{Uniform}(0, 1/N)$
    2. Para $j = 1, \ldots, N$: definir $U^{(j)} = U + (j-1)/N$
    3. Para cada $U^{(j)}$, encontrar $i$ tal que $\sum_{k=1}^{i-1} W^{(k)} < U^{(j)} \leq \sum_{k=1}^{i} W^{(k)}$
    4. Definir $\tilde{x}^{(j)} = x^{(i)}$

**Propriedades**:

| Propriedade | Valor |
|---|---|
| Unbiased | Sim |
| Variancia | $\leq$ stratified $\leq$ multinomial |
| Complexidade | $O(N)$ — um unico passo |
| Aleatoriedade | Apenas 1 numero aleatorio |

!!! tip "Vantagem Pratica"
    Systematic resampling e o **mais usado na pratica** por combinar:

    - Complexidade $O(N)$ (o mais rapido)
    - Variancia muito baixa (frequentemente a menor)
    - Implementacao simples (uma unica passagem)
    - Reproducibilidade (1 seed controla tudo)

!!! warning "Limitacao Teorica"
    Systematic resampling nao e, estritamente, unbiased para todas as funcoes teste (apenas assintoticamente). A garantia formal e mais fraca que stratified em alguns casos patologicos, mas na pratica performa tao bem ou melhor.

### 3.4 Resampling Residual

Residual resampling combina uma parte **deterministica** com multinomial para o residuo.

!!! note "Algoritmo: Residual Resampling (Liu & Chen, 1998)"
    1. **Parte deterministica**: $N_i^{\text{det}} = \lfloor N \cdot W^{(i)} \rfloor$
    2. **Residuo**: $R = N - \sum_i N_i^{\text{det}}$ particulas restantes
    3. **Parte estocastica**: amostrar $R$ particulas via multinomial com pesos residuais:

        $$
        \bar{W}^{(i)} = \frac{N W^{(i)} - N_i^{\text{det}}}{R}
        $$

    4. **Total**: $N_i = N_i^{\text{det}} + N_i^{\text{res}}$

**Propriedades**:

$$
\text{Var}_{\text{res}}[N_i] = \frac{R}{N^2} \bar{W}^{(i)}(1 - \bar{W}^{(i)}) \leq \text{Var}_{\text{mult}}[N_i]
$$

A reducao de variancia vem da parte deterministica: se $NW^{(i)}$ e proximo de um inteiro, a contagem e quase certa.

### 3.5 Killing Resampling

Projetado para **SMC samplers** onde particulas vivem em espacos diferentes.

!!! note "Algoritmo: Killing Resampling"
    Para cada particula $i$:

    1. Com probabilidade $\min(1, NW^{(i)}/c)$: manter a particula (peso $= c/N$)
    2. Caso contrario: "matar" a particula (removida)
    3. Particulas sobreviventes sao duplicadas para restaurar $N$

    onde $c$ e uma constante de calibracao.

!!! info "Uso em SMC Samplers"
    Killing resampling e preferivel quando:

    - Particulas carregam estado computacional complexo (e.g., cadeias MCMC)
    - A duplicacao de particulas e indesejavel (introduz correlacao)
    - O custo de "resetar" uma particula e alto

### 3.6 Optimal Transport Resampling

Uma abordagem recente que minimiza a **distancia de transporte** entre a medida ponderada e a reamostrada.

!!! note "Optimal Transport Resampling (Reich, 2013; Corenflos et al., 2021)"
    Em vez de amostrar indices, resolve o problema de transporte otimo:

    $$
    \min_{T_{ij} \geq 0} \sum_{i,j} T_{ij} \|x^{(i)} - x^{(j)}\|^2
    $$

    sujeito a:

    $$
    \sum_j T_{ij} = W^{(i)}, \quad \sum_i T_{ij} = \frac{1}{N} \quad \forall i, j
    $$

    A matriz de transporte $T$ define como mover massa das particulas ponderadas para as uniformes.

!!! tip "Vantagens"
    - **Preserva estrutura espacial**: particulas reamostradas ficam proximas as originais
    - **Reduz sample impoverishment**: menor perda de diversidade
    - **Util em altas dimensoes**: onde resampling classico sofre mais

!!! warning "Custo Computacional"
    O custo e $O(N^2)$ ou $O(N^2 \log N)$ dependendo do solver OT, comparado com $O(N)$ do systematic. Algoritmos aproximados (Sinkhorn) reduzem para $O(N^2 / \epsilon)$.

### 3.7 Comparacao de Algoritmos

| Algoritmo | Complexidade | Variancia | Correlacao | Uso tipico |
|---|---|---|---|---|
| Multinomial | $O(N \log N)$ | Maxima | Nenhuma | Baseline, teoria |
| Stratified | $O(N)$ | $\leq$ multinomial | Fraca | Geral |
| Systematic | $O(N)$ | $\approx$ stratified | Forte | **Padrao na pratica** |
| Residual | $O(N)$ | $\leq$ multinomial | Parcial | Quando $NW^{(i)} \approx$ inteiros |
| Killing | $O(N)$ | Variavel | Nenhuma | SMC samplers |
| Optimal Transport | $O(N^2)$ | Minima (espacial) | Forte | Alta dimensao |

---

## 4. Adaptive Resampling

### 4.1 Quando Resamplear: ESS como Criterio

Resamplear a cada passo introduz variancia desnecessaria quando os pesos ja sao razoavelmente uniformes. A estrategia **adaptativa** resampleia apenas quando necessario.

!!! note "Regra de Resampling Adaptativo"
    Resamplear no passo $t$ se e somente se:

    $$
    \text{ESS}_t = \frac{1}{\sum_{i=1}^{N} (W_t^{(i)})^2} < \kappa \cdot N
    $$

    onde $\kappa \in (0, 1]$ e o **threshold** de resampling.

    Valores tipicos: $\kappa = 0.5$ (resamplear quando ESS cai abaixo de $N/2$).

### 4.2 Escolha do Threshold

!!! abstract "Proposicao: Trade-off do Threshold (Liu & Chen, 1995)"
    O threshold $\kappa$ controla um trade-off:

    - **$\kappa$ alto** (proximo de 1): resampling frequente
        - Menor degeneracao dos pesos
        - Maior variancia por resampling repetido
        - Mais perda de diversidade (sample impoverishment)

    - **$\kappa$ baixo** (proximo de 0): resampling raro
        - Maior degeneracao dos pesos
        - Menor variancia por resampling
        - Melhor diversidade mantida

A escolha ideal depende do modelo:

| Cenario | $\kappa$ recomendado | Justificativa |
|---|---|---|
| Padrao | 0.5 | Bom equilibrio geral |
| Likelihood suave | 0.3 | Pesos variam pouco, menos resampling necessario |
| Likelihood concentrada | 0.7 | Pesos degeneram rapido, precisa resamplear mais |
| SMC sampler com MCMC | 0.5-0.7 | MCMC apos resampling restaura diversidade |
| Alta dimensao | 0.5-0.8 | Weight collapse e mais severo |

### 4.3 Impacto na Variancia do Estimador

!!! abstract "Teorema: Variancia com Resampling Adaptativo (Del Moral, Doucet & Jasra, 2012)"
    Considere SMC com resampling adaptativo no passo $t$ se $\text{ESS}_t < \kappa N$. A variancia assintotica do estimador satisfaz:

    $$
    \sigma_t^2(\varphi) = \sum_{s=0}^{t} \mathbb{1}_{R_s} \cdot V_s^{\text{res}}(\varphi_s) + V_s^{\text{IS}}(\varphi_s)
    $$

    onde $R_s$ indica se resampling ocorreu no passo $s$, $V_s^{\text{res}}$ e a variancia de resampling, e $V_s^{\text{IS}}$ e a variancia de IS.

!!! tip "Implicacao Pratica"
    O resampling adaptativo **nunca piora** o estimador assintoticamente (para $N \to \infty$) comparado com resampling a cada passo, e frequentemente melhora em pratica finita por:

    1. Evitar variancia desnecessaria quando pesos sao uniformes
    2. Preservar diversidade das particulas
    3. Manter trajetorias distintas (importante para smoothing)

!!! example "Exemplo: Impacto do Threshold"
    Considere um modelo de volatilidade estocastica com $T = 500$ observacoes e $N = 1000$ particulas.

    | $\kappa$ | Resampling steps | RMSE (filtragem) | Log-likelihood Var |
    |---|---|---|---|
    | 1.0 (sempre) | 500 | 0.142 | 2.34 |
    | 0.7 | ~350 | 0.128 | 1.87 |
    | 0.5 | ~200 | **0.121** | **1.65** |
    | 0.3 | ~100 | 0.135 | 1.91 |
    | 0.0 (nunca) | 0 | Diverge | $\infty$ |

    O valor $\kappa = 0.5$ tipicamente oferece o melhor trade-off, embora o otimo dependa do modelo especifico.

---

## Referencias

| Referencia | Contribuicao |
|---|---|
| Kong, Liu & Wong (1994). *Sequential imputations and Bayesian missing data problems* | Degeneracao dos pesos, ESS |
| Kitagawa (1996). *Monte Carlo filter and smoother* | Stratified resampling |
| Liu & Chen (1998). *Sequential Monte Carlo methods for dynamic systems* | Residual resampling |
| Carpenter, Clifford & Fearnhead (1999). *Improved particle filter* | Systematic resampling |
| Douc & Cappe (2005). *Comparison of resampling schemes for particle filtering* | Comparacao formal de variancias |
| Chopin (2004). *Central limit theorem for SMC* | CLT com resampling |
| Del Moral, Doucet & Jasra (2012). *On adaptive resampling strategies for SMC* | Teoria de resampling adaptativo |
| Reich (2013). *A nonparametric ensemble transform method for Bayesian inference* | Optimal transport resampling |
| Corenflos, Thornton, Deligiannidis & Doucet (2021). *Differentiable particle filtering via entropy-regularized OT* | OT resampling diferenciavel |
| Hol, Schon & Gustafsson (2006). *On resampling algorithms for particle filters* | Comparacao empirica de algoritmos |
| Gerber, Chopin & Whiteley (2019). *Negative association, ordering and convergence of resampling methods* | Teoria de associacao negativa e convergencia |
| Murray, Lee & Jacob (2016). *Parallel resampling in the particle filter* | Paralelizacao de resampling |
