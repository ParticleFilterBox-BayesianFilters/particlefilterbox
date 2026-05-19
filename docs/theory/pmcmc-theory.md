---
title: Particle MCMC - Teoria
description: Teoria de Particle MCMC - pseudo-marginal framework, PMMH, Particle Gibbs, PGAS, ancestor sampling, e resultados de convergencia.
---

# Particle MCMC: Teoria

Esta pagina desenvolve a teoria de **Particle Markov chain Monte Carlo (PMCMC)** (Andrieu, Doucet & Holenstein, 2010), que combina particle filters com MCMC para inferencia em modelos com likelihood intratavel. Os metodos PMCMC sao **exatos** — convergem para a posterior verdadeira apesar de usarem likelihood estimada.

---

## 1. Pseudo-Marginal Framework

### 1.1 O Problema da Likelihood Intratavel

Em modelos de estado-espaco, a likelihood marginal $p(y_{1:T} | \theta)$ requer integracao sobre os estados latentes:

$$
p(y_{1:T} | \theta) = \int p(y_{1:T}, x_{0:T} | \theta) \, dx_{0:T} = \int \prod_{t=1}^{T} g_\theta(y_t | x_t) \prod_{t=0}^{T} f_\theta(x_t | x_{t-1}) \, dx_{0:T}
$$

Esta integral e **intratavel** analiticamente para modelos nao-lineares/nao-gaussianos. Sem a likelihood, nao e possivel executar MCMC padrao (e.g., Metropolis-Hastings) para amostrar $\theta$.

!!! info "Ideia Central do Pseudo-Marginal"
    Em vez de calcular $p(y_{1:T}|\theta)$ exatamente, substituimos por um **estimador nao-enviesado** $\hat{p}(y_{1:T}|\theta)$ obtido via particle filter. O resultado notavel e que o MCMC resultante converge para a **posterior exata** $p(\theta | y_{1:T})$.

### 1.2 O Teorema Pseudo-Marginal

!!! abstract "Teorema: Pseudo-Marginal MCMC (Beaumont, 2003; Andrieu & Roberts, 2009)"
    Seja $\hat{p}(y_{1:T}|\theta)$ um estimador **nao-enviesado e nao-negativo** da likelihood $p(y_{1:T}|\theta)$, i.e.:

    $$
    \mathbb{E}_{U}[\hat{p}(y_{1:T}|\theta)] = p(y_{1:T}|\theta), \quad \hat{p}(y_{1:T}|\theta) \geq 0 \text{ q.c.}
    $$

    onde $U$ denota as variaveis aleatorias auxiliares usadas na estimacao (e.g., particulas e indices de resampling no PF).

    Entao, o algoritmo Metropolis-Hastings com acceptance ratio:

    $$
    \alpha(\theta^*, \theta) = \min\left(1, \frac{\hat{p}(y_{1:T}|\theta^*) \, p(\theta^*)  \, q(\theta | \theta^*)}{\hat{p}(y_{1:T}|\theta) \, p(\theta) \, q(\theta^* | \theta)}\right)
    $$

    tem como distribuicao estacionaria a **posterior exata** $p(\theta | y_{1:T})$.

??? note "Prova: Pseudo-Marginal via Extended Target"
    A prova procede pela construcao de um **espaco estendido** que inclui os parametros $\theta$ e as variaveis auxiliares $U$ usadas pelo estimador.

    **Passo 1: Definicao do espaco estendido.**

    Seja $U$ o vetor de variaveis aleatorias geradas pelo particle filter (particulas, indices de resampling) com distribuicao $\psi_\theta(u)$. Definimos a target estendida:

    $$
    \tilde{p}(\theta, u) = \frac{\hat{p}_u(y_{1:T}|\theta) \, p(\theta) \, \psi_\theta(u)}{p(y_{1:T})}
    $$

    onde $\hat{p}_u(y_{1:T}|\theta)$ denota o estimador como funcao deterministica de $(\theta, u)$.

    **Passo 2: Verificacao da marginal.**

    Integrando sobre $u$:

    $$
    \int \tilde{p}(\theta, u) \, du = \frac{p(\theta)}{p(y_{1:T})} \int \hat{p}_u(y_{1:T}|\theta) \, \psi_\theta(u) \, du
    = \frac{p(\theta)}{p(y_{1:T})} \cdot \mathbb{E}_U[\hat{p}_U(y_{1:T}|\theta)]
    $$

    Pela hipotese de unbiasedness:

    $$
    = \frac{p(\theta) \cdot p(y_{1:T}|\theta)}{p(y_{1:T})} = p(\theta | y_{1:T})
    $$

    Portanto, a **marginal em $\theta$** da target estendida e exatamente a posterior $p(\theta | y_{1:T})$.

    **Passo 3: Equivalencia do MH.**

    No espaco estendido, propomos $(\theta^*, u^*)$ onde $\theta^* \sim q(\cdot|\theta)$ e $u^* \sim \psi_{\theta^*}(\cdot)$ (novo run do PF). O acceptance ratio no espaco estendido e:

    $$
    \frac{\tilde{p}(\theta^*, u^*) \, q(\theta|\theta^*) \, \psi_\theta(u)}{\tilde{p}(\theta, u) \, q(\theta^*|\theta) \, \psi_{\theta^*}(u^*)}
    = \frac{\hat{p}_{u^*}(y|\theta^*) p(\theta^*) \psi_{\theta^*}(u^*) q(\theta|\theta^*) \psi_\theta(u)}{\hat{p}_u(y|\theta) p(\theta) \psi_\theta(u) q(\theta^*|\theta) \psi_{\theta^*}(u^*)}
    $$

    Os termos $\psi$ cancelam:

    $$
    = \frac{\hat{p}_{u^*}(y|\theta^*) \, p(\theta^*) \, q(\theta|\theta^*)}{\hat{p}_u(y|\theta) \, p(\theta) \, q(\theta^*|\theta)}
    $$

    que e exatamente o acceptance ratio do pseudo-marginal MH. $\blacksquare$

### 1.3 Impacto da Variancia do Estimador

A eficiencia do pseudo-marginal depende criticamente da **variancia** do estimador de likelihood.

!!! abstract "Proposicao: Impacto no Acceptance Rate (Pitt et al., 2012)"
    Seja $\sigma^2_{\log} = \text{Var}[\log \hat{p}(y|\theta)]$ avaliada no valor verdadeiro de $\theta$. Entao:

    - A **taxa de aceitacao media** diminui com $\sigma^2_{\log}$
    - O **mixing** da cadeia deteriora-se com $\sigma^2_{\log}$
    - A **regra otima**: $\sigma^2_{\log} \approx 1$ equilibra custo computacional e mixing

!!! tip "Regra Pratica: $\text{Var}[\log \hat{p}] \approx 1$"
    Deshpande, Lindsten & Schon (2016) e Sherlock et al. (2015) mostram que:

    - $\sigma^2_{\log} \ll 1$: estimador muito preciso, mas custo excessivo (N muito grande)
    - $\sigma^2_{\log} \gg 1$: estimador impreciso, a cadeia "fica presa" em estados com likelihood superestimada
    - $\sigma^2_{\log} \approx 1$: equilibrio otimo entre custo e mixing

    Na pratica, calibra-se $N$ (numero de particulas) para atingir $\sigma^2_{\log} \approx 1$ numa estimativa piloto.

---

## 2. Particle Marginal Metropolis-Hastings (PMMH)

### 2.1 Extended Target do PMMH

O PMMH (Andrieu, Doucet & Holenstein, 2010) e a instanciacao do pseudo-marginal com particle filter como estimador de likelihood. A target estendida e definida sobre $(\theta, x_{1:T}^{1:N}, a_{1:T-1}^{1:N})$, onde $x_t^{1:N}$ sao as particulas e $a_t^{1:N}$ os indices de ancestralidade.

!!! abstract "Definicao: Target Estendida do PMMH"
    A target estendida do PMMH e:

    $$
    \tilde{\pi}(\theta, x_{1:T}^{1:N}, a_{1:T-1}^{1:N}) = p(\theta) \frac{\hat{p}_N(y_{1:T}|\theta)}{N^T} \prod_{t=1}^{T} \left[\prod_{i=1}^{N} q_\theta(x_t^{(i)} | x_{t-1}^{(a_{t-1}^{(i)})}) \cdot \prod_{i=1}^{N} \frac{w_{t-1}^{(a_{t-1}^{(i)})}}{\sum_j w_{t-1}^{(j)}}\right]
    $$

    onde $\hat{p}_N(y_{1:T}|\theta) = \prod_{t=1}^T \frac{1}{N}\sum_{i=1}^N \tilde{w}_t^{(i)}$ e o estimador SMC da likelihood.

### 2.2 Marginal em $\theta$

!!! abstract "Teorema: Marginal Correta (Andrieu, Doucet & Holenstein, 2010)"
    A marginal da target estendida em $\theta$ e a posterior verdadeira:

    $$
    \int \tilde{\pi}(\theta, x_{1:T}^{1:N}, a_{1:T-1}^{1:N}) \, dx_{1:T}^{1:N} \, da_{1:T-1}^{1:N} \propto p(\theta) p(y_{1:T} | \theta) = p(\theta | y_{1:T}) \cdot p(y_{1:T})
    $$

    Este resultado decorre da unbiasedness do estimador SMC da likelihood (ver secao anterior).

### 2.3 Algoritmo PMMH

**Algoritmo: Particle Marginal Metropolis-Hastings**

1. **Inicializacao**: Escolher $\theta_0$, executar PF com $N$ particulas para obter $\hat{p}_N(y_{1:T}|\theta_0)$
2. Para $m = 1, 2, \ldots, M$:
    1. Propor $\theta^* \sim q(\cdot | \theta_{m-1})$
    2. Executar PF com $N$ particulas para obter $\hat{p}_N(y_{1:T}|\theta^*)$
    3. Calcular acceptance ratio:

    $$
    \alpha = \min\left(1, \frac{\hat{p}_N(y_{1:T}|\theta^*) \, p(\theta^*) \, q(\theta_{m-1}|\theta^*)}{\hat{p}_N(y_{1:T}|\theta_{m-1}) \, p(\theta_{m-1}) \, q(\theta^*|\theta_{m-1})}\right)
    $$

    4. Com probabilidade $\alpha$: $\theta_m = \theta^*$; senao: $\theta_m = \theta_{m-1}$

!!! warning "Nota Importante"
    Quando $\theta^*$ e **rejeitado**, o valor antigo de $\hat{p}_N(y|\theta_{m-1})$ e **reutilizado** — nao se re-executa o PF. Isto e essencial para a corretude: cada estado da cadeia esta associado a uma realizacao especifica das variaveis auxiliares.

### 2.4 Impacto de $N$ no Acceptance Rate

A escolha do numero de particulas $N$ gera um trade-off fundamental:

| $N$ | $\text{Var}[\log \hat{p}]$ | Acceptance rate | Custo por iteracao | Eficiencia |
|---|---|---|---|---|
| Pequeno | Alto | Baixo | Baixo | Baixa |
| Otimo | $\approx 1$ | Moderado | Moderado | **Maxima** |
| Grande | Baixo | Alto | Alto | Baixa |

!!! abstract "Proposicao: Acceptance Rate Assintotico (Sherlock et al., 2015)"
    Para propostas random-walk com escala otima, o acceptance rate assintotico do PMMH e:

    $$
    \bar{\alpha} \approx 2\Phi\left(-\frac{1}{2}\sqrt{d\ell^2 + \sigma^2_{\log}}\right)
    $$

    onde $d$ e a dimensao de $\theta$, $\ell$ e o step size da proposta, e $\Phi$ e a CDF normal. Para $\sigma^2_{\log} \to 0$, recupera-se o resultado classico de Roberts, Gelman & Gilks (1997) com acceptance rate otimo de $\approx 0.234$.

---

## 3. Particle Gibbs (PG)

### 3.1 Conditional SMC

O **Particle Gibbs** (Andrieu, Doucet & Holenstein, 2010) utiliza um mecanismo diferente do PMMH: em vez de substituir a likelihood, executa um **Conditional SMC** (CSMC) como um Gibbs step.

!!! abstract "Definicao: Conditional SMC"
    Dado um **reference trajectory** $x_{0:T}' = (x_0', x_1', \ldots, x_T')$, o Conditional SMC executa um particle filter padrao com a restricao de que **uma das particulas e fixada** como $x'$ em todos os tempos:

    - No tempo $t$: particula $x_t^{(N)} = x_t'$ e fixada (nao e amostrada da proposta)
    - As demais $N-1$ particulas sao amostradas normalmente
    - Resampling inclui a particula fixada como candidata

### 3.2 Invariancia da Target

!!! abstract "Teorema: Invariancia do Particle Gibbs (Andrieu, Doucet & Holenstein, 2010)"
    Considere a seguinte iteracao do Particle Gibbs:

    1. Amostrar $\theta^* \mid x_{0:T} \sim p(\theta | x_{0:T}, y_{1:T})$ (Gibbs step para $\theta$)
    2. Executar Conditional SMC com $\theta^*$ e reference trajectory $x_{0:T}$
    3. Selecionar novo trajectory $x_{0:T}^* = x_{0:T}^{(k)}$ com $k \sim \text{Categorical}(W_T^{1:N})$

    Esta iteracao **preserva a target estendida** $\tilde{\pi}(\theta, x_{0:T}^{1:N}, a_{0:T-1}^{1:N})$ e, consequentemente, a marginal $p(\theta, x_{0:T} | y_{1:T})$.

??? note "Sketch da Prova: Invariancia"
    A prova mostra que o CSMC define um kernel de transicao $\pi$-invariante no espaco estendido.

    1. **Espaco estendido**: Defina $z = (\theta, x_{1:T}^{1:N}, a_{1:T-1}^{1:N}, k)$ onde $k$ e o indice da trajetoria selecionada. A target estendida e:

    $$
    \tilde{\pi}(z) \propto p(\theta) \, W_T^{(k)} \prod_{t=1}^T \left[\prod_{i \neq b_t} q_\theta(x_t^{(i)} | x_{t-1}^{(a_{t-1}^{(i)})}) \frac{w_{t-1}^{(a_{t-1}^{(i)})}}{\sum_j w_{t-1}^{(j)}}\right] \prod_{t=1}^T g_\theta(y_t | x_t^{(b_t)})
    $$

    onde $b_t$ e o indice do ancestral de $k$ no tempo $t$.

    2. **CSMC como Gibbs update**: O CSMC amostra todas as variaveis auxiliares (particulas, indices) **condicionalmente** a trajetoria de referencia. Isto constitui um Gibbs step no espaco estendido.

    3. **Gibbs preserva target**: Como cada Gibbs step amostra de sua condicional completa, a composicao preserva a target conjunta. $\blacksquare$

### 3.3 Path Degeneracy

O problema fundamental do Particle Gibbs padrao e a **path degeneracy**: no particle filter, as trajetorias $x_{0:T}^{(i)}$ coalescem para tempos iniciais.

!!! danger "Path Degeneracy no PG"
    Apos resampling repetido, todos os ancestrais do PF convergem para um unico ancestral inicial:

    $$
    x_{0:t_0}^{(i)} = x_{0:t_0}^{(j)} \quad \text{para todo } i, j \text{ e algum } t_0 \text{ proximo de } 0
    $$

    **Consequencia para PG**: Como a trajetoria de referencia e fixada, e as demais trajetorias coalescem com ela para tempos iniciais, o CSMC efetivamente **nao atualiza** a parte inicial da trajetoria. Isto resulta em:

    - Mixing extremamente lento para $x_t$ com $t$ pequeno
    - Autocorrelacao alta na cadeia MCMC
    - Necessidade de $N$ muito grande para mitigar (impratico para $T$ grande)

---

## 4. Particle Gibbs with Ancestor Sampling (PGAS)

### 4.1 Ancestor Sampling como Solucao

O **PGAS** (Lindsten, Jordan & Schon, 2014) resolve a path degeneracy adicionando um passo de **ancestor sampling** ao Conditional SMC.

!!! abstract "Definicao: Ancestor Sampling"
    No CSMC com ancestor sampling, em cada tempo $t$, o **ancestral da particula de referencia** e re-amostrado:

    $$
    a_{t-1}^{(N)} \sim \text{Categorical}(\tilde{w}_{t|T}^{(1)}, \ldots, \tilde{w}_{t|T}^{(N)})
    $$

    com pesos de ancestor sampling:

    $$
    \tilde{w}_{t|T}^{(i)} \propto w_{t-1}^{(i)} \cdot f_\theta(x_t' | x_{t-1}^{(i)}) \cdot \frac{p_\theta(y_{t+1:T} | x_t')}{p_\theta(y_{t+1:T} | x_t')}
    $$

    Na pratica, como $p(y_{t+1:T} | x_t')$ cancela (e a mesma para todo $i$), os pesos simplificam para:

    $$
    \tilde{w}_{t|T}^{(i)} \propto w_{t-1}^{(i)} \cdot f_\theta(x_t' | x_{t-1}^{(i)})
    $$

!!! tip "Intuicao"
    O ancestor sampling **reconecta** a trajetoria de referencia a diferentes historias do particle filter. Em vez de manter o ancestral fixo (que leva a degeneracy), permite que a trajetoria de referencia "salte" entre as diferentes linhagens ancestrais, quebrando a coalescencia.

### 4.2 Preservacao da Target

!!! abstract "Teorema: Validade do PGAS (Lindsten, Jordan & Schon, 2014)"
    O CSMC com ancestor sampling define um kernel de transicao que preserva a target estendida do Particle Gibbs. Consequentemente, o PGAS converge para a posterior verdadeira $p(\theta, x_{0:T} | y_{1:T})$.

??? note "Prova: Ancestor Sampling Preserva a Target"
    A prova mostra que o ancestor sampling e equivalente a um Gibbs update na variavel de ancestralidade.

    **Passo 1: Target estendida.**

    No espaco estendido, a target inclui os indices de ancestralidade $a_{t-1}^{(N)}$ para a trajetoria de referencia. A condicional completa de $a_{t-1}^{(N)}$ dado o resto e:

    $$
    p(a_{t-1}^{(N)} = i \mid \text{resto}) \propto \tilde{\pi}(\text{tudo com } a_{t-1}^{(N)} = i)
    $$

    **Passo 2: Calculo da condicional.**

    Examinando a dependencia da target estendida em $a_{t-1}^{(N)}$, os unicos termos que dependem de $i$ sao:

    - O peso de resampling: $w_{t-1}^{(i)} / \sum_j w_{t-1}^{(j)}$
    - A transicao: $f_\theta(x_t^{(N)} | x_{t-1}^{(i)})$

    Portanto:

    $$
    p(a_{t-1}^{(N)} = i \mid \text{resto}) \propto w_{t-1}^{(i)} \cdot f_\theta(x_t' | x_{t-1}^{(i)})
    $$

    **Passo 3: Equivalencia.**

    Os pesos do ancestor sampling $\tilde{w}_{t|T}^{(i)} \propto w_{t-1}^{(i)} f_\theta(x_t'|x_{t-1}^{(i)})$ sao exatamente a **condicional completa** de $a_{t-1}^{(N)}$. Portanto, o ancestor sampling e um Gibbs update valido. $\blacksquare$

### 4.3 Uniform Ergodicity

!!! abstract "Teorema: Uniform Ergodicity do PGAS (Lindsten, Jordan & Schon, 2014)"
    Sob condicoes de regularidade:

    1. **Transition density limitada**: $0 < c_f \leq f_\theta(x_t|x_{t-1}) \leq C_f < \infty$
    2. **Observation density limitada**: $0 < c_g \leq g_\theta(y_t|x_t) \leq C_g < \infty$

    O kernel do PGAS e **uniformemente ergodico**: existe $\rho \in [0, 1)$ tal que:

    $$
    \|K_{\text{PGAS}}^m(z, \cdot) - \pi(\cdot)\|_{\text{TV}} \leq C \rho^m
    $$

    para todo estado inicial $z$ e todo $m \geq 1$.

!!! info "Importancia da Uniform Ergodicity"
    - **Convergencia geometrica**: a cadeia esquece seu estado inicial a taxa exponencial
    - **Independencia de $T$**: o bound vale para qualquer comprimento de serie $T$, com constantes que podem depender de $T$ mas a taxa $\rho$ pode ser mantida
    - **Independencia de $N$**: a ergodicidade vale **para qualquer $N \geq 2$**, incluindo $N = 2$ (uma particula livre + referencia)

### 4.4 Comparacao de Mixing: PGAS vs PG

A vantagem do PGAS sobre o PG padrao e dramatica:

| Propriedade | Particle Gibbs (PG) | PGAS |
|---|---|---|
| Path degeneracy | Severa para $T$ grande | Resolvida por ancestor sampling |
| Mixing para $x_t$, $t$ pequeno | Muito lento | Rapido |
| $N$ necessario | $O(T)$ para mixing razoavel | $N = O(1)$ suficiente |
| Uniform ergodicity | Requer $N \to \infty$ com $T$ | $N \geq 2$ suficiente |
| Autocorrelacao | Alta, cresce com $T$ | Baixa, estavel com $T$ |

!!! tip "Resultado Pratico"
    Em problemas com $T = 1000$, o PG pode necessitar $N = 500$+ particulas para mixing aceitavel, enquanto PGAS funciona bem com $N = 50$-$100$. A economia computacional e de uma **ordem de magnitude** ou mais.

---

## 5. Resultados de Convergencia

### 5.1 Ergodicidade dos Algoritmos PMCMC

Os algoritmos PMCMC sao **cadeias de Markov no espaco estendido**. Seus resultados de convergencia derivam da teoria geral de MCMC aplicada a este espaco.

!!! abstract "Teorema: Ergodicidade do PMMH (Andrieu & Roberts, 2009)"
    Se o MCMC "ideal" (com likelihood exata) e geometricamente ergodico, entao o PMMH tambem e geometricamente ergodico para qualquer $N \geq 1$, desde que:

    $$
    \mathbb{E}\left[\hat{p}(y_{1:T}|\theta)^{2+\epsilon}\right] < \infty \quad \text{para algum } \epsilon > 0
    $$

    A taxa de convergencia deteriora-se com a variancia do estimador, mas a **ergodicidade geometrica e preservada**.

??? note "Sketch da Prova"
    1. O PMMH opera no espaco estendido $(\theta, U)$ onde $U$ sao as variaveis auxiliares do PF

    2. A target estendida $\tilde{\pi}(\theta, u) \propto p(\theta) \hat{p}_u(y|\theta) \psi_\theta(u)$ tem marginal $p(\theta|y)$ em $\theta$

    3. A condicao de **drift** para ergodicidade geometrica do MH ideal transfere-se para o espaco estendido:

    $$
    \mathbb{E}[V(\theta^*, U^*) | \theta, u] \leq \lambda V(\theta) + L
    $$

    com funcao de Lyapunov $V(\theta) = p(\theta|y)^{-\delta}$ para $\delta > 0$ pequeno

    4. A condicao de momento $\mathbb{E}[\hat{p}^{2+\epsilon}] < \infty$ garante que os pesos no espaco estendido sao suficientemente bem-comportados. $\blacksquare$

### 5.2 Mixing Time Bounds

!!! abstract "Proposicao: Mixing Time do PMMH"
    O mixing time do PMMH satisfaz:

    $$
    t_{\text{mix}}^{\text{PMMH}}(\epsilon) \leq t_{\text{mix}}^{\text{ideal}}(\epsilon) \cdot \left(1 + C \cdot \text{Var}[\log \hat{p}(y|\theta)]\right)
    $$

    onde $t_{\text{mix}}^{\text{ideal}}$ e o mixing time do MH ideal (com likelihood exata).

Isto formaliza a intuicao de que:

- $\text{Var}[\log \hat{p}] \to 0$ ($N \to \infty$): PMMH recupera o mixing do MH ideal
- $\text{Var}[\log \hat{p}]$ grande: mixing time inflado proporcionalmente

### 5.3 Central Limit Theorems para PMCMC

!!! abstract "Teorema: CLT para Estimadores PMCMC"
    Para funcoes $\varphi(\theta)$ com $\pi(\varphi^2) < \infty$, os estimadores PMCMC satisfazem o CLT:

    $$
    \sqrt{M}\left(\frac{1}{M}\sum_{m=1}^{M} \varphi(\theta_m) - \mathbb{E}_\pi[\varphi(\theta)]\right) \xrightarrow{d} \mathcal{N}(0, \sigma_{\text{PMCMC}}^2(\varphi))
    $$

    onde $M$ e o numero de iteracoes MCMC e a **variancia assintotica** admite a decomposicao:

    $$
    \sigma_{\text{PMCMC}}^2(\varphi) = \sigma_{\text{MCMC}}^2(\varphi) + \sigma_{\text{noise}}^2(\varphi)
    $$

    - $\sigma_{\text{MCMC}}^2$: variancia devido a correlacao da cadeia (presente mesmo com likelihood exata)
    - $\sigma_{\text{noise}}^2$: variancia adicional devida a estimacao da likelihood (desaparece com $N \to \infty$)

### 5.4 Comparacao de Variancia Assintotica

Para os tres principais algoritmos PMCMC:

!!! abstract "Proposicao: Ordenacao de Eficiencia"
    Sob condicoes de regularidade e para $N$ suficientemente grande:

    $$
    \sigma_{\text{PGAS}}^2(\varphi) \leq \sigma_{\text{PG}}^2(\varphi) \leq \sigma_{\text{PMMH}}^2(\varphi)
    $$

    com desigualdade estrita em geral. O PGAS e **uniformemente mais eficiente** que PG, que por sua vez tende a ser mais eficiente que PMMH para problemas com estados latentes correlacionados.

| Algoritmo | Variancia assintotica | Custo por iteracao | Eficiencia total |
|---|---|---|---|
| PMMH | $\sigma_{\text{MCMC}}^2 + O(1/N)$ | $O(NT)$ | Melhor para $\theta$ de baixa dimensao |
| PG | Menor que PMMH (Gibbs) | $O(NT)$ | Melhor com condicionais conjugadas |
| PGAS | Menor que PG | $O(NT)$ | **Melhor em geral** para estado-espaco |

### 5.5 Implicacoes Praticas para Tuning

!!! tip "Guia de Tuning Baseado na Teoria"
    **PMMH:**

    1. Calibrar $N$ para $\text{Var}[\log \hat{p}(y|\theta)] \approx 1$
    2. Usar proposta random-walk adaptativa com $\Sigma = 2.38^2 / d \cdot \hat{\Sigma}_\theta$
    3. Target acceptance rate: $\sim 15\%$-$30\%$ (ajustado para noise do estimador)

    **Particle Gibbs:**

    1. Usar $N$ suficiente para evitar path degeneracy ($N \geq T/5$ como regra)
    2. Explorar condicionais conjugadas quando disponiveis
    3. Monitorar autocorrelacao dos estados latentes

    **PGAS:**

    1. $N = 50$-$100$ tipicamente suficiente, independente de $T$
    2. Monitorar acceptance rate do ancestor sampling
    3. Se transition density $f_\theta$ e muito informativa: ancestor sampling funciona bem
    4. Se $f_\theta$ e vaga: ancestor sampling menos eficaz, considerar aumentar $N$

!!! warning "Diagnosticos Essenciais"
    - **Trace plots** de $\theta$ e $\log p(y|\theta)$: verificar convergencia e mixing
    - **Autocorrelacao**: ESS efetivo deve ser razoavel ($> 100$ para inferencia confiavel)
    - **$\hat{R}$** (Gelman-Rubin): executar multiplas cadeias e verificar $\hat{R} < 1.01$
    - **Acceptance rate**: PMMH deve ter $15\%$-$30\%$; muito baixo indica $N$ insuficiente ou proposta ruim

---

## Referencias

| Referencia | Contribuicao |
|---|---|
| Andrieu, Doucet & Holenstein (2010). *Particle Markov chain Monte Carlo methods* | Framework geral de PMCMC: PMMH, PG, PGAS |
| Andrieu & Roberts (2009). *The pseudo-marginal approach for efficient Monte Carlo computations* | Teoria pseudo-marginal, ergodicidade |
| Beaumont (2003). *Estimation of population growth or decline in genetically monitored populations* | Primeiro uso de pseudo-marginal MH |
| Lindsten, Jordan & Schon (2014). *Particle Gibbs with ancestor sampling* | PGAS, uniform ergodicity |
| Chopin & Singh (2015). *On particle Gibbs sampling* | Analise teorica do Particle Gibbs |
| Sherlock, Thiery, Roberts & Rosenthal (2015). *On the efficiency of pseudo-marginal random walk Metropolis algorithms* | Tuning otimo de PMMH, regra $\sigma^2 \approx 1$ |
| Pitt, dos Santos Silva, Giordani & Kohn (2012). *On some properties of Markov chain Monte Carlo simulation methods based on the particle filter* | Impacto de $N$ no acceptance rate |
| Doucet, Pitt, Deligiannidis & Kohn (2015). *Efficient implementation of Markov chain Monte Carlo when using an unbiased likelihood estimator* | Implementacao eficiente de PMCMC |
| Roberts, Gelman & Gilks (1997). *Weak convergence and optimal scaling of random walk Metropolis algorithms* | Scaling otimo de MH, acceptance rate 0.234 |
| Lindsten & Schon (2013). *Backward simulation methods for Monte Carlo statistical inference* | Backward simulation e conexao com PGAS |
