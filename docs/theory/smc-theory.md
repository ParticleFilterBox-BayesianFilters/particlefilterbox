---
title: Sequential Monte Carlo - Fundamentos Teoricos
description: Teoria de Importance Sampling, Sequential Importance Sampling, framework SMC e estimacao de marginal likelihood.
---

# Sequential Monte Carlo: Fundamentos Teoricos

Esta pagina apresenta os fundamentos matematicos de **Sequential Monte Carlo (SMC)**, desde importance sampling ate o framework completo com garantias de convergencia.

---

## 1. Importance Sampling

### 1.1 Distribuicao Target vs Proposal

O objetivo fundamental e calcular integrais da forma:

$$
I = \mathbb{E}_{\pi}[\varphi(x)] = \int \varphi(x) \, \pi(x) \, dx
$$

onde $\pi(x)$ e a **distribuicao target** (alvo). Quando nao e possivel amostrar diretamente de $\pi$, utilizamos uma **distribuicao proposal** (proposta) $q(x)$ da qual sabemos amostrar.

!!! info "Ideia Central"
    Importance sampling reformula a integral como uma expectativa sob $q$:

    $$
    I = \int \varphi(x) \frac{\pi(x)}{q(x)} q(x) \, dx = \mathbb{E}_{q}\left[\varphi(x) \, w(x)\right]
    $$

    onde $w(x) = \frac{\pi(x)}{q(x)}$ sao os **importance weights**.

### 1.2 Importance Weights e Self-Normalized IS

Na pratica, $\pi(x)$ frequentemente e conhecida apenas ate uma constante de normalizacao: $\pi(x) = \tilde{\pi}(x) / Z_{\pi}$. Neste caso, usamos o estimador **self-normalized**:

$$
\hat{I}_{N}^{\text{SN}} = \frac{\sum_{i=1}^{N} \tilde{w}^{(i)} \varphi(x^{(i)})}{\sum_{i=1}^{N} \tilde{w}^{(i)}}
= \sum_{i=1}^{N} W^{(i)} \varphi(x^{(i)})
$$

onde $\tilde{w}^{(i)} = \tilde{\pi}(x^{(i)}) / q(x^{(i)})$ sao os pesos nao-normalizados e $W^{(i)} = \tilde{w}^{(i)} / \sum_j \tilde{w}^{(j)}$ sao os pesos normalizados.

!!! note "Propriedades do Estimador Self-Normalized"
    - **Consistente**: $\hat{I}_N^{\text{SN}} \xrightarrow{a.s.} I$ quando $N \to \infty$
    - **Enviesado** para $N$ finito (razao de estimadores)
    - **Vies**: $\text{Bias} = O(1/N)$, desaparecendo assintoticamente

### 1.3 Variancia e Eficiencia

!!! abstract "Teorema: Variancia do IS"
    Para o estimador de importance sampling $\hat{I}_N = \frac{1}{N}\sum_{i=1}^{N} w(x^{(i)}) \varphi(x^{(i)})$ com $x^{(i)} \sim q$:

    $$
    \text{Var}(\hat{I}_N) = \frac{1}{N} \text{Var}_{q}\left[\frac{\pi(X)}{q(X)} \varphi(X)\right]
    = \frac{1}{N} \int \left(\frac{\pi(x)}{q(x)}\right)^2 [\varphi(x) - I]^2 \, q(x) \, dx
    $$

A eficiencia de IS depende criticamente da **similaridade** entre $q$ e $\pi$. Uma metrica fundamental e o **Effective Sample Size (ESS)**:

$$
\text{ESS} = \frac{N}{1 + \text{Var}_{q}[w(X)]} \approx \frac{\left(\sum_{i=1}^{N} w^{(i)}\right)^2}{\sum_{i=1}^{N} (w^{(i)})^2}
$$

!!! warning "Maldimensionalidade do IS"
    Em dimensoes altas, a variancia dos pesos cresce exponencialmente com a dimensao $d$, tornando IS puro inviavel. Para $\pi = \mathcal{N}(0, I_d)$ e $q = \mathcal{N}(\mu, I_d)$:

    $$
    \text{Var}_q[w(X)] = \exp(d \|\mu\|^2) - 1
    $$

    Esta **maldimensionalidade** motiva a abordagem sequencial.

---

## 2. Sequential Importance Sampling (SIS)

### 2.1 Extensao para Sequencias de Distribuicoes

Considere uma sequencia de distribuicoes $\pi_t(x_{1:t})$ para $t = 1, \ldots, T$, onde $x_{1:t} = (x_1, \ldots, x_t)$ denota o historico completo. A ideia central de SIS e construir a proposta **incrementalmente**:

$$
q(x_{1:T}) = q_1(x_1) \prod_{t=2}^{T} q_t(x_t | x_{1:t-1})
$$

Os pesos sao atualizados recursivamente:

$$
w_t(x_{1:t}) = \frac{\tilde{\pi}_t(x_{1:t})}{q(x_{1:t})}
= w_{t-1}(x_{1:t-1}) \cdot \frac{\tilde{\pi}_t(x_{1:t})}{\tilde{\pi}_{t-1}(x_{1:t-1}) \, q_t(x_t|x_{1:t-1})}
$$

!!! info "Peso Incremental"
    Definimos o **peso incremental** como:

    $$
    \alpha_t(x_{1:t}) = \frac{\tilde{\pi}_t(x_{1:t})}{\tilde{\pi}_{t-1}(x_{1:t-1}) \, q_t(x_t | x_{1:t-1})}
    $$

    de modo que $w_t = w_{t-1} \cdot \alpha_t$.

### 2.2 Weight Degeneracy Problem

O problema fundamental de SIS e a **degeneracao dos pesos**: apos poucos passos, um unico peso domina todos os outros.

!!! abstract "Teorema: Degeneracao Exponencial dos Pesos (Kong et al., 1994)"
    Seja $\text{ESS}_t$ o effective sample size no passo $t$. Sob condicoes de regularidade:

    $$
    \frac{\text{ESS}_t}{N} \xrightarrow{p} \frac{1}{1 + \text{Var}_{\pi}[w_t]} \leq \frac{1}{1 + \text{cv}_t^2}
    $$

    onde $\text{cv}_t^2 = \text{Var}[w_t] / \mathbb{E}[w_t]^2$ e o coeficiente de variacao dos pesos.

??? note "Sketch da Prova: Degeneracao Exponencial"
    **Proposicao**: Para targets $\pi_t$ em espacos de dimensao fixa, o coeficiente de variacao dos pesos cresce exponencialmente com $t$:

    $$
    \text{cv}_T^2 \geq \prod_{t=1}^{T} (1 + \sigma_t^2) - 1
    $$

    onde $\sigma_t^2 = \text{Var}[\alpha_t]$.

    **Prova (sketch)**:

    1. Os pesos fatorizam como $w_T = \prod_{t=1}^{T} \alpha_t$
    2. Pela desigualdade de Jensen aplicada a funcao convexa $x^2$:

        $$
        \mathbb{E}[w_T^2] = \mathbb{E}\left[\prod_{t=1}^{T} \alpha_t^2\right]
        \geq \prod_{t=1}^{T} \mathbb{E}[\alpha_t^2]
        $$

    3. Portanto:

        $$
        \text{cv}_T^2 = \frac{\mathbb{E}[w_T^2]}{\mathbb{E}[w_T]^2} - 1
        \geq \prod_{t=1}^{T}(1 + \sigma_t^2) - 1
        $$

    4. Se $\sigma_t^2 \geq \sigma_{\min}^2 > 0$ para todo $t$, entao $\text{cv}_T^2 \geq (1 + \sigma_{\min}^2)^T - 1$, que cresce **exponencialmente** em $T$. $\blacksquare$

!!! danger "Consequencia Pratica"
    Sem resampling, SIS requer um numero de particulas **exponencial em $T$** para manter estimativas acuradas. Para $T = 100$ passos, isso e computacionalmente impossivel.

### 2.3 Exemplo: SIS para Filtragem

!!! example "SIS para Hidden Markov Model"
    Para o modelo $x_t \sim f(x_t|x_{t-1})$, $y_t \sim g(y_t|x_t)$, usando a transicao como proposta $q_t(x_t|x_{1:t-1}) = f(x_t|x_{t-1})$:

    $$
    \alpha_t = g(y_t | x_t)
    $$

    Os pesos incrementais sao simplesmente a likelihood da observacao. Apos $T$ passos, o peso acumulado e $w_T = \prod_{t=1}^{T} g(y_t | x_t^{(i)})$, que degenera rapidamente.

---

## 3. Framework SMC

O framework SMC resolve o problema de degeneracao combinando **SIS + Resampling + Mutation**.

### 3.1 Sequencia de Distribuicoes Target

O framework geral de SMC opera sobre uma sequencia de targets:

$$
\pi_1, \pi_2, \ldots, \pi_T
$$

definidas em espacos possivelmente diferentes. Cada $\pi_t$ e conhecida ate uma constante: $\pi_t(x) = \tilde{\pi}_t(x) / Z_t$.

!!! info "Exemplos de Sequencias de Targets"
    | Aplicacao | $\pi_t$ |
    |---|---|
    | Particle filtering | $p(x_{0:t} \mid y_{1:t})$ |
    | SMC sampler (tempering) | $\pi_0^{1-\beta_t} \pi^{\beta_t}$, $0 = \beta_0 < \cdots < \beta_T = 1$ |
    | Data annealing | $p(\theta \mid y_{1:t})$ com dados adicionados sequencialmente |
    | Rare event simulation | $p(x \mid \varphi(x) > \gamma_t)$, $\gamma_t$ crescente |

### 3.2 Algoritmo SMC Generico

O algoritmo SMC opera em tres etapas a cada iteracao:

**Algoritmo: Sequential Monte Carlo**

Para $t = 1, \ldots, T$:

1. **Resampling** (condicional): se $\text{ESS}_t < N_{\text{threshold}}$, reamostrar indices $a_{t-1}^{(i)} \sim \text{Categorical}(W_{t-1}^{(1:N)})$
2. **Mutation**: propagar $x_t^{(i)} \sim K_t(x_t | x_{t-1}^{a_{t-1}^{(i)}})$
3. **Reweighting**: calcular pesos $\tilde{w}_t^{(i)} = \alpha_t(x_{t-1}^{a_{t-1}^{(i)}}, x_t^{(i)})$ e normalizar $W_t^{(i)} = \tilde{w}_t^{(i)} / \sum_j \tilde{w}_t^{(j)}$

### 3.3 Consistencia e CLT para Estimadores SMC

!!! abstract "Teorema: Consistencia do SMC (Del Moral, 2004)"
    Sob condicoes de regularidade (targets limitados, kernels de transicao mixing), para qualquer funcao teste limitada $\varphi$:

    $$
    \hat{\pi}_t^N(\varphi) = \sum_{i=1}^{N} W_t^{(i)} \varphi(x_t^{(i)}) \xrightarrow{a.s.} \pi_t(\varphi) = \int \varphi(x) \, \pi_t(x) \, dx
    $$

    quando $N \to \infty$.

??? note "Sketch da Prova: Consistencia"
    A prova segue por inducao em $t$:

    1. **Base** ($t=1$): IS classico garante consistencia por LGN
    2. **Passo indutivo**: Assumindo $\hat{\pi}_{t-1}^N \to \pi_{t-1}$:
        - Resampling preserva expectativas (unbiased)
        - Mutation via kernel $K_t$ e IS corrigido pelos pesos incrementais
        - LGN aplicada condicional a geracao anterior
    3. A composicao resampling + mutation + reweighting preserva a consistencia. $\blacksquare$

!!! abstract "Teorema: CLT para SMC (Chopin, 2004; Del Moral, 2004)"
    Sob condicoes de regularidade, para funcoes $\varphi$ com $\pi_t(\varphi^2) < \infty$:

    $$
    \sqrt{N}\left(\hat{\pi}_t^N(\varphi) - \pi_t(\varphi)\right) \xrightarrow{d} \mathcal{N}\left(0, \sigma_t^2(\varphi)\right)
    $$

    onde a **variancia assintotica** $\sigma_t^2(\varphi)$ admite a decomposicao:

    $$
    \sigma_t^2(\varphi) = \sum_{s=1}^{t} \sigma_{s|t}^2(\varphi)
    $$

    Cada termo $\sigma_{s|t}^2$ corresponde a contribuicao da variancia introduzida no passo $s$ para o erro no passo $t$.

### 3.4 Taxa de Convergencia

!!! abstract "Teorema: Taxa de Convergencia $O(1/\sqrt{N})$"
    Para qualquer funcao $\varphi$ limitada e qualquer $t$ fixo:

    $$
    \mathbb{E}\left[\left|\hat{\pi}_t^N(\varphi) - \pi_t(\varphi)\right|^2\right] \leq \frac{C_t \|\varphi\|_{\infty}^2}{N}
    $$

    onde $C_t$ depende de $t$ mas nao de $N$. Isto implica taxa de convergencia $O(1/\sqrt{N})$.

??? note "Sketch da Prova"
    1. Para IS puro: $\text{Var}(\hat{I}_N) = \sigma^2/N$ diretamente pela LGN
    2. Para SMC: a prova usa uma **decomposicao telescopica** do erro:

        $$
        \hat{\pi}_t^N(\varphi) - \pi_t(\varphi) = \sum_{s=1}^{t} \left[\hat{\Phi}_{s,t}^N(\varphi) - \Phi_{s,t}\hat{\pi}_{s-1}^N(\varphi)\right]
        $$

        onde $\Phi_{s,t}$ sao os operadores de propagacao do SMC.

    3. Cada termo da soma tem variancia $O(1/N)$ pelo CLT para IS
    4. A soma de $t$ termos preserva a taxa $O(1/N)$ com constante $C_t$ que pode crescer com $t$. $\blacksquare$

!!! warning "Dependencia em $t$"
    Embora a taxa em $N$ seja $O(1/\sqrt{N})$ para $t$ fixo, a constante $C_t$ pode crescer com $t$:

    - **Caso favoravel** (mixing forte): $C_t$ permanece limitado — estabilidade temporal
    - **Caso desfavoravel** (mixing fraco): $C_t$ pode crescer exponencialmente com $t$

    A **estabilidade temporal** do SMC e um dos resultados mais profundos da teoria (Del Moral & Guionnet, 2001).

---

## 4. Marginal Likelihood Estimation

### 4.1 Estimador Unbiased de $p(y_{1:T})$

Uma das propriedades mais notaveis do SMC e fornecer um estimador **nao-enviesado** da marginal likelihood (evidencia).

!!! abstract "Teorema: Estimador Unbiased da Marginal Likelihood (Del Moral, 2004)"
    O estimador SMC da marginal likelihood:

    $$
    \hat{p}(y_{1:T}) = \prod_{t=1}^{T} \left(\frac{1}{N} \sum_{i=1}^{N} \tilde{w}_t^{(i)}\right)
    $$

    e um estimador **nao-enviesado** de $p(y_{1:T})$:

    $$
    \mathbb{E}\left[\hat{p}(y_{1:T})\right] = p(y_{1:T})
    $$

??? note "Sketch da Prova"
    A prova segue de uma identidade telescopica. Defina $Z_t = p(y_{1:t})$ a normalizacao cumulativa.

    1. Para $t=1$: $\frac{1}{N}\sum_i g(y_1|x_1^{(i)})$ com $x_1^{(i)} \sim \mu$ e unbiased para $p(y_1) = \int g(y_1|x_1) \mu(dx_1)$

    2. Para $t > 1$: apos resampling, os ancestrais sao amostras aproximadas de $\pi_{t-1}$. Condicional aos ancestrais:

        $$
        \mathbb{E}\left[\frac{1}{N}\sum_{i=1}^{N} \tilde{w}_t^{(i)} \;\middle|\; x_{t-1}^{(1:N)}\right] = \frac{Z_t}{Z_{t-1}} + O(1/N)
        $$

    3. O resampling introduz um vies de $O(1/N)$ em cada passo, mas o **produto** de medias permanece unbiased:

        $$
        \mathbb{E}\left[\prod_{t=1}^{T}\frac{1}{N}\sum_{i=1}^{N} \tilde{w}_t^{(i)}\right] = \prod_{t=1}^{T} \frac{Z_t}{Z_{t-1}} = Z_T
        $$

    Esta prova utiliza a propriedade de que resampling preserva expectativas marginais. $\blacksquare$

### 4.2 Produto de Likelihood Incrementais

A marginal likelihood se decompoe naturalmente como um produto de **likelihood incrementais** (predictive likelihoods):

$$
p(y_{1:T}) = \prod_{t=1}^{T} p(y_t | y_{1:t-1})
$$

O estimador SMC explora esta decomposicao:

$$
\hat{p}(y_t | y_{1:t-1}) = \frac{1}{N} \sum_{i=1}^{N} \tilde{w}_t^{(i)}
$$

!!! tip "Log-Marginal Likelihood"
    Na pratica, trabalha-se com o logaritmo para estabilidade numerica:

    $$
    \log \hat{p}(y_{1:T}) = \sum_{t=1}^{T} \log\left(\frac{1}{N}\sum_{i=1}^{N} \tilde{w}_t^{(i)}\right)
    $$

    Note que $\log \hat{p}$ e **enviesado** (pela desigualdade de Jensen), mas o vies e $O(1/N)$.

### 4.3 Importancia para Model Selection e PMCMC

!!! info "Aplicacoes do Estimador de Marginal Likelihood"
    **Model Selection (Bayes Factors)**:

    Para comparar modelos $\mathcal{M}_1$ e $\mathcal{M}_2$, o Bayes factor e:

    $$
    \text{BF}_{12} = \frac{p(y_{1:T} | \mathcal{M}_1)}{p(y_{1:T} | \mathcal{M}_2)} \approx \frac{\hat{p}_1(y_{1:T})}{\hat{p}_2(y_{1:T})}
    $$

    SMC fornece estimativas diretamente comparaveis sem necessidade de metodos adicionais.

    **Particle MCMC (PMCMC)**:

    O estimador unbiased e a base do **Particle Marginal Metropolis-Hastings (PMMH)** (Andrieu et al., 2010). Como $\hat{p}(y_{1:T}|\theta)$ e unbiased, substituir a likelihood verdadeira pelo estimador SMC no acceptance ratio de MH produz uma cadeia MCMC com distribuicao estacionaria **exata**:

    $$
    \alpha(\theta^* | \theta) = 1 \wedge \frac{\hat{p}(y_{1:T}|\theta^*) \, p(\theta^*)}{\hat{p}(y_{1:T}|\theta) \, p(\theta)}
    $$

    Este resultado notavel (pseudo-marginal approach) garante inferencia exata mesmo com likelihood estimada.

---

## Referencias

| Referencia | Contribuicao |
|---|---|
| Del Moral (2004). *Feynman-Kac Formulae* | Framework teorico geral para SMC, consistencia e CLT |
| Chopin (2004). *Central limit theorem for SMC* | CLT preciso para estimadores SMC |
| Doucet, de Freitas & Gordon (2001). *Sequential Monte Carlo Methods in Practice* | Referencia abrangente sobre SMC e particle filters |
| Kong, Liu & Wong (1994). *Sequential imputations and Bayesian missing data problems* | Degeneracao dos pesos em SIS |
| Del Moral & Guionnet (2001). *On the stability of interacting processes* | Estabilidade temporal de SMC |
| Andrieu, Doucet & Holenstein (2010). *Particle MCMC methods* | PMCMC e pseudo-marginal approach |
| Liu (2001). *Monte Carlo Strategies in Scientific Computing* | Importance sampling e metodos Monte Carlo |
| Cappe, Moulines & Ryden (2005). *Inference in HMMs* | Inferencia em modelos de Markov ocultos |
