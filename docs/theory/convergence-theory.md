---
title: Convergencia de Metodos SMC
description: Teoria de convergencia para particle filters - Law of Large Numbers, Central Limit Theorem, stability, curse of dimensionality e finite-sample bounds.
---

# Convergencia de Metodos SMC

Esta pagina apresenta os resultados de **convergencia** para metodos Sequential Monte Carlo, incluindo a Lei dos Grandes Numeros, o Teorema Central do Limite, estabilidade, maldimensionalidade e bounds finitos. Estes resultados fornecem as garantias teoricas que fundamentam o uso pratico de particle filters.

---

## 1. Law of Large Numbers para Particle Filters

### 1.1 Estimador por Particulas

O particle filter aproxima a distribuicao filtering $\pi_t(x_t) = p(x_t | y_{1:t})$ por uma medida empirica ponderada:

$$
\hat{\pi}_t^N(dx_t) = \sum_{i=1}^{N} W_t^{(i)} \, \delta_{x_t^{(i)}}(dx_t)
$$

onde $\{x_t^{(i)}, W_t^{(i)}\}_{i=1}^N$ sao as particulas e seus pesos normalizados. Para qualquer funcao teste $\varphi$ integravel, o estimador por particulas e:

$$
\hat{I}_t^N[\varphi] = \sum_{i=1}^{N} W_t^{(i)} \, \varphi(x_t^{(i)}) \approx \int \varphi(x_t) \, \pi_t(x_t) \, dx_t = \mathbb{E}_{\pi_t}[\varphi(x_t)]
$$

### 1.2 LLN para Particle Filters

!!! abstract "Teorema: Lei dos Grandes Numeros (Crisan & Doucet, 2002)"
    Seja $\{x_t^{(i)}, W_t^{(i)}\}_{i=1}^N$ o sistema de particulas gerado por um particle filter com resampling multinomial. Se:

    1. A funcao teste $\varphi$ e **limitada**: $\|\varphi\|_\infty < \infty$
    2. Os pesos de importance sampling sao **uniformemente limitados**: $\sup_x \frac{\pi_t(x)}{q_t(x)} < \infty$
    3. O modelo e **estavel**: as densidades de transicao e observacao sao positivas e limitadas

    Entao, para cada $t \geq 0$:

    $$
    \hat{I}_t^N[\varphi] = \sum_{i=1}^{N} W_t^{(i)} \, \varphi(x_t^{(i)}) \xrightarrow{a.s.} \mathbb{E}_{\pi_t}[\varphi(x_t)] \quad \text{quando } N \to \infty
    $$

??? note "Sketch da Prova: LLN via Decomposicao Sequencial"
    A prova procede por inducao em $t$, decompondo o erro em dois componentes: **sampling** e **resampling**.

    **Passo 1: Caso base ($t=0$).**

    Para $t=0$, as particulas $x_0^{(i)} \sim q_0$ sao i.i.d., e o estimador de importance sampling padrao satisfaz a LLN classica:

    $$
    \frac{1}{N}\sum_{i=1}^{N} w_0^{(i)} \varphi(x_0^{(i)}) \xrightarrow{a.s.} \mathbb{E}_{\pi_0}[\varphi(x_0)]
    $$

    **Passo 2: Passo indutivo.**

    Assumindo que $\hat{\pi}_{t-1}^N \xrightarrow{w} \pi_{t-1}$ quando $N \to \infty$, mostramos que:

    - **Resampling**: apos resampling multinomial, as particulas resampleadas $\{\bar{x}_{t-1}^{(i)}\}$ sao condicionalmente i.i.d. dada $\hat{\pi}_{t-1}^N$, e pela LLN condicional, $\frac{1}{N}\sum_i \psi(\bar{x}_{t-1}^{(i)}) \xrightarrow{a.s.} \hat{\pi}_{t-1}^N[\psi]$
    - **Propagacao + Ponderacao**: propagando via $q_t(\cdot | \bar{x}_{t-1}^{(i)})$ e ponderando por $w_t^{(i)}$, o novo estimador converge para $\pi_t[\varphi]$

    O passo chave e que a composicao de duas operacoes consistentes (resampling e importance sampling) preserva consistencia.

    **Passo 3: Controle dos pesos.**

    A condicao $\sup_x \pi_t(x)/q_t(x) < \infty$ garante que os pesos sao limitados, prevenindo que uma unica particula domine o estimador. $\blacksquare$

### 1.3 Taxa de Convergencia

A taxa de convergencia do estimador por particulas e $O_p(N^{-1/2})$ para cada $t$ fixo:

$$
\hat{I}_t^N[\varphi] - \mathbb{E}_{\pi_t}[\varphi(x_t)] = O_p\left(\frac{1}{\sqrt{N}}\right)
$$

!!! warning "Dependencia em $t$"
    A constante implicita no $O_p(N^{-1/2})$ pode crescer com $t$. A questao crucial e **como** essa constante cresce — se linearmente, polinomialmente ou exponencialmente em $t$. Os resultados de estabilidade (Secao 3) mostram que sob condicoes de mixing, a constante permanece **limitada** uniformemente em $t$.

### 1.4 LLN para Marginal Likelihood

Um resultado particularmente importante e a convergencia do estimador da marginal likelihood:

!!! abstract "Teorema: Unbiasedness da Marginal Likelihood"
    O estimador da marginal likelihood produzido pelo particle filter:

    $$
    \hat{p}^N(y_{1:T}) = \prod_{t=1}^{T} \left(\frac{1}{N}\sum_{i=1}^{N} w_t^{(i)}\right)
    $$

    e um estimador **nao-enviesado** de $p(y_{1:T})$:

    $$
    \mathbb{E}\left[\hat{p}^N(y_{1:T})\right] = p(y_{1:T})
    $$

    Alem disso, $\hat{p}^N(y_{1:T}) \xrightarrow{a.s.} p(y_{1:T})$ quando $N \to \infty$.

Este resultado e a base do framework **pseudo-marginal** usado em PMCMC.

---

## 2. Central Limit Theorem

### 2.1 CLT para Particle Filters

!!! abstract "Teorema: CLT para Particle Filters (Chopin, 2004; Del Moral, 2004)"
    Sob as condicoes do Teorema LLN, e assumindo adicionalmente que $\mathbb{E}_{\pi_t}[\varphi^2(x_t)] < \infty$, temos:

    $$
    \sqrt{N}\left(\hat{I}_t^N[\varphi] - \mathbb{E}_{\pi_t}[\varphi(x_t)]\right) \xrightarrow{d} \mathcal{N}\left(0, \sigma_t^2(\varphi)\right)
    $$

    onde $\sigma_t^2(\varphi)$ e a **variancia assintotica**.

### 2.2 Variancia Assintotica

A variancia assintotica $\sigma_t^2(\varphi)$ depende do tipo de resampling e da qualidade da proposal. Para o Bootstrap Particle Filter com resampling a cada passo:

$$
\sigma_t^2(\varphi) = \sum_{s=0}^{t} \text{Var}_{\pi_s}\left[\mathbb{E}_{\pi_{s:t}}\left[\varphi(x_t) \, | \, x_s\right] \cdot \frac{g_s(y_s | x_s)}{p(y_s | y_{1:s-1})}\right]
$$

onde $\pi_{s:t}$ denota a distribuicao de $x_t$ dado $x_s$ sob o modelo.

!!! info "Decomposicao da Variancia"
    A variancia assintotica pode ser decomposta em contribuicoes de cada passo temporal:

    $$
    \sigma_t^2(\varphi) = \underbrace{\sigma_{t,\text{IS}}^2}_{\text{importance sampling}} + \underbrace{\sigma_{t,\text{R}}^2}_{\text{resampling}}
    $$

    - **$\sigma_{t,\text{IS}}^2$**: variancia do importance sampling, determinada pela qualidade da proposal $q_t$
    - **$\sigma_{t,\text{R}}^2$**: variancia adicional introduzida pelo resampling, proporcional a $\text{Var}_{\pi_t}[\varphi(x_t)]$

### 2.3 Estimacao da Variancia Assintotica

Na pratica, $\sigma_t^2(\varphi)$ pode ser estimada de varias formas:

=== "Estimador Plug-in"

    Substituindo esperancas populacionais por medias amostrais:

    $$
    \hat{\sigma}_t^2 = \sum_{i=1}^{N} W_t^{(i)} \left[\varphi(x_t^{(i)}) - \hat{I}_t^N[\varphi]\right]^2
    $$

    Este estimador subestima a variancia verdadeira pois ignora a contribuicao do resampling.

=== "Estimador por Genealogia"

    Usando a genealogia das particulas (ancestor paths):

    $$
    \hat{\sigma}_t^2 = \sum_{s=0}^{t} \frac{1}{N}\sum_{i=1}^{N} W_s^{(i)} \left[\hat{\varphi}_{s:t}(x_s^{(i)}) - \hat{I}_t^N[\varphi]\right]^2
    $$

    onde $\hat{\varphi}_{s:t}(x_s^{(i)})$ e a media condicional estimada de $\varphi(x_t)$ dado o ancestral $x_s^{(i)}$.

=== "Estimador por Replicacao"

    Executando $R$ particle filters independentes e usando a variancia amostral:

    $$
    \hat{\sigma}_t^2 = \frac{N}{R-1} \sum_{r=1}^{R} \left[\hat{I}_{t,r}^N[\varphi] - \bar{I}_t[\varphi]\right]^2
    $$

    Este e o metodo mais robusto mas requer $R$ execucoes independentes.

### 2.4 Dependencia da Variancia com $T$

!!! warning "Crescimento da Variancia"
    Para um particle filter **sem** propriedades de estabilidade, a variancia assintotica pode crescer como:

    $$
    \sigma_T^2(\varphi) = O(T)
    $$

    Isto significa que para sequencias longas, a precisao do estimador degrada. A condicao para que $\sigma_T^2$ permaneca limitada e que o modelo satisfaca as propriedades de **forgetting** discutidas na Secao 3.

### 2.5 CLT para Log-Marginal Likelihood

!!! abstract "Teorema: CLT para Log-Marginal Likelihood"
    O estimador de log-marginal likelihood satisfaz:

    $$
    \sqrt{N}\left(\log \hat{p}^N(y_{1:T}) - \log p(y_{1:T})\right) \xrightarrow{d} \mathcal{N}\left(0, \sum_{t=1}^{T} \sigma_{t,\text{ML}}^2\right)
    $$

    onde $\sigma_{t,\text{ML}}^2$ depende da variabilidade dos pesos no passo $t$. A variancia total cresce **linearmente** com $T$, o que tem implicacoes para a eficiencia de PMCMC em series longas.

---

## 3. Stability e Forgetting

### 3.1 Particle Filter como Sistema Dinamico

O particle filter pode ser visto como um sistema dinamico que mapeia uma medida empirica $\hat{\pi}_{t-1}^N$ para $\hat{\pi}_t^N$ atraves de tres operacoes:

$$
\hat{\pi}_{t-1}^N \xrightarrow{\text{resampling}} \bar{\pi}_{t-1}^N \xrightarrow{\text{propagacao}} \hat{\mu}_t^N \xrightarrow{\text{ponderacao}} \hat{\pi}_t^N
$$

A questao de estabilidade e: **perturbacoes na condicao inicial $\hat{\pi}_0^N$ desaparecem com o tempo?**

### 3.2 Forgetting Property

!!! abstract "Teorema: Forgetting do Filtro Otimo (Del Moral & Guionnet, 2001)"
    Considere o filtro otimo (populacao infinita) com duas condicoes iniciais $\pi_0$ e $\pi_0'$. Se o modelo satisfaz as condicoes de mixing (Secao 3.3), entao:

    $$
    \|\pi_t(\cdot | y_{1:t}) - \pi_t'(\cdot | y_{1:t})\|_{TV} \leq C \, \rho^t
    $$

    onde $\|\cdot\|_{TV}$ e a distancia em variacao total, $C > 0$ e uma constante, e $0 < \rho < 1$ e a taxa de esquecimento.

Isto significa que o efeito da condicao inicial **desvanece exponencialmente** — o filtro "esquece" sua inicializacao.

??? note "Sketch da Prova: Forgetting via Acoplamento"
    **Construcao de Acoplamento:**

    Definimos dois processos de filtragem identicos exceto na condicao inicial: $\pi_0$ e $\pi_0'$. Construimos um acoplamento $(X_t, X_t')$ onde $X_t \sim \pi_t$ e $X_t' \sim \pi_t'$.

    **Passo 1: Probabilidade de coalescencia.**

    Sob a condicao de mixing forte, em cada passo temporal existe probabilidade $\epsilon > 0$ de que $X_t = X_t'$ (coalescencia), independentemente dos estados anteriores:

    $$
    \mathbb{P}(X_t = X_t' | X_{t-1} \neq X_{t-1}') \geq \epsilon
    $$

    **Passo 2: Desigualdade de acoplamento.**

    Pelo coupling inequality:

    $$
    \|\pi_t - \pi_t'\|_{TV} \leq \mathbb{P}(X_t \neq X_t') \leq (1 - \epsilon)^t
    $$

    Portanto $\rho = 1 - \epsilon$, e o forgetting e exponencial. $\blacksquare$

### 3.3 Mixing Conditions

As condicoes que garantem estabilidade envolvem restricoes sobre o kernel de transicao e a funcao de observacao:

!!! info "Condicao de Mixing Forte"
    Existe $\epsilon > 0$ e uma medida de referencia $\lambda$ tal que para todo $x$:

    $$
    \epsilon \, \lambda(A) \leq \int_A f_\theta(x' | x) \, dx' \leq \epsilon^{-1} \, \lambda(A)
    $$

    para todo conjunto mensuravel $A$. Equivalentemente, o kernel de transicao e **uniformemente ergodico**.

!!! info "Condicao sobre a Observacao"
    A funcao de verossimilhanca da observacao e limitada acima e abaixo:

    $$
    0 < g_{\min} \leq g_\theta(y_t | x_t) \leq g_{\max} < \infty
    $$

    para todo $x_t$ e $y_t$.

=== "Exemplos que satisfazem"

    - **Modelo linear-gaussiano**: $x_t = Ax_{t-1} + \eta_t$, $y_t = Cx_t + \varepsilon_t$ com $\|A\| < 1$
    - **Stochastic Volatility**: $h_t = \mu + \phi(h_{t-1} - \mu) + \sigma_\eta \eta_t$ com $|\phi| < 1$
    - **Modelos com ruido de estado limitado inferiormente**

=== "Exemplos que NAO satisfazem"

    - **Modelos com absorcao**: estados que uma vez alcancados nao podem ser deixados
    - **Modelos deterministicos**: $f_\theta(x'|x) = \delta_{h(x)}(x')$ (sem ruido de processo)
    - **Modelos com observacao degenerada**: $g_\theta(y|x) = 0$ para regioes do estado

### 3.4 Bound de Erro Uniforme em $T$

!!! abstract "Teorema: Erro Uniforme (Del Moral, 2004)"
    Sob as condicoes de mixing forte, para toda funcao teste $\varphi$ com $\|\varphi\|_\infty \leq 1$:

    $$
    \sup_{t \geq 0} \, \mathbb{E}\left[\left|\hat{I}_t^N[\varphi] - \mathbb{E}_{\pi_t}[\varphi]\right|^p\right]^{1/p} \leq \frac{C_p}{\sqrt{N}}
    $$

    onde $C_p$ depende de $p$ e das constantes de mixing, mas **nao de $t$**.

??? note "Sketch da Prova: Bound Uniforme"
    A prova combina a decomposicao do erro com a propriedade de forgetting:

    **Passo 1: Decomposicao temporal.**

    O erro no tempo $t$ pode ser decomposto como soma de contribuicoes de cada passo:

    $$
    \hat{I}_t^N[\varphi] - \pi_t[\varphi] = \sum_{s=0}^{t} \Delta_{s,t}^N
    $$

    onde $\Delta_{s,t}^N$ representa a contribuicao do erro de amostragem no passo $s$ ao erro total no passo $t$.

    **Passo 2: Decaimento exponencial.**

    Pelo forgetting, a contribuicao de erros passados decai exponencialmente:

    $$
    \|\Delta_{s,t}^N\|_p \leq \frac{C}{\sqrt{N}} \rho^{t-s}
    $$

    **Passo 3: Soma geometrica.**

    $$
    \|\hat{I}_t^N[\varphi] - \pi_t[\varphi]\|_p \leq \frac{C}{\sqrt{N}} \sum_{s=0}^{t} \rho^{t-s} \leq \frac{C}{\sqrt{N}} \cdot \frac{1}{1-\rho}
    $$

    O bound e independente de $t$, pois a serie geometrica converge. $\blacksquare$

!!! success "Implicacao Pratica"
    O bound uniforme garante que um particle filter com $N$ particulas fixo pode rodar **indefinidamente** sem perda de precisao, desde que o modelo satisfaca as condicoes de mixing. Isto e fundamental para aplicacoes em tempo real e series temporais longas.

---

## 4. Curse of Dimensionality

### 4.1 O Problema Fundamental

O particle filter sofre de uma **maldimensionalidade** (curse of dimensionality): o numero de particulas $N$ necessario para manter uma precisao fixa cresce **exponencialmente** com a dimensao $d_x$ do espaco de estados.

!!! abstract "Teorema: Maldimensionalidade (Bengtsson, Bickel & Li, 2008; Snyder et al., 2008)"
    Considere um modelo de estado-espaco com estado $x_t \in \mathbb{R}^{d_x}$. Para o Bootstrap Particle Filter, o ESS satisfaz:

    $$
    \frac{\text{ESS}}{N} \xrightarrow{d_x \to \infty} 0 \quad \text{quando } N = o\left(\exp(c \, d_x)\right)
    $$

    para alguma constante $c > 0$ que depende do modelo. Ou seja, se $N$ nao cresce exponencialmente com $d_x$, o ESS colapsa para zero.

### 4.2 Intuicao e Mecanismo

A maldimensionalidade surge porque em altas dimensoes, a probabilidade de uma particula proposta estar na regiao de alta densidade da posterior torna-se negligivel:

$$
\text{Var}_q[w(x)] \propto \exp(c \cdot d_x) - 1
$$

!!! example "Exemplo: Modelo Gaussiano"
    Para $x_t \sim \mathcal{N}(Ax_{t-1}, Q)$ e $y_t \sim \mathcal{N}(Cx_t, R)$ com $x_t \in \mathbb{R}^{d_x}$:

    - A prior predictive $p(x_t | x_{t-1})$ e centrada em $Ax_{t-1}$
    - A posterior filtering $p(x_t | x_{t-1}, y_t) \propto p(y_t|x_t) p(x_t|x_{t-1})$ e deslocada em direcao a $y_t$
    - O deslocamento medio entre prior e posterior cresce com $d_x$ como $\sqrt{d_x}$
    - A sobreposicao entre as duas distribuicoes decai exponencialmente:

    $$
    \text{ESS}_{\text{efetivo}} \approx N \cdot \exp\left(-\frac{d_x}{2} \cdot \text{KL}(p_{\text{posterior}} \| p_{\text{prior}})\right)
    $$

### 4.3 Provas e Bounds

??? note "Prova: Colapso dos Pesos em Alta Dimensao"
    **Setup**: Considere o modelo $x_t = A x_{t-1} + \eta_t$ com $\eta_t \sim \mathcal{N}(0, Q)$ e $y_t = Cx_t + \varepsilon_t$ com $\varepsilon_t \sim \mathcal{N}(0, R)$. Seja $d_x$ a dimensao do estado.

    **Passo 1: Distribuicao do log-peso.**

    Para o Bootstrap PF, o peso incremental e $w_t^{(i)} = p(y_t | x_t^{(i)})$ com $x_t^{(i)} \sim p(x_t | x_{t-1}^{(i)})$. O log-peso e:

    $$
    \log w_t^{(i)} = -\frac{1}{2}\left[(y_t - Cx_t^{(i)})^\top R^{-1} (y_t - Cx_t^{(i)}) + d_y \log(2\pi) + \log|R|\right]
    $$

    **Passo 2: Momento do log-peso.**

    Sob a prior, $Cx_t^{(i)} \sim \mathcal{N}(CA x_{t-1}^{(i)}, CQC^\top)$ e:

    $$
    \text{Var}\left[\log w_t^{(i)}\right] = \frac{1}{2}\text{tr}\left[(R^{-1}CQC^\top)^2\right] + (y_t - CA x_{t-1}^{(i)})^\top (R + CQC^\top)^{-1} CQC^\top (R + CQC^\top)^{-1} R^{-1} (\cdot)
    $$

    Quando $d_x \to \infty$ (e $d_y = d_x$), ambos os termos crescem como $O(d_x)$.

    **Passo 3: Colapso do ESS.**

    Pelo CLT, $\log w_t^{(i)} \approx \mathcal{N}(\mu_w, \sigma_w^2)$ com $\sigma_w^2 = O(d_x)$. Para pesos log-normais:

    $$
    \frac{\text{ESS}}{N} \approx \frac{1}{1 + \text{Var}[w/\mathbb{E}[w]]} = \frac{1}{1 + e^{\sigma_w^2} - 1} = e^{-\sigma_w^2} = e^{-O(d_x)}
    $$

    Portanto, ESS$/N \to 0$ exponencialmente em $d_x$. $\blacksquare$

### 4.4 Estrategias de Mitigacao

=== "Rao-Blackwellized PF (RBPF)"

    Quando parte do estado admite filtragem analitica (e.g., via Kalman filter), o RBPF reduz a dimensao efetiva:

    $$
    x_t = \begin{pmatrix} x_t^{(1)} \\ x_t^{(2)} \end{pmatrix}, \quad p(x_t | y_{1:t}) = p(x_t^{(1)} | y_{1:t}) \cdot p(x_t^{(2)} | x_t^{(1)}, y_{1:t})
    $$

    - $x_t^{(2)} | x_t^{(1)}$ e filtrado analiticamente (e.g., via `kalmanbox`)
    - Particulas sao usadas apenas para $x_t^{(1)} \in \mathbb{R}^{d_1}$ com $d_1 \ll d_x$

    !!! success "Reducao de Variancia"
        O RBPF reduz a variancia por um fator exponencial:

        $$
        \text{Var}_{\text{RBPF}} = O\left(e^{c \cdot d_1}\right) \quad \text{vs} \quad \text{Var}_{\text{BPF}} = O\left(e^{c \cdot d_x}\right)
        $$

=== "Proposals Locais"

    Usar proposals que incorporam a observacao $y_t$:

    $$
    q_t(x_t | x_{t-1}, y_t) \approx p(x_t | x_{t-1}, y_t) \propto g(y_t | x_t) f(x_t | x_{t-1})
    $$

    - **Optimal proposal** (quando tratavel): $q_t^* = p(x_t | x_{t-1}, y_t)$ — minimiza a variancia dos pesos
    - **Extended Kalman proposal**: lineariza $g$ e $f$ para obter uma aproximacao gaussiana
    - **Unscented proposal**: usa sigma points para capturar nao-linearidades

    A reducao na variancia dos pesos pode mudar a dependencia dimensional de exponencial para **polinomial** em casos favoraveis.

=== "Tempering / Bridging"

    Introduz distribuicoes intermediarias entre a prior e a posterior:

    $$
    \pi_t^{(k)}(x_t) \propto g(y_t | x_t)^{\gamma_k} \, f(x_t | x_{t-1}), \quad 0 = \gamma_0 < \gamma_1 < \cdots < \gamma_K = 1
    $$

    Cada passo de tempering envolve um "deslocamento" menor no espaco de peso, reduzindo a variancia:

    $$
    \text{Var}[w^{(k)}] = O\left(e^{c \cdot d_x \cdot (\gamma_k - \gamma_{k-1})}\right)
    $$

    Escolhendo $K = O(d_x)$ passos de tempering, a variancia total permanece controlada.

=== "Block Sampling"

    Atualizar blocos de coordenadas do estado separadamente, reduzindo a dimensao efetiva de cada passo de importance sampling:

    $$
    x_t = (x_t^{[1]}, \ldots, x_t^{[B]}), \quad q_t(x_t | \cdot) = \prod_{b=1}^{B} q_t^{[b]}(x_t^{[b]} | \cdot)
    $$

    Cada bloco e de dimensao $d_x / B$, reduzindo a variancia dos pesos.

---

## 5. Finite-Sample Bounds

### 5.1 Bounds Nao-Assintoticos

Os resultados assintoticos (LLN, CLT) garantem convergencia quando $N \to \infty$, mas na pratica $N$ e finito. Bounds nao-assintoticos fornecem garantias para **qualquer** $N$.

!!! abstract "Teorema: Concentracao do Estimador por Particulas (Del Moral, 2004)"
    Sob as condicoes de mixing, para qualquer funcao teste $\varphi$ com $\|\varphi\|_\infty \leq 1$ e para todo $\epsilon > 0$:

    $$
    \mathbb{P}\left(\left|\hat{I}_t^N[\varphi] - \mathbb{E}_{\pi_t}[\varphi]\right| > \epsilon\right) \leq 2 \exp\left(-\frac{N \epsilon^2}{C_t^2}\right)
    $$

    onde $C_t$ e uma constante que depende do modelo mas e **uniforme em $t$** sob mixing.

??? note "Prova: Concentracao via Desigualdade de Hoeffding Condicional"
    **Passo 1: Martingale decomposition.**

    Definimos a filtracao $\mathcal{F}_s$ gerada pelas particulas ate o passo $s$. O erro pode ser escrito como soma de diferencas de martingale:

    $$
    \hat{I}_t^N[\varphi] - \pi_t[\varphi] = \sum_{s=0}^{t} D_s^N, \quad \mathbb{E}[D_s^N | \mathcal{F}_{s-1}] = 0
    $$

    **Passo 2: Bounds para cada incremento.**

    Cada $D_s^N$ e uma media de $N$ termos limitados (apos ponderacao e forgetting), portanto:

    $$
    |D_s^N| \leq \frac{C \rho^{t-s}}{\sqrt{N}}, \quad \text{Var}(D_s^N | \mathcal{F}_{s-1}) \leq \frac{C^2 \rho^{2(t-s)}}{N}
    $$

    **Passo 3: Aplicacao da desigualdade de Azuma-Hoeffding.**

    Para a soma de martingale com incrementos limitados:

    $$
    \mathbb{P}\left(\left|\sum_{s=0}^t D_s^N\right| > \epsilon\right) \leq 2\exp\left(-\frac{\epsilon^2}{2\sum_{s=0}^t C^2\rho^{2(t-s)}/N}\right) \leq 2\exp\left(-\frac{N\epsilon^2}{2C^2/(1-\rho^2)}\right)
    $$

    O bound e uniforme em $t$ gracas a convergencia da serie geometrica. $\blacksquare$

### 5.2 Concentracao para ESS

O Effective Sample Size concentra-se em torno de seu valor esperado:

$$
\mathbb{P}\left(\left|\frac{\text{ESS}}{N} - \frac{1}{1 + \chi^2(\pi_t \| q_t)}\right| > \epsilon\right) \leq 2\exp\left(-c N \epsilon^2\right)
$$

onde $\chi^2(\pi_t \| q_t)$ e a divergencia chi-quadrado entre target e proposal.

### 5.3 Concentracao para Marginal Likelihood

!!! abstract "Teorema: Concentracao da Log-Marginal Likelihood"
    Para o estimador de log-marginal likelihood:

    $$
    \mathbb{P}\left(\left|\log \hat{p}^N(y_{1:T}) - \log p(y_{1:T})\right| > \epsilon\right) \leq 2T \exp\left(-\frac{N \epsilon^2}{C^2 T}\right)
    $$

    Para obter erro $\leq \epsilon$ com probabilidade $\geq 1 - \delta$, necessitamos:

    $$
    N \geq \frac{C^2 T}{\epsilon^2} \log\left(\frac{2T}{\delta}\right)
    $$

### 5.4 Aplicacao Pratica: Escolha de $N$

!!! tip "Guia Pratico para Escolha de $N$"
    Combinando os bounds teoricos com heuristicas praticas:

    | Criterio | $N$ Minimo | Fundamento |
    |----------|-----------|------------|
    | ESS estavel | $N$ tal que $\text{ESS} \geq N/2$ | Evitar degeneracy |
    | Erro de filtragem $\leq \epsilon$ | $N \geq C_t^2 / \epsilon^2$ | Concentracao |
    | Log-ML com erro $\leq \epsilon$ | $N \geq C^2 T / \epsilon^2$ | Concentracao ML |
    | PMCMC eficiente | $N$ tal que $\text{Var}[\log \hat{p}] \approx 1$ | Doucet et al., 2015 |
    | Alta dimensao ($d_x$ grande) | $N \geq \exp(c \cdot d_x^{\text{eff}})$ | Curse of dimensionality |

    Onde $d_x^{\text{eff}}$ e a dimensao efetiva apos Rao-Blackwellization ou proposals otimas.

!!! example "Regra Pratica"
    Para modelos de volatilidade estocastica ($d_x = 1$):

    - **Filtragem**: $N = 100$-$500$ e tipicamente suficiente
    - **PMCMC**: $N = 500$-$2000$ para $\text{Var}[\log \hat{p}] \approx 1$-$3$
    - **Comparacao de modelos**: $N = 2000$-$5000$ para marginal likelihood precisa

    Para modelos DSGE ($d_x \sim 5$-$20$):

    - **Com RBPF** (usando `kalmanbox`): $N = 500$-$2000$
    - **Sem RBPF**: $N$ pode precisar ser $10^4$-$10^5$ dependendo de $d_x$

---

## Resumo dos Resultados

| Resultado | Condicao | Rate | Uniforme em $T$? |
|-----------|----------|------|-------------------|
| LLN | Pesos limitados, modelo estavel | $N^{-1/2}$ | Sob mixing |
| CLT | Momentos finitos | $N^{-1/2}$ | Variancia pode crescer |
| Stability / Forgetting | Mixing forte | $\rho^t$ esquecimento | Sim |
| Curse of Dimensionality | — | $\exp(c \cdot d_x)$ particulas | — |
| Concentracao | Mixing | $\exp(-N\epsilon^2/C^2)$ | Sim |

---

## Referencias

!!! quote "Referencias Principais"
    - **Del Moral, P.** (2004). *Feynman-Kac Formulae: Genealogical and Interacting Particle Systems with Applications*. Springer.
    - **Chopin, N.** (2004). Central limit theorem for sequential Monte Carlo methods and its application to Bayesian inference. *Annals of Statistics*, 32(6), 2385-2411.
    - **Crisan, D. & Doucet, A.** (2002). A survey of convergence results on particle filtering methods for practitioners. *IEEE Transactions on Signal Processing*, 50(3), 736-746.
    - **Del Moral, P. & Guionnet, A.** (2001). On the stability of interacting processes with applications to filtering and genetic algorithms. *Annales de l'Institut Henri Poincare*, 37(2), 155-194.
    - **Bengtsson, T., Bickel, P. & Li, B.** (2008). Curse-of-dimensionality revisited: Collapse of the particle filter in very large scale systems. *IMS Collections*, 2, 316-334.
    - **Snyder, C., Bengtsson, T., Bickel, P. & Anderson, J.** (2008). Obstacles to high-dimensional particle filtering. *Monthly Weather Review*, 136(12), 4629-4640.
    - **Doucet, A., Pitt, M., Deligiannidis, G. & Kohn, R.** (2015). Efficient implementation of Markov chain Monte Carlo when using an unbiased likelihood estimator. *Biometrika*, 102(2), 295-313.
