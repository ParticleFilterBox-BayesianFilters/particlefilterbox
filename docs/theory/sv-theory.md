---
title: Teoria de Modelos de Volatilidade Estocastica
description: Teoria de modelos SV - derivacao, propriedades estatisticas, filtragem por particle filters, estimacao via PMCMC, e extensoes (leverage, fat tails, jumps).
---

# Teoria de Modelos de Volatilidade Estocastica

Esta pagina desenvolve a teoria de modelos de **Volatilidade Estocastica (SV)**, desde a derivacao a partir de principios de no-arbitrage ate a estimacao via metodos de particle MCMC. Modelos SV sao a aplicacao canonica de particle filters em financas.

---

## 1. Modelo SV Basico

### 1.1 Formulacao

O modelo de volatilidade estocastica basico (Taylor, 1986; Ghysels, Harvey & Renault, 1996) e definido como:

$$
\boxed{
\begin{aligned}
y_t &= \exp\left(\frac{h_t}{2}\right) \varepsilon_t, & \varepsilon_t &\sim \mathcal{N}(0, 1) \\
h_t &= \mu + \phi(h_{t-1} - \mu) + \sigma_\eta \eta_t, & \eta_t &\sim \mathcal{N}(0, 1)
\end{aligned}
}
$$

onde:

- $y_t$: retorno observado no periodo $t$
- $h_t$: **log-volatilidade** latente (estado nao-observavel)
- $\mu$: nivel medio da log-volatilidade
- $\phi \in (-1, 1)$: persistencia da volatilidade
- $\sigma_\eta > 0$: volatilidade da volatilidade

Os parametros sao $\theta = (\mu, \phi, \sigma_\eta)$.

### 1.2 Derivacao a partir de No-Arbitrage

!!! info "De Tempo Continuo para Tempo Discreto"
    O modelo SV discreto pode ser derivado como **discretizacao** do modelo de volatilidade estocastica em tempo continuo de Heston (1993).

    Em tempo continuo, o processo de precos $S_t$ e volatilidade $V_t$ seguem:

    $$
    \begin{aligned}
    \frac{dS_t}{S_t} &= \mu_S \, dt + \sqrt{V_t} \, dB_t^S \\
    dV_t &= \kappa(\bar{V} - V_t) \, dt + \sigma_V \sqrt{V_t} \, dB_t^V
    \end{aligned}
    $$

    Definindo $h_t = \log V_t$ e discretizando por Euler com passo $\Delta t = 1$:

    $$
    \begin{aligned}
    y_t &= \log S_t - \log S_{t-1} \approx \mu_S + \sqrt{V_t} \, \varepsilon_t \\
    h_t &\approx h_{t-1} + \kappa(\bar{h} - h_{t-1}) + \sigma_h \eta_t
    \end{aligned}
    $$

    onde $\bar{h} = \log \bar{V}$ e $\sigma_h = \sigma_V / \sqrt{V_t}$. Reparametrizando:
    $\mu = \bar{h}$, $\phi = 1 - \kappa$, $\sigma_\eta = \sigma_h$, obtemos o modelo SV discreto.

    A condicao de **no-arbitrage** garante a existencia de uma medida martingale equivalente, sob a qual $S_t$ descontada e um martingale.

### 1.3 Relacao com GARCH

!!! note "SV vs GARCH"
    Ambos os modelos capturam volatilidade variante no tempo, mas de formas fundamentalmente diferentes:

    | Aspecto | GARCH(1,1) | SV |
    |---------|-----------|-----|
    | Volatilidade | $\sigma_t^2 = \omega + \alpha y_{t-1}^2 + \beta \sigma_{t-1}^2$ | $h_t = \mu + \phi(h_{t-1}-\mu) + \sigma_\eta \eta_t$ |
    | Fonte de aleatoriedade | 1 choque ($\varepsilon_t$) | 2 choques ($\varepsilon_t, \eta_t$) |
    | Volatilidade e... | Funcao deterministica dos retornos passados | Processo latente estocastico |
    | Likelihood | Tratavel analiticamente | **Intratavel** |
    | Estimacao | MLE direta | Requer metodos de Monte Carlo |

    **SV como limite de GARCH**: Nelson (1990) mostrou que o GARCH(1,1) converge fracamente para um processo de difusao SV quando o intervalo de tempo $\Delta t \to 0$:

    $$
    \text{GARCH}(1,1) \xrightarrow{\Delta t \to 0} dV_t = \kappa(\bar{V} - V_t)\,dt + \sigma_V V_t \, dB_t
    $$

### 1.4 Propriedades Estatisticas

O modelo SV reproduz varios **fatos estilizados** de retornos financeiros:

=== "Leptokurtosis"

    A distribuicao marginal de $y_t$ (integrando sobre $h_t$) tem caudas mais pesadas que a gaussiana:

    $$
    y_t = \exp(h_t / 2) \varepsilon_t, \quad h_t \sim \mathcal{N}\left(\mu, \frac{\sigma_\eta^2}{1 - \phi^2}\right)
    $$

    A curtose em excesso e:

    $$
    \kappa_y = \frac{\mathbb{E}[y_t^4]}{(\mathbb{E}[y_t^2])^2} - 3 = 3\left[\exp\left(\frac{\sigma_\eta^2}{1-\phi^2}\right) - 1\right] > 0
    $$

    Para valores tipicos ($\phi = 0.97$, $\sigma_\eta = 0.15$), a curtose em excesso e $\kappa_y \approx 1.1$, gerando caudas pesadas moderadas.

=== "Volatility Clustering"

    Periodos de alta volatilidade tendem a ser seguidos por periodos de alta volatilidade. Isto emerge naturalmente da persistencia ($\phi$ proximo de 1) do processo latente:

    $$
    \text{Corr}(y_t^2, y_{t+k}^2) = \frac{\exp(\phi^k \sigma_\eta^2 / (1-\phi^2)) - 1}{\exp(\sigma_\eta^2 / (1-\phi^2)) - 1} \approx \phi^k \cdot \frac{\sigma_\eta^2}{1-\phi^2}
    $$

    A autocorrelacao dos retornos ao quadrado decai geometricamente com taxa $\phi$, mas muito lentamente quando $\phi \approx 1$.

=== "Leverage Effect"

    No modelo basico, $\text{Corr}(\varepsilon_t, \eta_t) = 0$, portanto nao ha efeito leverage. A extensao com leverage (Secao 4.1) adiciona correlacao negativa entre retornos e volatilidade futura.

=== "Retornos Nao-Correlacionados"

    Os retornos $y_t$ sao uma **diferenca de martingale** (serialmente nao-correlacionados):

    $$
    \mathbb{E}[y_t | y_{1:t-1}] = \mathbb{E}\left[\mathbb{E}[y_t | h_t, y_{1:t-1}]\right] = \mathbb{E}[\exp(h_t/2) \cdot \mathbb{E}[\varepsilon_t]] = 0
    $$

    enquanto $y_t^2$ e positivamente autocorrelacionado (volatility clustering).

---

## 2. Filtragem em Modelos SV

### 2.1 Por que o Filtro de Kalman Nao Funciona

O modelo SV e um modelo de estado-espaco **nao-linear**. A equacao de observacao:

$$
y_t = \exp(h_t / 2) \, \varepsilon_t
$$

e nao-linear em $h_t$, pois $y_t | h_t \sim \mathcal{N}(0, \exp(h_t))$. Isto viola a hipotese de linearidade gaussiana requerida pelo Filtro de Kalman.

!!! warning "Tentativa de Linearizacao"
    Uma abordagem classica (Harvey, Ruiz & Shephard, 1994) e transformar a observacao:

    $$
    \log y_t^2 = h_t + \log \varepsilon_t^2
    $$

    Agora a equacao e **linear** em $h_t$, mas $\log \varepsilon_t^2 \sim \log \chi^2(1)$, que nao e gaussiano. A aproximacao gaussiana $\log \chi^2(1) \approx \mathcal{N}(-1.27, \pi^2/2)$ introduz erros significativos, especialmente para observacoes proximas de zero ($y_t \approx 0$).

    Esta abordagem pode ser usada como inicializacao, mas **nao e eficiente** para inferencia final.

### 2.2 Particle Filter como Solucao Natural

O particle filter trata o modelo SV diretamente, sem necessidade de linearizacao ou aproximacoes gaussianas:

$$
p(h_t | y_{1:t}) \approx \sum_{i=1}^{N} W_t^{(i)} \, \delta_{h_t^{(i)}}(dh_t)
$$

!!! info "Algoritmo: Bootstrap PF para SV"
    1. **Inicializacao**: $h_0^{(i)} \sim \mathcal{N}(\mu, \sigma_\eta^2/(1-\phi^2))$ para $i = 1, \ldots, N$
    2. Para $t = 1, \ldots, T$:
        - **Propagacao**: $h_t^{(i)} \sim \mathcal{N}(\mu + \phi(h_{t-1}^{(i)} - \mu), \sigma_\eta^2)$
        - **Ponderacao**: $w_t^{(i)} = p(y_t | h_t^{(i)}) = \frac{1}{\sqrt{2\pi \exp(h_t^{(i)})}} \exp\left(-\frac{y_t^2}{2\exp(h_t^{(i)})}\right)$
        - **Normalizacao**: $W_t^{(i)} = w_t^{(i)} / \sum_j w_t^{(j)}$
        - **Resampling**: se $\text{ESS} < N/2$, resample

### 2.3 Eficiencia de Diferentes PFs para SV

Como o modelo SV tem estado unidimensional ($d_x = 1$), a curse of dimensionality nao e um problema. A eficiencia relativa depende da qualidade da proposal:

=== "Bootstrap PF"

    - **Proposal**: $q(h_t | h_{t-1}) = p(h_t | h_{t-1})$ (prior)
    - **Eficiencia**: Boa para SV, pois a prior e informativa em 1D
    - **ESS tipico**: $0.3N$-$0.7N$ para dados tipicos
    - **Custo**: $O(N)$ por passo

=== "Optimal Proposal"

    Para SV, a optimal proposal $p(h_t | h_{t-1}, y_t)$ nao tem forma fechada, mas pode ser aproximada:

    $$
    q^*(h_t | h_{t-1}, y_t) = p(h_t | h_{t-1}, y_t) \propto p(y_t | h_t) \, p(h_t | h_{t-1})
    $$

    - **Metodo**: Aproximacao por Laplace (expansao de Taylor de segunda ordem de $\log p(y_t | h_t) + \log p(h_t | h_{t-1})$)
    - **Resultado**: $q^*(h_t) \approx \mathcal{N}(\hat{h}_t, \hat{\sigma}_t^2)$ onde:

    $$
    \hat{h}_t = \underset{h}{\arg\max} \left[\log p(y_t|h) + \log p(h|h_{t-1})\right], \quad \hat{\sigma}_t^{-2} = -\frac{\partial^2}{\partial h^2}\left[\log p(y_t|h) + \log p(h|h_{t-1})\right]\bigg|_{h=\hat{h}_t}
    $$

    - **Eficiencia**: ESS tipico $0.7N$-$0.95N$, significativamente melhor que Bootstrap
    - **Custo**: $O(N)$ por passo, com constante maior (otimizacao por particula)

=== "Auxiliary PF"

    O Auxiliary PF (Pitt & Shephard, 1999) foi originalmente desenvolvido para modelos SV:

    - Usa uma primeira etapa de ponderacao baseada na previsao $p(y_t | h_{t-1}^{(i)}) \approx \mathcal{N}(0, \exp(\mu + \phi(h_{t-1}^{(i)} - \mu)))$
    - Resamplea **antes** de propagar, favorecendo particulas com alta likelihood preditiva
    - **Eficiencia**: Similar a optimal proposal em muitos cenarios

---

## 3. Estimacao de Parametros

### 3.1 Likelihood Intratavel

A likelihood do modelo SV requer integracao sobre toda a trajetoria latente:

$$
p(y_{1:T} | \theta) = \int \prod_{t=1}^{T} p(y_t | h_t) \prod_{t=1}^{T} p(h_t | h_{t-1}, \theta) \, dh_{1:T}
$$

Esta e uma integral em $\mathbb{R}^T$ (dimensao $T$, tipicamente $T > 1000$), **intratavel** analiticamente e por quadratura numerica. Metodos classicos (MLE direto, EM) nao sao aplicaveis.

!!! note "Estimador por Particle Filter"
    O particle filter fornece um estimador **nao-enviesado** da likelihood:

    $$
    \hat{p}^N(y_{1:T} | \theta) = \prod_{t=1}^{T} \left(\frac{1}{N}\sum_{i=1}^{N} w_t^{(i)}\right)
    $$

    Este estimador pode ser usado diretamente em algoritmos MCMC via o framework **pseudo-marginal**.

### 3.2 PMMH para Modelos SV

O **Particle Marginal Metropolis-Hastings (PMMH)** (Andrieu, Doucet & Holenstein, 2010) combina o pseudo-marginal framework com Metropolis-Hastings para amostrar $\theta \sim p(\theta | y_{1:T})$:

!!! info "Algoritmo: PMMH para SV"
    1. **Inicializacao**: Escolher $\theta^{(0)}$, executar PF para obter $\hat{p}^N(y_{1:T} | \theta^{(0)})$
    2. Para $m = 1, \ldots, M$:
        - **Propor**: $\theta^* \sim q(\cdot | \theta^{(m-1)})$
        - **Executar PF**: obter $\hat{p}^N(y_{1:T} | \theta^*)$
        - **Aceitar/Rejeitar** com probabilidade:

        $$
        \alpha = \min\left(1, \frac{\hat{p}^N(y_{1:T} | \theta^*) \, p(\theta^*) \, q(\theta^{(m-1)} | \theta^*)}{\hat{p}^N(y_{1:T} | \theta^{(m-1)}) \, p(\theta^{(m-1)}) \, q(\theta^* | \theta^{(m-1)})}\right)
        $$

    **Resultado**: A cadeia $\{\theta^{(m)}\}$ converge para a posterior exata $p(\theta | y_{1:T})$.

??? note "Detalhes de Implementacao para SV"
    **Reparametrizacao**: Trabalhar com parametros transformados para proposals mais eficientes:

    $$
    \begin{aligned}
    \tilde{\mu} &= \mu \\
    \tilde{\phi} &= \text{logit}\left(\frac{\phi + 1}{2}\right) = \log\frac{1+\phi}{1-\phi} \\
    \tilde{\sigma} &= \log \sigma_\eta
    \end{aligned}
    $$

    **Proposal conjunta**: Random walk gaussiana no espaco transformado:

    $$
    \tilde{\theta}^* \sim \mathcal{N}\left(\tilde{\theta}^{(m-1)}, \lambda^2 \hat{\Sigma}\right)
    $$

    onde $\hat{\Sigma}$ e estimada de um piloto run e $\lambda \approx 2.38 / \sqrt{3}$ (regra de Roberts & Rosenthal).

### 3.3 PGAS para Modelos SV

O **Particle Gibbs with Ancestor Sampling (PGAS)** (Lindsten, Jordan & Schon, 2014) e uma alternativa ao PMMH que amostra a trajetoria latente $h_{1:T}$ juntamente com $\theta$:

!!! info "Algoritmo: PGAS para SV"
    Gibbs sampler que alterna entre:

    1. **Amostrar $h_{1:T} | \theta, y_{1:T}$**: Conditional SMC com ancestor sampling

        - Fixa a trajetoria referencia $h_{1:T}^{(m-1)}$ como particula $N$
        - Executa PF condicional com $N-1$ novas particulas
        - Ancestor sampling: no passo $t$, reamostra o ancestral da particula fixa com pesos:

        $$
        \tilde{w}_{t-1}^{(i)} \propto W_{t-1}^{(i)} \, p(h_t^{\text{ref}} | h_{t-1}^{(i)}, \theta)
        $$

        - Seleciona a nova trajetoria $h_{1:T}^{(m)}$ do filtro condicional

    2. **Amostrar $\theta | h_{1:T}, y_{1:T}$**: Com $h_{1:T}$ fixo, a posterior de $\theta$ tem forma fechada parcial:

        $$
        p(\theta | h_{1:T}, y_{1:T}) = p(\mu, \phi, \sigma_\eta | h_{1:T})
        $$

        pois os retornos $y_t$ nao dependem de $\theta$ dado $h_t$. A condicional e:

        $$
        h_t = \mu + \phi(h_{t-1} - \mu) + \sigma_\eta \eta_t \implies \text{regressao linear em } (\mu, \phi)
        $$

!!! success "PGAS vs PMMH para SV"
    | Aspecto | PMMH | PGAS |
    |---------|------|------|
    | $N$ particulas necessarias | $500$-$2000$ | $20$-$100$ |
    | Mixing em $\theta$ | Depende de proposal | Bom (Gibbs) |
    | Mixing em $h_{1:T}$ | Nao amostra diretamente | Excelente (ancestor sampling) |
    | Implementacao | Simples | Mais complexa |
    | Series longas ($T > 5000$) | Var[$\log \hat{p}$] cresce | **Estavel** |

### 3.4 Escolha de Priors

!!! tip "Priors Recomendados para Modelo SV"
    | Parametro | Prior | Justificativa |
    |-----------|-------|---------------|
    | $\mu$ | $\mathcal{N}(0, 10)$ | Difuso; log-vol tipicamente $\in [-5, 5]$ |
    | $(\phi+1)/2$ | $\text{Beta}(20, 1.5)$ | Concentrado em $\phi \approx 0.95$-$0.99$; reflete alta persistencia empirica |
    | $\sigma_\eta$ | $\text{Half-Cauchy}(0, 1)$ | Fracamente informativo; permite tanto $\sigma_\eta$ pequeno quanto grande |

    **Identificacao**: O modelo SV e bem identificado — $\mu$ determina o nivel de volatilidade, $\phi$ determina a persistencia, e $\sigma_\eta$ determina a variabilidade. A unica fonte de nao-identificacao potencial e quando $\sigma_\eta \to 0$ (modelo colapsa para volatilidade constante).

---

## 4. Extensoes

### 4.1 SV com Leverage

O **efeito leverage** (Black, 1976) e a correlacao negativa observada entre retornos e volatilidade futura: quedas no preco tendem a aumentar a volatilidade.

$$
\boxed{
\begin{aligned}
y_t &= \exp(h_t/2) \, \varepsilon_t \\
h_{t+1} &= \mu + \phi(h_t - \mu) + \sigma_\eta \eta_t \\
\begin{pmatrix} \varepsilon_t \\ \eta_t \end{pmatrix} &\sim \mathcal{N}\left(\begin{pmatrix} 0 \\ 0 \end{pmatrix}, \begin{pmatrix} 1 & \rho \\ \rho & 1 \end{pmatrix}\right)
\end{aligned}
}
$$

onde $\rho < 0$ tipicamente ($\rho \approx -0.3$ a $-0.7$ para acoes).

!!! note "Implicacoes para Filtragem"
    Com leverage, a observacao $y_t$ contem informacao sobre $\eta_t$ e portanto sobre $h_{t+1}$:

    $$
    \mathbb{E}[\eta_t | \varepsilon_t] = \rho \varepsilon_t = \rho \, y_t \, \exp(-h_t/2)
    $$

    Isto altera a proposal otima. O Bootstrap PF, que ignora esta informacao, perde eficiencia. A proposal deve incorporar:

    $$
    q(h_{t+1} | h_t, y_t) = \mathcal{N}\left(\mu + \phi(h_t - \mu) + \rho \sigma_\eta y_t e^{-h_t/2}, \, \sigma_\eta^2(1 - \rho^2)\right)
    $$

### 4.2 SV-t (Fat Tails)

Substituir a distribuicao dos retornos por uma $t$-Student para capturar caudas mais pesadas:

$$
y_t = \exp(h_t/2) \, \varepsilon_t, \quad \varepsilon_t \sim t_\nu
$$

Equivalentemente, usando a representacao de mistura de escala:

$$
\begin{aligned}
y_t | h_t, \lambda_t &\sim \mathcal{N}(0, \lambda_t \exp(h_t)) \\
\lambda_t &\sim \text{Inv-Gamma}(\nu/2, \nu/2)
\end{aligned}
$$

!!! info "Vantagem da Representacao de Mistura"
    A representacao de mistura permite tratar $\lambda_t$ como variavel latente adicional, mantendo a gaussianidade condicional. No PGAS, $\lambda_t$ pode ser amostrado via Gibbs:

    $$
    \lambda_t | y_t, h_t, \nu \sim \text{Inv-Gamma}\left(\frac{\nu+1}{2}, \frac{\nu + y_t^2 \exp(-h_t)}{2}\right)
    $$

### 4.3 SV com Jumps

Adicionar jumps (saltos) nos retornos para capturar movimentos abruptos:

$$
\begin{aligned}
y_t &= \exp(h_t/2) \, \varepsilon_t + J_t \, \kappa_t \\
J_t &\sim \text{Bernoulli}(\lambda_J) \\
\kappa_t &\sim \mathcal{N}(\mu_J, \sigma_J^2)
\end{aligned}
$$

onde $J_t$ indica a ocorrencia de um jump e $\kappa_t$ e seu tamanho.

??? note "State-Space Aumentado"
    O espaco de estados expandido e $z_t = (h_t, J_t, \kappa_t)$, com dimensao efetiva 3. O particle filter opera neste espaco aumentado:

    - **Propagacao**: amostrar $h_t$ do AR(1) e $(J_t, \kappa_t)$ das priors
    - **Ponderacao**: $w_t \propto p(y_t | h_t, J_t, \kappa_t)$

    A natureza discreta de $J_t$ pode ser marginalizada (Rao-Blackwellized):

    $$
    p(y_t | h_t) = (1-\lambda_J) \, \mathcal{N}(y_t; 0, e^{h_t}) + \lambda_J \, \mathcal{N}(y_t; \mu_J, e^{h_t} + \sigma_J^2)
    $$

### 4.4 Multi-Factor SV

Modelos com multiplos fatores de volatilidade para capturar dinamicas de curto e longo prazo:

$$
\begin{aligned}
y_t &= \exp\left(\frac{h_{1,t} + h_{2,t}}{2}\right) \varepsilon_t \\
h_{1,t} &= \mu_1 + \phi_1(h_{1,t-1} - \mu_1) + \sigma_1 \eta_{1,t} & \text{(componente rapido)} \\
h_{2,t} &= \mu_2 + \phi_2(h_{2,t-1} - \mu_2) + \sigma_2 \eta_{2,t} & \text{(componente lento)}
\end{aligned}
$$

com $0 < \phi_1 < \phi_2 < 1$, onde $h_{1,t}$ captura flutuacoes de curto prazo e $h_{2,t}$ captura tendencias de longo prazo na volatilidade.

!!! warning "Desafios de Estimacao"
    - **Identificacao**: Sem restricoes, os dois fatores nao sao distinguiveis. Tipicamente, fixa-se $\phi_1 < \phi_2$ ou $\mu_1 = 0$
    - **Dimensionalidade**: O estado $z_t = (h_{1,t}, h_{2,t})$ e bidimensional; o particle filter funciona bem
    - **Priors**: Priors informativos sao criticos para separar os dois componentes

### 4.5 Tabela Comparativa das Extensoes

| Modelo | Estado | Parametros Adicionais | Complexidade PF |
|--------|--------|----------------------|-----------------|
| SV basico | $h_t \in \mathbb{R}$ | $(\mu, \phi, \sigma_\eta)$ | Baixa |
| SV-Leverage | $h_t \in \mathbb{R}$ | $+\rho$ | Baixa (proposal modificada) |
| SV-t | $(h_t, \lambda_t) \in \mathbb{R}^2$ | $+\nu$ | Baixa (RB em $\lambda_t$) |
| SV-Jump | $(h_t, J_t) \in \mathbb{R} \times \{0,1\}$ | $+(\lambda_J, \mu_J, \sigma_J)$ | Media (RB em $J_t$) |
| SV-2Factor | $(h_{1,t}, h_{2,t}) \in \mathbb{R}^2$ | $+(\mu_2, \phi_2, \sigma_2)$ | Media |

---

## Resumo

O modelo SV e o caso de uso canonico para particle filters em financas:

1. **Estado unidimensional** ($h_t$) torna o particle filter altamente eficiente
2. **Likelihood intratavel** exige metodos de Monte Carlo para estimacao
3. **PGAS** e o metodo de referencia para estimacao, combinando baixo $N$ com mixing excelente
4. **Extensoes** (leverage, fat tails, jumps) sao tratadas naturalmente pelo framework de particulas

---

## Referencias

!!! quote "Referencias Principais"
    - **Taylor, S.J.** (1986). *Modelling Financial Time Series*. John Wiley & Sons.
    - **Ghysels, E., Harvey, A.C. & Renault, E.** (1996). Stochastic volatility. In *Handbook of Statistics*, Vol. 14, 119-191.
    - **Harvey, A.C., Ruiz, E. & Shephard, N.** (1994). Multivariate stochastic variance models. *Review of Economic Studies*, 61(2), 247-264.
    - **Kim, S., Shephard, N. & Chib, S.** (1998). Stochastic volatility: likelihood inference and comparison with ARCH models. *Review of Economic Studies*, 65(3), 361-393.
    - **Pitt, M.K. & Shephard, N.** (1999). Filtering via simulation: Auxiliary particle filters. *Journal of the American Statistical Association*, 94(446), 590-599.
    - **Andrieu, C., Doucet, A. & Holenstein, R.** (2010). Particle Markov chain Monte Carlo methods. *Journal of the Royal Statistical Society: Series B*, 72(3), 269-342.
    - **Lindsten, F., Jordan, M.I. & Schon, T.B.** (2014). Particle Gibbs with ancestor sampling. *Journal of Machine Learning Research*, 15, 2145-2184.
    - **Nelson, D.B.** (1990). ARCH models as diffusion approximations. *Journal of Econometrics*, 45(1-2), 7-38.
    - **Black, F.** (1976). Studies of stock price volatility changes. *Proceedings of the Business and Economics Statistics Section*, American Statistical Association, 177-181.
    - **Heston, S.L.** (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options. *Review of Financial Studies*, 6(2), 327-343.
