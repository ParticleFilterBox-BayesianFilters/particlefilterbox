---
title: SMC Samplers - Teoria
description: Teoria de SMC Samplers como metodo geral de amostragem - sequencia de distribuicoes, mutation kernels, weight updates, adaptive tempering e estimacao de constantes de normalizacao.
---

# SMC Samplers: Teoria

Esta pagina desenvolve a teoria de **SMC Samplers** como um metodo geral de amostragem de distribuicoes complexas. Diferentemente do particle filter classico, que opera em modelos de estado-espaco, o SMC sampler e aplicavel a **qualquer distribuicao target**, utilizando uma sequencia de distribuicoes intermediarias para guiar as particulas da prior ate a posterior.

---

## 1. SMC como Metodo de Amostragem Geral

### 1.1 Sequencia de Distribuicoes

O framework de SMC samplers (Del Moral, Doucet & Jasra, 2006) opera sobre uma sequencia de distribuicoes **definidas pelo usuario**:

$$
\pi_0, \pi_1, \ldots, \pi_P
$$

onde:

- $\pi_0$ e uma distribuicao **facil de amostrar** (e.g., prior, uniforme, gaussiana)
- $\pi_P = \pi$ e a **distribuicao target** de interesse
- As intermediarias $\pi_n$ para $0 < n < P$ formam uma **ponte** entre $\pi_0$ e $\pi_P$

Cada distribuicao e conhecida ate uma constante de normalizacao:

$$
\pi_n(x) = \frac{\tilde{\pi}_n(x)}{Z_n}, \quad Z_n = \int \tilde{\pi}_n(x) \, dx
$$

!!! info "Ideia Central"
    Em vez de amostrar diretamente de uma distribuicao complexa $\pi_P$, construimos um **caminho gradual** de distribuicoes que transforma amostras faceis ($\pi_0$) em amostras da target ($\pi_P$), aplicando SMC ao longo deste caminho.

### 1.2 Tempering (Annealing)

A construcao mais comum e o **geometric bridging** (tempering):

$$
\tilde{\pi}_n(x) = \pi_0(x)^{1 - \beta_n} \, \pi_P(x)^{\beta_n}
$$

onde $0 = \beta_0 < \beta_1 < \cdots < \beta_P = 1$ e uma sequencia crescente de **temperaturas inversas**.

!!! tip "Interpretacao"
    - Para $\beta_n = 0$: $\pi_n = \pi_0$ (distribuicao inicial)
    - Para $\beta_n = 1$: $\pi_n = \pi_P$ (target)
    - Para $0 < \beta_n < 1$: interpolacao geometrica entre as duas

No contexto Bayesiano com $\pi_0(x) = p(\theta)$ (prior) e $\pi_P(x) = p(\theta | y) \propto p(\theta) p(y|\theta)$:

$$
\tilde{\pi}_n(\theta) = p(\theta) \, p(y|\theta)^{\beta_n}
$$

Isto corresponde a **aquecer** a likelihood gradualmente — a temperatura $1/\beta_n$ diminui de $\infty$ (prior pura) ate $1$ (posterior completa).

### 1.3 Outras Construcoes de Sequencias

!!! info "Estrategias Alternativas de Bridging"
    | Estrategia | Sequencia $\pi_n$ | Aplicacao |
    |---|---|---|
    | **Geometric tempering** | $\pi_0^{1-\beta_n} \pi_P^{\beta_n}$ | Amostragem de posteriores |
    | **Data annealing** | $p(\theta) \prod_{t=1}^{n} p(y_t \mid \theta)$ | Dados sequenciais (IBIS) |
    | **Likelihood tempering** | $p(\theta) p(y \mid \theta)^{\beta_n}$ | Caso especial de geometric |
    | **Constraint relaxation** | $\pi(x) \mathbb{1}[\|h(x)\| \leq \epsilon_n]$ | Problemas com restricoes |
    | **Rare event** | $p(x \mid \varphi(x) > \gamma_n)$ | Estimacao de probabilidades raras |

### 1.4 Espaco Estendido

Para justificar rigorosamente o SMC sampler, define-se uma distribuicao no **espaco estendido** $\mathcal{X}^{P+1}$:

$$
\bar{\pi}_P(x_{0:P}) = \pi_P(x_P) \prod_{n=1}^{P} L_n(x_{n-1} | x_n)
$$

onde $L_n$ sao **backward kernels** (ver Secao 3). A marginal de $x_P$ sob $\bar{\pi}_P$ e exatamente $\pi_P$, a target de interesse.

---

## 2. Mutation Kernels

### 2.1 MCMC Kernels que Preservam $\pi_n$

Apos o reweighting, as particulas sao **diversificadas** aplicando um kernel de transicao $K_n(x_n | x_{n-1})$ que preserva a distribuicao corrente $\pi_n$.

!!! abstract "Definicao: Kernel $\pi_n$-Invariante"
    Um kernel de Markov $K_n$ e **$\pi_n$-invariante** se:

    $$
    \int \pi_n(x) K_n(x' | x) \, dx = \pi_n(x')
    $$

    Isto garante que, se $X \sim \pi_n$, entao $X' \sim K_n(\cdot | X)$ tambem segue $\pi_n$.

### 2.2 Detailed Balance e Reversibilidade

Uma condicao suficiente para invariancia e o **detailed balance** (reversibilidade):

$$
\pi_n(x) K_n(x' | x) = \pi_n(x') K_n(x | x')
$$

!!! note "Detailed Balance $\Rightarrow$ Invariancia"
    Integrando ambos os lados em $x$:

    $$
    \int \pi_n(x) K_n(x' | x) \, dx = \pi_n(x') \int K_n(x | x') \, dx = \pi_n(x')
    $$

    A reversibilidade e **suficiente mas nao necessaria** para invariancia. Kernels nao-reversiveis (e.g., HMC com integracao deterministica) podem ter mixing superior.

### 2.3 Escolha do Mutation Kernel

=== "Metropolis-Hastings"

    O kernel MH com proposta $q_n(x' | x)$:

    1. Propor $x' \sim q_n(\cdot | x)$
    2. Aceitar com probabilidade:

    $$
    \alpha_n(x, x') = \min\left(1, \frac{\tilde{\pi}_n(x') \, q_n(x | x')}{\tilde{\pi}_n(x) \, q_n(x' | x)}\right)
    $$

    O kernel resultante e:

    $$
    K_n^{\text{MH}}(x' | x) = \alpha_n(x, x') q_n(x' | x) + \left(1 - \int \alpha_n(x, z) q_n(z|x) dz\right) \delta_x(x')
    $$

    !!! tip "Propostas Comuns"
        - **Random walk**: $q_n(x'|x) = \mathcal{N}(x' ; x, \Sigma_n)$
        - **Independent**: $q_n(x'|x) = q_n(x')$ — util quando boa aproximacao existe
        - **Adaptativa**: $\Sigma_n$ estimada a partir da populacao corrente de particulas

=== "Hamiltonian Monte Carlo"

    HMC utiliza a dinamica Hamiltoniana para explorar eficientemente a target:

    1. Amostrar momento $p \sim \mathcal{N}(0, M)$
    2. Integrar as equacoes de Hamilton por $L$ passos de tamanho $\epsilon$:

    $$
    \frac{dx}{dt} = M^{-1}p, \quad \frac{dp}{dt} = \nabla \log \tilde{\pi}_n(x)
    $$

    3. Aceitar/rejeitar com correcao de Metropolis

    !!! warning "Requisito"
        HMC requer que $\tilde{\pi}_n$ seja diferenciavel. Para targets discretas ou nao-difernciaveis, use MH ou Gibbs.

=== "Gibbs Sampling"

    Para targets com estrutura condicional $x = (x^{(1)}, \ldots, x^{(K)})$, Gibbs sampling atualiza cada componente condicionalmente:

    $$
    x^{(k)} \sim \pi_n(x^{(k)} | x^{(-k)})
    $$

    onde $x^{(-k)}$ denota todos os componentes exceto o $k$-esimo. Um ciclo completo de Gibbs e $\pi_n$-invariante (embora nao reversivel em geral).

### 2.4 Multiplos Passos de MCMC

Na pratica, aplica-se **mais de um passo** de MCMC como mutation. Se $K_n$ e $\pi_n$-invariante, entao $K_n^R$ (composicao de $R$ passos) tambem e $\pi_n$-invariante:

$$
K_n^R(x' | x) = \underbrace{K_n \circ K_n \circ \cdots \circ K_n}_{R \text{ vezes}}(x' | x)
$$

!!! tip "Quantos Passos?"
    - Mais passos de MCMC $\Rightarrow$ particulas mais diversificadas $\Rightarrow$ melhor mixing
    - Mas maior custo computacional por iteracao do SMC
    - Regra pratica: monitorar a **acceptance rate** e o ESS apos mutation
    - Abordagem adaptativa: aumentar $R$ quando o ESS pos-mutation e baixo

---

## 3. Weight Update e Backward Kernels

### 3.1 Pesos Incrementais

O peso incremental no passo $n$ corrige a discrepancia entre a target $\pi_n$ e a distribuicao obtida apos mutation a partir de $\pi_{n-1}$.

!!! abstract "Proposicao: Peso Incremental Geral"
    Dado o espaco estendido com backward kernels $L_n$, o peso incremental e:

    $$
    w_n(x_{n-1}, x_n) = \frac{\tilde{\pi}_n(x_n) \, L_n(x_{n-1} | x_n)}{\tilde{\pi}_{n-1}(x_{n-1}) \, K_n(x_n | x_{n-1})}
    $$

    onde $K_n$ e o mutation kernel e $L_n$ e o backward kernel.

??? note "Derivacao dos Pesos Incrementais"
    Considere a distribuicao no espaco estendido:

    $$
    \bar{\pi}_P(x_{0:P}) = \pi_P(x_P) \prod_{n=1}^{P} L_n(x_{n-1} | x_n)
    $$

    e a distribuicao proposta:

    $$
    \bar{q}_P(x_{0:P}) = \pi_0(x_0) \prod_{n=1}^{P} K_n(x_n | x_{n-1})
    $$

    O peso de importance sampling no espaco estendido e:

    $$
    W(x_{0:P}) = \frac{\bar{\pi}_P(x_{0:P})}{\bar{q}_P(x_{0:P})} = \frac{\tilde{\pi}_P(x_P) \prod_{n=1}^{P} L_n(x_{n-1}|x_n)}{Z_P \, \tilde{\pi}_0(x_0)/Z_0 \prod_{n=1}^{P} K_n(x_n|x_{n-1})}
    $$

    Rearranjando e usando a decomposicao telescopica:

    $$
    W(x_{0:P}) = \frac{Z_0}{Z_P} \prod_{n=1}^{P} \frac{\tilde{\pi}_n(x_n) L_n(x_{n-1}|x_n)}{\tilde{\pi}_{n-1}(x_{n-1}) K_n(x_n|x_{n-1})}
    = \frac{Z_0}{Z_P} \prod_{n=1}^{P} w_n(x_{n-1}, x_n)
    $$

    Portanto, os pesos incrementais permitem computacao sequencial do peso total. $\blacksquare$

### 3.2 Backward Kernel e sua Importancia

O backward kernel $L_n(x_{n-1} | x_n)$ e um **grau de liberdade** no design do algoritmo. Qualquer distribuicao condicional propria pode ser usada, mas a escolha afeta a **variancia dos pesos**.

!!! warning "Impacto na Variancia"
    A escolha de $L_n$ nao afeta a **corretude** do algoritmo (o estimador permanece consistente para qualquer $L_n$ proprio), mas afeta dramaticamente a **eficiencia**. Um $L_n$ ruim leva a pesos altamente variaveis e ESS baixo.

### 3.3 Backward Kernel Otimo

!!! abstract "Proposicao: Backward Kernel Otimo (Del Moral, Doucet & Jasra, 2006)"
    O backward kernel que **minimiza a variancia** dos pesos incrementais e:

    $$
    L_n^{\text{opt}}(x_{n-1} | x_n) = \frac{\pi_n(x_{n-1}) \, K_n(x_n | x_{n-1})}{\int \pi_n(x') \, K_n(x_n | x') \, dx'}
    = \frac{\pi_n(x_{n-1}) \, K_n(x_n | x_{n-1})}{\pi_n K_n(x_n)}
    $$

    onde $\pi_n K_n(x_n) = \int \pi_n(x') K_n(x_n | x') dx'$ e a distribuicao marginal de $x_n$ apos aplicar $K_n$ a $\pi_n$.

??? note "Derivacao do Backward Kernel Otimo"
    Substituindo $L_n^{\text{opt}}$ na expressao do peso incremental:

    $$
    w_n(x_{n-1}, x_n) = \frac{\tilde{\pi}_n(x_n) \cdot \frac{\pi_n(x_{n-1}) K_n(x_n|x_{n-1})}{\pi_n K_n(x_n)}}{\tilde{\pi}_{n-1}(x_{n-1}) \cdot K_n(x_n|x_{n-1})}
    $$

    Os termos $K_n(x_n|x_{n-1})$ cancelam:

    $$
    w_n = \frac{\tilde{\pi}_n(x_n) \cdot \pi_n(x_{n-1})}{\pi_n K_n(x_n) \cdot \tilde{\pi}_{n-1}(x_{n-1})}
    = \frac{\tilde{\pi}_n(x_n) \cdot \tilde{\pi}_n(x_{n-1}) / Z_n}{\pi_n K_n(x_n) \cdot \tilde{\pi}_{n-1}(x_{n-1})}
    $$

    Se $K_n$ e $\pi_n$-invariante, entao $\pi_n K_n = \pi_n$, e:

    $$
    w_n = \frac{\tilde{\pi}_n(x_n) \cdot \tilde{\pi}_n(x_{n-1})}{ Z_n \cdot \pi_n(x_n) \cdot \tilde{\pi}_{n-1}(x_{n-1})}
    = \frac{Z_n \pi_n(x_n) \cdot \tilde{\pi}_n(x_{n-1})}{Z_n \cdot \pi_n(x_n) \cdot \tilde{\pi}_{n-1}(x_{n-1})}
    = \frac{\tilde{\pi}_n(x_{n-1})}{\tilde{\pi}_{n-1}(x_{n-1})}
    $$

    O peso depende **apenas de $x_{n-1}$** e nao de $x_n$, o que e intuitivamente otimo: a mutation nao introduz variabilidade adicional nos pesos.

    Este resultado mostra que, com $K_n$ sendo $\pi_n$-invariante e backward kernel otimo, os pesos simplificam para:

    $$
    \boxed{w_n(x_{n-1}) = \frac{\tilde{\pi}_n(x_{n-1})}{\tilde{\pi}_{n-1}(x_{n-1})}}
    $$

    $\blacksquare$

### 3.4 Caso Especial: Tempering

Para a sequencia de tempering $\tilde{\pi}_n(\theta) = p(\theta) \, p(y|\theta)^{\beta_n}$, com backward kernel otimo e mutation $\pi_n$-invariante, o peso incremental simplifica para:

$$
w_n(\theta) = \frac{p(\theta) \, p(y|\theta)^{\beta_n}}{p(\theta) \, p(y|\theta)^{\beta_{n-1}}} = p(y|\theta)^{\beta_n - \beta_{n-1}}
$$

!!! tip "Simplicidade Notavel"
    O peso incremental no tempering e simplesmente a **likelihood elevada ao incremento de temperatura**. Nao depende da prior e e computacionalmente barato.

---

## 4. Adaptive Tempering

### 4.1 Motivacao

A eficiencia do SMC sampler depende criticamente da escolha do **schedule de temperaturas** $\{\beta_n\}_{n=0}^{P}$. Um schedule mal escolhido leva a:

- $\beta_n - \beta_{n-1}$ **muito grande**: pesos degenerados, ESS baixo
- $\beta_n - \beta_{n-1}$ **muito pequeno**: muitas iteracoes, custo computacional excessivo

A solucao e escolher $\beta_n$ **adaptativamente** para manter um nivel desejado de ESS.

### 4.2 Criterio: ESS Target

!!! abstract "Definicao: Adaptive Tempering"
    Dado um ESS target $\text{ESS}^* \in (0, N)$, escolhe-se $\beta_n$ como a solucao de:

    $$
    \text{ESS}(\beta_n) = \text{ESS}^*
    $$

    onde, dado o conjunto de particulas $\{x_{n-1}^{(i)}\}_{i=1}^N$ com pesos uniformes (apos resampling):

    $$
    \text{ESS}(\beta) = \frac{\left(\sum_{i=1}^{N} p(y | x_{n-1}^{(i)})^{\beta - \beta_{n-1}}\right)^2}{\sum_{i=1}^{N} p(y | x_{n-1}^{(i)})^{2(\beta - \beta_{n-1})}}
    $$

Na pratica, usa-se frequentemente $\text{ESS}^* = \alpha N$ com $\alpha \in [0.5, 0.9]$, sendo $\alpha = 0.5$ (metade das particulas) uma escolha popular.

### 4.3 Bisection Method

O ESS como funcao de $\beta$ e **monotonicamente decrescente** para $\beta > \beta_{n-1}$ (incrementos maiores de temperatura produzem pesos mais variaveis). Isto garante que a equacao $\text{ESS}(\beta) = \text{ESS}^*$ tem solucao unica, que pode ser encontrada por **bisecao**.

**Algoritmo: Bisection para $\beta_n$**

1. Inicializar: $\beta_{\min} = \beta_{n-1}$, $\beta_{\max} = 1$
2. Repetir ate convergencia ($|\beta_{\max} - \beta_{\min}| < \epsilon$):
    1. $\beta_{\text{mid}} = (\beta_{\min} + \beta_{\max}) / 2$
    2. Calcular $\text{ESS}(\beta_{\text{mid}})$
    3. Se $\text{ESS}(\beta_{\text{mid}}) > \text{ESS}^*$: $\beta_{\min} \leftarrow \beta_{\text{mid}}$
    4. Senao: $\beta_{\max} \leftarrow \beta_{\text{mid}}$
3. Retornar $\beta_n = \beta_{\text{mid}}$

!!! note "Caso Especial: $\text{ESS}(1) \geq \text{ESS}^*$"
    Se $\text{ESS}(\beta = 1) \geq \text{ESS}^*$, podemos pular diretamente para a target final: $\beta_n = 1$. Isto significa que as particulas ja estao suficientemente proximas da posterior.

### 4.4 Propriedades da Adaptacao

!!! abstract "Proposicao: Convergencia do Adaptive Schedule"
    Sob condicoes de regularidade (likelihood limitada e continua), o adaptive tempering com ESS target $\text{ESS}^* = \alpha N$ satisfaz:

    1. **Terminacao finita**: o algoritmo atinge $\beta_P = 1$ em $P < \infty$ passos
    2. **Adaptacao automatica a dificuldade**: problemas faceis (posterior proxima da prior) requerem poucos passos; problemas dificeis (posterior concentrada) requerem mais passos
    3. **Invariancia a $N$**: o numero de passos $P$ nao depende de $N$ (para $N$ grande)

??? note "Sketch da Prova: Terminacao Finita"
    1. Em cada passo, $\beta_n > \beta_{n-1}$ (progresso positivo), pois $\text{ESS}(\beta_{n-1}) = N > \text{ESS}^*$
    2. O incremento minimo $\Delta \beta_{\min} > 0$ e limitado inferiormente pela regularidade da likelihood
    3. Como $\beta$ esta limitado em $[0, 1]$, o algoritmo termina em no maximo $P \leq 1/\Delta\beta_{\min}$ passos $\blacksquare$

!!! warning "Validade Teorica da Adaptacao"
    A adaptacao de $\beta_n$ baseada nas particulas correntes quebra a estrutura de **Feynman-Kac** classica, pois os pesos passam a depender de toda a populacao. Chopin (2002) e Del Moral, Doucet & Jasra (2012) mostram que os resultados de convergencia se mantem sob condicoes adicionais, mas a prova requer argumentos mais sofisticados de **interacting particle systems**.

### 4.5 Algoritmo Completo: SMC Sampler Adaptativo

**Algoritmo: Adaptive SMC Sampler**

1. **Inicializacao**: Amostrar $x_0^{(i)} \sim \pi_0$ para $i = 1, \ldots, N$. Definir $\beta_0 = 0$
2. Para $n = 1, 2, \ldots$:
    1. **Adaptive tempering**: Encontrar $\beta_n$ por bisecao tal que $\text{ESS}(\beta_n) = \text{ESS}^*$
    2. **Reweighting**: Calcular $w_n^{(i)} = p(y | x_{n-1}^{(i)})^{\beta_n - \beta_{n-1}}$
    3. **Resampling**: Se $\text{ESS} < N_{\text{thr}}$, reamostrar com $W_n^{(i)} \propto w_n^{(i)}$
    4. **Mutation**: Aplicar $R$ passos de MCMC kernel $K_n$ que preserva $\pi_n$
    5. **Parada**: Se $\beta_n = 1$, terminar
3. Retornar particulas $\{x_P^{(i)}, W_P^{(i)}\}_{i=1}^N$ como amostras de $\pi_P$

---

## 5. Estimacao de Constantes de Normalizacao

### 5.1 Estimador da Razao de Normalizacao

Uma das aplicacoes mais importantes do SMC sampler e a estimacao da **razao de constantes de normalizacao** $Z_P / Z_0$.

!!! abstract "Teorema: Estimador Unbiased de $Z_P / Z_0$ (Del Moral, Doucet & Jasra, 2006)"
    O estimador SMC da razao de normalizacao:

    $$
    \widehat{\frac{Z_P}{Z_0}} = \prod_{n=1}^{P} \left(\frac{1}{N} \sum_{i=1}^{N} w_n^{(i)}\right)
    $$

    e **nao-enviesado**:

    $$
    \mathbb{E}\left[\widehat{\frac{Z_P}{Z_0}}\right] = \frac{Z_P}{Z_0}
    $$

??? note "Sketch da Prova: Unbiasedness"
    A prova segue a mesma estrutura do estimador de marginal likelihood em particle filters.

    1. **Espaco estendido**: Define-se a distribuicao $\bar{\pi}_P$ no espaco estendido das trajetorias. O SMC sampler e um algoritmo de importance sampling sequencial neste espaco.

    2. **Decomposicao telescopica**: A razao $Z_P/Z_0$ decompoe-se como:

    $$
    \frac{Z_P}{Z_0} = \prod_{n=1}^{P} \frac{Z_n}{Z_{n-1}}
    $$

    3. **Unbiasedness de cada fator**: Condicional as particulas do passo $n-1$, o estimador $\frac{1}{N}\sum_i w_n^{(i)}$ e unbiased para $Z_n/Z_{n-1}$. Formalmente:

    $$
    \mathbb{E}\left[\frac{1}{N}\sum_{i=1}^N w_n^{(i)} \;\middle|\; \mathcal{F}_{n-1}\right] = \frac{Z_n}{Z_{n-1}}
    $$

    onde $\mathcal{F}_{n-1}$ e a $\sigma$-algebra gerada pelas particulas ate o passo $n-1$.

    4. **Produto de unbiased**: Pela propriedade de tower rule (lei das expectativas iteradas):

    $$
    \mathbb{E}\left[\prod_{n=1}^P \frac{1}{N}\sum_i w_n^{(i)}\right] = \prod_{n=1}^P \frac{Z_n}{Z_{n-1}} = \frac{Z_P}{Z_0}
    $$

    $\blacksquare$

### 5.2 Aplicacao: Marginal Likelihood e Bayes Factors

No contexto Bayesiano com $\pi_0(\theta) = p(\theta)$ e $\pi_P(\theta) \propto p(\theta) p(y|\theta)$:

$$
\frac{Z_P}{Z_0} = \frac{\int p(\theta) p(y|\theta) d\theta}{\int p(\theta) d\theta} = p(y)
$$

A razao de normalizacao e exatamente a **marginal likelihood** (evidencia do modelo).

!!! info "Bayes Factors via SMC Sampler"
    Para comparar modelos $\mathcal{M}_1$ e $\mathcal{M}_2$:

    $$
    \text{BF}_{12} = \frac{p(y | \mathcal{M}_1)}{p(y | \mathcal{M}_2)} \approx \frac{\hat{Z}_P^{(1)} / \hat{Z}_0^{(1)}}{\hat{Z}_P^{(2)} / \hat{Z}_0^{(2)}}
    $$

    Cada SMC sampler fornece um estimador **nao-enviesado** da evidencia, permitindo comparacao direta.

### 5.3 Variancia do Estimador

A variancia do estimador de $Z_P/Z_0$ depende da **variancia dos pesos em cada passo**:

$$
\text{Var}\left[\widehat{\frac{Z_P}{Z_0}}\right] \approx \left(\frac{Z_P}{Z_0}\right)^2 \sum_{n=1}^{P} \frac{1}{N} \text{Var}_{\pi_{n-1}}\left[\frac{\tilde{\pi}_n(x)}{\tilde{\pi}_{n-1}(x)}\right]
$$

!!! tip "Implicacoes Praticas"
    - Mais particulas ($N$ maior) $\Rightarrow$ menor variancia ($O(1/N)$ por passo)
    - Mais passos intermediarios ($P$ maior) $\Rightarrow$ pesos menos variaveis por passo, mas mais termos na soma
    - O adaptive tempering equilibra automaticamente estes dois efeitos
    - A variancia do logaritmo e frequentemente mais relevante:

    $$
    \text{Var}\left[\log \widehat{\frac{Z_P}{Z_0}}\right] \approx \sum_{n=1}^{P} \frac{1}{N} \text{Var}_{\pi_{n-1}}\left[\frac{\tilde{\pi}_n(x)}{\tilde{\pi}_{n-1}(x)}\right]
    $$

### 5.4 Consistencia e CLT

!!! abstract "Teorema: CLT para o Estimador de Normalizacao"
    Sob condicoes de regularidade (distribuicoes intermediarias com suporte comum, pesos limitados):

    $$
    \sqrt{N}\left(\widehat{\frac{Z_P}{Z_0}} - \frac{Z_P}{Z_0}\right) \xrightarrow{d} \mathcal{N}\left(0, \sigma_Z^2\right)
    $$

    onde a variancia assintotica $\sigma_Z^2$ admite a decomposicao:

    $$
    \sigma_Z^2 = \left(\frac{Z_P}{Z_0}\right)^2 \sum_{n=1}^{P} \text{Var}_{\pi_{n-1}}\left[\frac{\tilde{\pi}_n(X)}{\tilde{\pi}_{n-1}(X)}\right]
    $$

    Este resultado fornece **intervalos de confianca** para a evidencia do modelo e, consequentemente, para Bayes factors.

---

## Referencias

| Referencia | Contribuicao |
|---|---|
| Del Moral, Doucet & Jasra (2006). *Sequential Monte Carlo samplers* | Framework teorico geral para SMC samplers |
| Chopin (2002). *A sequential particle filter method for static models* | SMC para modelos estaticos, adaptive resampling |
| Neal (2001). *Annealed importance sampling* | Precursor: importance sampling com annealing |
| Del Moral (2004). *Feynman-Kac Formulae* | Framework geral de Feynman-Kac para SMC |
| Del Moral, Doucet & Jasra (2012). *An adaptive sequential Monte Carlo method for approximate Bayesian computation* | Convergencia de SMC adaptativo |
| Jasra, Stephens & Holmes (2007). *On population-based simulation for static inference* | Comparacao de SMC samplers com MCMC |
| Beskos et al. (2016). *On the convergence of adaptive SMC methods* | Resultados rigorosos de convergencia para SMC adaptativo |
| Zhou, Johansen & Aston (2016). *Toward automatic model comparison: an adaptive sequential Monte Carlo approach* | Adaptive tempering e model comparison |
