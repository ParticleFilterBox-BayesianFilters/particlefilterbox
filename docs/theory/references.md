---
title: References
description: Complete bibliography of Sequential Monte Carlo, particle filters, and PMCMC literature referenced throughout the particlefilterbox documentation.
---

# References

This page collects the complete bibliography for the theoretical foundations of **particlefilterbox**. References are organized by topic and cross-linked to the documentation pages where they are discussed.

---

## 1. Foundational Works

These seminal works established the field of Sequential Monte Carlo and form the theoretical backbone of particlefilterbox.

<div class="reference-list" markdown>

**Gordon, N. J., Salmond, D. J., & Smith, A. F. M. (1993).** Novel approach to nonlinear/non-Gaussian Bayesian state estimation. *IEE Proceedings F - Radar and Signal Processing*, 140(2), 107--113.
[:material-link-variant: DOI](https://doi.org/10.1049/ip-f-2.1993.0015){ .md-button .md-button--primary }

> Introduced the **bootstrap particle filter** -- the first practical SMC algorithm for nonlinear state-space models.
> Referenced in: [Particle Filter Theory](particle-filter-theory.md), [SMC Theory](smc-theory.md)

---

**Doucet, A., de Freitas, N., & Gordon, N. (Eds.) (2001).** *Sequential Monte Carlo Methods in Practice*. Springer, New York.
[:material-link-variant: DOI](https://doi.org/10.1007/978-1-4757-3437-9){ .md-button .md-button--primary }

> Comprehensive edited volume covering the full spectrum of SMC methods -- the definitive early reference for the field.
> Referenced in: [SMC Theory](smc-theory.md), [Particle Filter Theory](particle-filter-theory.md), [Resampling Theory](resampling-theory.md)

---

**Del Moral, P. (2004).** *Feynman-Kac Formulae: Genealogical and Interacting Particle Systems with Applications*. Springer, New York.
[:material-link-variant: DOI](https://doi.org/10.1007/978-1-4684-9393-1){ .md-button .md-button--primary }

> Rigorous mathematical treatment of interacting particle systems through Feynman-Kac models, providing the measure-theoretic foundations of SMC.
> Referenced in: [SMC Theory](smc-theory.md), [Convergence Theory](convergence-theory.md)

---

**Chopin, N., & Papaspiliopoulos, O. (2020).** *An Introduction to Sequential Monte Carlo*. Springer Series in Statistics, Springer, Cham.
[:material-link-variant: DOI](https://doi.org/10.1007/978-3-030-47845-2){ .md-button .md-button--primary }

> Modern textbook providing a unified, accessible treatment of SMC with emphasis on methodology and implementation.
> Referenced in: [SMC Theory](smc-theory.md), [SMC Sampler Theory](smc-sampler-theory.md), [Convergence Theory](convergence-theory.md)

---

**Liu, J. S. (2001).** *Monte Carlo Strategies in Scientific Computing*. Springer, New York.
[:material-link-variant: DOI](https://doi.org/10.1007/978-0-387-76371-2){ .md-button .md-button--primary }

> Broad coverage of Monte Carlo methods including importance sampling, SIS, and resampling -- essential background for SMC practitioners.
> Referenced in: [SMC Theory](smc-theory.md), [Resampling Theory](resampling-theory.md)

</div>

---

## 2. Particle Filters

Core particle filtering algorithms implemented in particlefilterbox.

<div class="reference-list" markdown>

**Pitt, M. K., & Shephard, N. (1999).** Filtering via simulation: Auxiliary particle filters. *Journal of the American Statistical Association*, 94(446), 590--599.
[:material-link-variant: DOI](https://doi.org/10.1080/01621459.1999.10474153){ .md-button .md-button--primary }

> Introduced the **auxiliary particle filter** which pre-selects particles using a first-stage weighting to improve proposal efficiency.
> Referenced in: [Particle Filter Theory](particle-filter-theory.md)

---

**Doucet, A., Godsill, S. J., & Andrieu, C. (2000).** On sequential Monte Carlo sampling methods for Bayesian filtering. *Statistics and Computing*, 10(3), 197--208.
[:material-link-variant: DOI](https://doi.org/10.1023/A:1008935410038){ .md-button .md-button--primary }

> Unified framework for SMC filtering introducing the **Rao-Blackwellized particle filter** (RBPF) that marginalizes linear sub-structures analytically.
> Referenced in: [RBPF Theory](rbpf-theory.md), [Particle Filter Theory](particle-filter-theory.md)

---

**Van der Merwe, R., Doucet, A., de Freitas, N., & Wan, E. A. (2000).** The unscented particle filter. In *Advances in Neural Information Processing Systems 13* (NIPS 2000).
[:material-link-variant: Link](https://papers.nips.cc/paper/2000/hash/f5c3dd7514bf620a1b85450d2ae374b1-Abstract.html){ .md-button .md-button--primary }

> Combines the unscented Kalman filter proposal with particle filtering to build a better importance density for nonlinear models.
> Referenced in: [Particle Filter Theory](particle-filter-theory.md)

---

**Musso, C., Oudjane, N., & Le Gland, F. (2001).** Improving regularised particle filters. In *Sequential Monte Carlo Methods in Practice* (pp. 247--271). Springer, New York.
[:material-link-variant: DOI](https://doi.org/10.1007/978-1-4757-3437-9_12){ .md-button .md-button--primary }

> Develops the **regularized particle filter** that uses kernel density estimation to produce continuous approximations and reduce sample impoverishment.
> Referenced in: [Particle Filter Theory](particle-filter-theory.md)

---

**Evensen, G. (2003).** The Ensemble Kalman Filter: Theoretical formulation and practical implementation. *Ocean Dynamics*, 53(4), 343--367.
[:material-link-variant: DOI](https://doi.org/10.1007/s10236-003-0036-9){ .md-button .md-button--primary }

> Formulation of the **Ensemble Kalman Filter (EnKF)** -- a Monte Carlo approach to Kalman filtering widely used in high-dimensional geophysical applications.
> Referenced in: [Particle Filter Theory](particle-filter-theory.md)

---

**Doucet, A., & Johansen, A. M. (2009).** A tutorial on particle filtering and smoothing: Fifteen years later. In *The Oxford Handbook of Nonlinear Filtering* (pp. 656--704). Oxford University Press.
[:material-link-variant: Link](https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf){ .md-button .md-button--primary }

> Authoritative tutorial covering the state of the art in particle filtering and smoothing circa 2009 -- excellent pedagogical reference.
> Referenced in: [Particle Filter Theory](particle-filter-theory.md), [Smoothing Theory](smoothing-theory.md)

---

**Kong, A., Liu, J. S., & Wong, W. H. (1994).** Sequential imputations and Bayesian missing data problems. *Journal of the American Statistical Association*, 89(425), 278--288.
[:material-link-variant: DOI](https://doi.org/10.1080/01621459.1994.10476469){ .md-button .md-button--primary }

> Early work on Sequential Importance Sampling and the concept of effective sample size (ESS) for monitoring weight degeneracy.
> Referenced in: [SMC Theory](smc-theory.md), [Resampling Theory](resampling-theory.md)

</div>

---

## 3. Resampling

Resampling schemes that address weight degeneracy in SMC algorithms.

<div class="reference-list" markdown>

**Kitagawa, G. (1996).** Monte Carlo filter and smoother for non-Gaussian nonlinear state space models. *Journal of Computational and Graphical Statistics*, 5(1), 1--25.
[:material-link-variant: DOI](https://doi.org/10.1080/10618600.1996.10474692){ .md-button .md-button--primary }

> Introduced **systematic resampling** for particle filters -- the default resampling method in most SMC implementations due to its low variance.
> Referenced in: [Resampling Theory](resampling-theory.md), [Particle Filter Theory](particle-filter-theory.md)

---

**Carpenter, J., Clifford, P., & Fearnhead, P. (1999).** Improved particle filter for nonlinear problems. *IEE Proceedings - Radar, Sonar and Navigation*, 146(1), 2--7.
[:material-link-variant: DOI](https://doi.org/10.1049/ip-rsn:19990255){ .md-button .md-button--primary }

> Analysis and comparison of resampling algorithms including stratified resampling for particle filters.
> Referenced in: [Resampling Theory](resampling-theory.md)

---

**Douc, R., & Cappe, O. (2005).** Comparison of resampling schemes for particle filtering. In *Proceedings of the 4th International Symposium on Image and Signal Processing and Analysis (ISPA)* (pp. 64--69).
[:material-link-variant: DOI](https://doi.org/10.1109/ISPA.2005.195385){ .md-button .md-button--primary }

> Systematic comparison of multinomial, residual, stratified, and systematic resampling -- establishing relative variance properties.
> Referenced in: [Resampling Theory](resampling-theory.md)

---

**Gerber, M., Chopin, N., & Whiteley, N. (2019).** Negative association, ordering and convergence of resampling methods. *The Annals of Statistics*, 47(4), 2236--2260.
[:material-link-variant: DOI](https://doi.org/10.1214/18-AOS1746){ .md-button .md-button--primary }

> Establishes **optimal transport resampling** and proves negative association properties that yield sharper convergence bounds.
> Referenced in: [Resampling Theory](resampling-theory.md), [Convergence Theory](convergence-theory.md)

---

**Murray, L. M., Lee, A., & Jacob, P. E. (2016).** Parallel resampling in the particle filter. *Journal of Computational and Graphical Statistics*, 25(3), 789--805.
[:material-link-variant: DOI](https://doi.org/10.1080/10618600.2015.1062015){ .md-button .md-button--primary }

> Methods for parallelizing resampling steps in particle filters -- relevant for GPU and multi-core implementations.
> Referenced in: [Resampling Theory](resampling-theory.md)

</div>

---

## 4. Particle Smoothing

Algorithms for estimating the smoothing distribution $p(x_{0:T} \mid y_{1:T})$.

<div class="reference-list" markdown>

**Doucet, A., Godsill, S. J., & Andrieu, C. (2000).** On sequential Monte Carlo sampling methods for Bayesian filtering. *Statistics and Computing*, 10(3), 197--208.
[:material-link-variant: DOI](https://doi.org/10.1023/A:1008935410038){ .md-button .md-button--primary }

> Introduced **Forward Filtering Backward Smoothing (FFBSm)** and **Forward Filtering Backward Simulation (FFBSi)** algorithms.
> Referenced in: [Smoothing Theory](smoothing-theory.md)

---

**Godsill, S. J., Doucet, A., & West, M. (2004).** Monte Carlo smoothing for nonlinear time series. *Journal of the American Statistical Association*, 99(465), 156--168.
[:material-link-variant: DOI](https://doi.org/10.1198/016214504000000151){ .md-button .md-button--primary }

> Extended backward simulation methods for efficient particle smoothing in general state-space models.
> Referenced in: [Smoothing Theory](smoothing-theory.md)

---

**Briers, M., Doucet, A., & Maskell, S. (2010).** Smoothing algorithms for state-space models. *Annals of the Institute of Statistical Mathematics*, 62(1), 61--89.
[:material-link-variant: DOI](https://doi.org/10.1007/s10463-009-0236-2){ .md-button .md-button--primary }

> Comprehensive treatment of particle smoothing algorithms with complexity analysis and practical guidelines.
> Referenced in: [Smoothing Theory](smoothing-theory.md)

---

**Lindsten, F., & Schon, T. B. (2013).** Backward simulation methods for Monte Carlo statistical inference. *Foundations and Trends in Machine Learning*, 6(1), 1--143.
[:material-link-variant: DOI](https://doi.org/10.1561/2200000045){ .md-button .md-button--primary }

> Monograph-length survey of backward simulation methods including rejection-based and MCMC variants for improved efficiency.
> Referenced in: [Smoothing Theory](smoothing-theory.md)

---

**Fearnhead, P., Wyncoll, D., & Tawn, J. (2010).** A sequential smoothing algorithm with linear computational cost. *Biometrika*, 97(2), 447--464.
[:material-link-variant: DOI](https://doi.org/10.1093/biomet/asq013){ .md-button .md-button--primary }

> Proposed an $O(N)$ two-filter smoother that avoids the $O(N^2)$ cost of standard FFBSm.
> Referenced in: [Smoothing Theory](smoothing-theory.md)

</div>

---

## 5. SMC Methods

Advanced SMC algorithms beyond standard particle filtering.

<div class="reference-list" markdown>

**Del Moral, P., Doucet, A., & Jasra, A. (2006).** Sequential Monte Carlo samplers. *Journal of the Royal Statistical Society: Series B*, 68(3), 411--436.
[:material-link-variant: DOI](https://doi.org/10.1111/j.1467-9868.2006.00553.x){ .md-button .md-button--primary }

> Introduced **SMC Samplers** -- a general framework for sampling from sequences of distributions on common spaces using MCMC kernels.
> Referenced in: [SMC Sampler Theory](smc-sampler-theory.md), [SMC Theory](smc-theory.md)

---

**Chopin, N. (2002).** A sequential particle filter method for static models. *Biometrika*, 89(3), 539--552.
[:material-link-variant: DOI](https://doi.org/10.1093/biomet/89.3.539){ .md-button .md-button--primary }

> Introduced **Iterated Batch Importance Sampling (IBIS)** -- an SMC approach to sequential Bayesian parameter learning.
> Referenced in: [SMC Sampler Theory](smc-sampler-theory.md), [PMCMC Theory](pmcmc-theory.md)

---

**Chopin, N., Jacob, P. E., & Papaspiliopoulos, O. (2013).** SMC^2: An efficient algorithm for sequential analysis of state-space models. *Journal of the Royal Statistical Society: Series B*, 75(3), 397--426.
[:material-link-variant: DOI](https://doi.org/10.1111/j.1467-9868.2012.01046.x){ .md-button .md-button--primary }

> Proposed **SMC$^2$** -- a nested SMC algorithm that performs online joint state and parameter estimation in state-space models.
> Referenced in: [SMC Sampler Theory](smc-sampler-theory.md), [PMCMC Theory](pmcmc-theory.md)

---

**Dau, H.-D., & Chopin, N. (2022).** Waste-free Sequential Monte Carlo. *Journal of the Royal Statistical Society: Series B*, 84(1), 114--148.
[:material-link-variant: DOI](https://doi.org/10.1111/rssb.12475){ .md-button .md-button--primary }

> Introduced **Waste-Free SMC** that reuses all MCMC proposals (not just accepted moves), improving particle diversity and efficiency.
> Referenced in: [SMC Sampler Theory](smc-sampler-theory.md)

---

**Neal, R. M. (2001).** Annealed importance sampling. *Statistics and Computing*, 11(2), 125--139.
[:material-link-variant: DOI](https://doi.org/10.1023/A:1008923215028){ .md-button .md-button--primary }

> Introduced tempering-based importance sampling that bridges prior and posterior -- a precursor to SMC tempering.
> Referenced in: [SMC Sampler Theory](smc-sampler-theory.md)

---

**Jasra, A., Stephens, D. A., Doucet, A., & Tsagaris, T. (2011).** Inference for Levy-driven stochastic volatility models via adaptive sequential Monte Carlo. *Scandinavian Journal of Statistics*, 38(1), 1--22.
[:material-link-variant: DOI](https://doi.org/10.1111/j.1467-9469.2010.00723.x){ .md-button .md-button--primary }

> Application of adaptive SMC samplers to Levy-driven SV models demonstrating automatic tempering schedule selection.
> Referenced in: [SMC Sampler Theory](smc-sampler-theory.md), [SV Theory](sv-theory.md)

</div>

---

## 6. Particle MCMC (PMCMC)

Methods combining particle filters with MCMC for Bayesian inference in state-space models.

<div class="reference-list" markdown>

**Andrieu, C., Doucet, A., & Holenstein, R. (2010).** Particle Markov chain Monte Carlo methods. *Journal of the Royal Statistical Society: Series B*, 72(3), 269--342.
[:material-link-variant: DOI](https://doi.org/10.1111/j.1467-9868.2009.00736.x){ .md-button .md-button--primary }

> Foundational paper introducing **PMMH**, **Particle Gibbs**, and **Conditional SMC** -- the three core PMCMC algorithms.
> Referenced in: [PMCMC Theory](pmcmc-theory.md)

---

**Lindsten, F., Jordan, M. I., & Schon, T. B. (2014).** Particle Gibbs with ancestor sampling. *Journal of Machine Learning Research*, 15(1), 2145--2184.
[:material-link-variant: Link](https://jmlr.org/papers/v15/lindsten14a.html){ .md-button .md-button--primary }

> Introduced **Particle Gibbs with Ancestor Sampling (PGAS)** that improves mixing by sampling ancestors for the reference trajectory.
> Referenced in: [PMCMC Theory](pmcmc-theory.md)

---

**Andrieu, C., & Roberts, G. O. (2009).** The pseudo-marginal approach for efficient Monte Carlo computations. *The Annals of Statistics*, 37(2), 697--725.
[:material-link-variant: DOI](https://doi.org/10.1214/07-AOS574){ .md-button .md-button--primary }

> Established the theoretical foundation for **pseudo-marginal MCMC** -- using unbiased likelihood estimates within Metropolis-Hastings.
> Referenced in: [PMCMC Theory](pmcmc-theory.md)

---

**Doucet, A., Pitt, M. K., Deligiannidis, G., & Kohn, R. (2015).** Efficient implementation of Markov chain Monte Carlo when using an unbiased likelihood estimator. *Biometrika*, 102(2), 295--313.
[:material-link-variant: DOI](https://doi.org/10.1093/biomet/asu075){ .md-button .md-button--primary }

> Practical guidance on tuning the variance of the likelihood estimate in PMMH for optimal acceptance rates.
> Referenced in: [PMCMC Theory](pmcmc-theory.md)

---

**Chopin, N., & Singh, S. S. (2015).** On particle Gibbs sampling. *Bernoulli*, 21(3), 1855--1883.
[:material-link-variant: DOI](https://doi.org/10.3150/14-BEJ629){ .md-button .md-button--primary }

> Theoretical analysis of Particle Gibbs including mixing time bounds and comparison with PMMH.
> Referenced in: [PMCMC Theory](pmcmc-theory.md)

</div>

---

## 7. Applications in Economics and Finance

Applications of particle filters and PMCMC to economic and financial models.

<div class="reference-list" markdown>

**Kim, S., Shephard, N., & Chib, S. (1998).** Stochastic volatility: Likelihood inference and comparison with ARCH models. *Review of Economic Studies*, 65(3), 361--393.
[:material-link-variant: DOI](https://doi.org/10.1111/1467-937X.00050){ .md-button .md-button--primary }

> Landmark paper on stochastic volatility estimation establishing benchmark methods and the canonical SV model.
> Referenced in: [SV Theory](sv-theory.md)

---

**Fernandez-Villaverde, J., & Rubio-Ramirez, J. F. (2007).** Estimating macroeconomic models: A likelihood approach. *Review of Economic Studies*, 74(4), 1059--1087.
[:material-link-variant: DOI](https://doi.org/10.1111/j.1467-937X.2007.00437.x){ .md-button .md-button--primary }

> Pioneering application of particle filters to estimate **DSGE models** via likelihood evaluation, opening the door to Bayesian DSGE estimation.
> Referenced in: [DSGE Theory](dsge-theory.md)

---

**Herbst, E. P., & Schorfheide, F. (2015).** *Bayesian Estimation of DSGE Models*. Princeton University Press.
[:material-link-variant: DOI](https://doi.org/10.23943/princeton/9780691161082.001.0001){ .md-button .md-button--primary }

> Comprehensive book on Bayesian estimation of DSGE models covering particle filters, PMCMC, and SMC methods for macroeconomic applications.
> Referenced in: [DSGE Theory](dsge-theory.md), [PMCMC Theory](pmcmc-theory.md)

---

**Shephard, N. (2005).** *Stochastic Volatility: Selected Readings*. Oxford University Press.
[:material-link-variant: Link](https://global.oup.com/academic/product/stochastic-volatility-9780199257201){ .md-button .md-button--primary }

> Curated collection of key papers on stochastic volatility models and estimation methods.
> Referenced in: [SV Theory](sv-theory.md)

---

**Creal, D. (2012).** A survey of sequential Monte Carlo methods for economics and finance. *Econometric Reviews*, 31(3), 245--296.
[:material-link-variant: DOI](https://doi.org/10.1080/07474938.2011.607333){ .md-button .md-button--primary }

> Accessible survey of SMC methods tailored to economists covering particle filters, likelihood estimation, and parameter learning.
> Referenced in: [SV Theory](sv-theory.md), [DSGE Theory](dsge-theory.md)

---

**Herbst, E. P., & Schorfheide, F. (2014).** Sequential Monte Carlo sampling for DSGE models. *Journal of Applied Econometrics*, 29(7), 1073--1098.
[:material-link-variant: DOI](https://doi.org/10.1002/jae.2397){ .md-button .md-button--primary }

> Application of SMC samplers to posterior inference in DSGE models, demonstrating advantages over standard MCMC.
> Referenced in: [DSGE Theory](dsge-theory.md), [SMC Sampler Theory](smc-sampler-theory.md)

</div>

---

## 8. Convergence Theory

Theoretical results on the asymptotic behavior and stability of SMC algorithms.

<div class="reference-list" markdown>

**Del Moral, P., & Miclo, L. (2000).** Branching and interacting particle systems approximations of Feynman-Kac formulae with applications to non-linear filtering. In *Seminaire de Probabilites XXXIV*, Lecture Notes in Mathematics, vol. 1729 (pp. 1--145). Springer.
[:material-link-variant: DOI](https://doi.org/10.1007/BFb0103798){ .md-button .md-button--primary }

> Foundational results on the **stability** and time-uniform convergence of interacting particle systems.
> Referenced in: [Convergence Theory](convergence-theory.md)

---

**Chopin, N. (2004).** Central limit theorem for sequential Monte Carlo methods and its application to Bayesian inference. *The Annals of Statistics*, 32(6), 2385--2411.
[:material-link-variant: DOI](https://doi.org/10.1214/009053604000000698){ .md-button .md-button--primary }

> Established the **Central Limit Theorem for SMC** -- proving $\sqrt{N}$-convergence of particle approximations.
> Referenced in: [Convergence Theory](convergence-theory.md), [SMC Theory](smc-theory.md)

---

**Whiteley, N. (2013).** Stability properties of some particle filters. *The Annals of Applied Probability*, 23(6), 2500--2537.
[:material-link-variant: DOI](https://doi.org/10.1214/12-AAP909){ .md-button .md-button--primary }

> Establishes **time-uniform stability** and mixing properties of particle filters under weak assumptions.
> Referenced in: [Convergence Theory](convergence-theory.md)

---

**Del Moral, P., & Guionnet, A. (2001).** On the stability of interacting processes with applications to filtering and genetic algorithms. *Annales de l'Institut Henri Poincare (B) Probabilites et Statistiques*, 37(2), 155--194.
[:material-link-variant: DOI](https://doi.org/10.1016/S0246-0203(00)01064-5){ .md-button .md-button--primary }

> Exponential stability results for interacting particle systems providing non-asymptotic error bounds.
> Referenced in: [Convergence Theory](convergence-theory.md)

---

**Crisan, D., & Doucet, A. (2002).** A survey of convergence results on particle filtering methods for practitioners. *IEEE Transactions on Signal Processing*, 50(3), 736--746.
[:material-link-variant: DOI](https://doi.org/10.1109/78.984773){ .md-button .md-button--primary }

> Practical survey of convergence results including $L^p$ bounds and almost sure convergence for particle filters.
> Referenced in: [Convergence Theory](convergence-theory.md)

</div>

---

## 9. Software and Tutorials

Tutorials, software descriptions, and implementation references.

<div class="reference-list" markdown>

**Murray, L. M. (2013).** Bayesian state-space modelling on high-performance hardware using LibBi. *arXiv preprint arXiv:1306.3277*.
[:material-link-variant: Link](https://arxiv.org/abs/1306.3277){ .md-button .md-button--primary }

> Description of LibBi -- a software package for Bayesian inference in state-space models using SMC on GPUs and clusters.
> Referenced in: [Particle Filter Theory](particle-filter-theory.md)

---

**Doucet, A., & Johansen, A. M. (2011).** A tutorial on particle filtering and smoothing: Fifteen years later. In *The Oxford Handbook of Nonlinear Filtering*. Oxford University Press.
[:material-link-variant: Link](https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf){ .md-button .md-button--primary }

> Widely-cited tutorial providing a self-contained introduction to particle filtering and smoothing with pseudocode.
> Referenced in: [Particle Filter Theory](particle-filter-theory.md), [Smoothing Theory](smoothing-theory.md)

---

**Sarkka, S. (2013).** *Bayesian Filtering and Smoothing*. Cambridge University Press.
[:material-link-variant: DOI](https://doi.org/10.1017/CBO9781139344203){ .md-button .md-button--primary }

> Accessible textbook covering Kalman filters, particle filters, and smoothers with a unified Bayesian perspective.
> Referenced in: [Particle Filter Theory](particle-filter-theory.md), [Smoothing Theory](smoothing-theory.md)

---

**Naesseth, C. A., Lindsten, F., & Schon, T. B. (2019).** Elements of Sequential Monte Carlo. *Foundations and Trends in Machine Learning*, 12(3), 307--392.
[:material-link-variant: DOI](https://doi.org/10.1561/2200000074){ .md-button .md-button--primary }

> Modern tutorial on SMC emphasizing the algorithmic building blocks and connections to variational inference.
> Referenced in: [SMC Theory](smc-theory.md), [SMC Sampler Theory](smc-sampler-theory.md)

</div>

---

## 10. Adaptive Methods and ESS

Methods for adaptive tuning and effective sample size monitoring in SMC.

<div class="reference-list" markdown>

**Liu, J. S., & Chen, R. (1998).** Sequential Monte Carlo methods for dynamic systems. *Journal of the American Statistical Association*, 93(443), 1032--1044.
[:material-link-variant: DOI](https://doi.org/10.1080/01621459.1998.10473765){ .md-button .md-button--primary }

> Formalized Sequential Monte Carlo for dynamic systems and introduced key concepts around weight monitoring.
> Referenced in: [SMC Theory](smc-theory.md), [Particle Filter Theory](particle-filter-theory.md)

---

**Jasra, A., Stephens, D. A., & Holmes, C. C. (2007).** On population-based simulation for static inference. *Statistics and Computing*, 17(3), 263--279.
[:material-link-variant: DOI](https://doi.org/10.1007/s11222-007-9028-9){ .md-button .md-button--primary }

> Discussion of adaptive tempering schedules using ESS thresholds for population-based methods.
> Referenced in: [SMC Sampler Theory](smc-sampler-theory.md), [Convergence Theory](convergence-theory.md)

</div>

---

## How to Cite particlefilterbox

If you use particlefilterbox in your research, please cite:

```bibtex
@software{particlefilterbox,
  title  = {particlefilterbox: Particle Filters, SMC, and PMCMC for
            Nonlinear State-Space Models},
  author = {NodeSEcon},
  url    = {https://github.com/nodesecon/particlefilterbox},
  year   = {2024}
}
```

---

!!! tip "Contributing References"
    If you notice a missing reference or would like to suggest additions, please
    [open an issue](https://github.com/nodesecon/particlefilterbox/issues) or
    submit a pull request.
