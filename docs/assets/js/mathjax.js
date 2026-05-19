window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    tags: "ams",
    tagIndent: "0.8em",
    multlineWidth: "85%",
    macros: {
      // Particle set notation: \particles{N}{i} -> {x^{(i)}}_{i=1}^{N}
      particles: [
        "\\left\\{ x^{(#2)} \\right\\}_{#2=1}^{#1}",
        2,
        "N",
      ],
      // Normalized weights: \weights{N}{i} -> {w^{(i)}}_{i=1}^{N}
      weights: [
        "\\left\\{ w^{(#2)} \\right\\}_{#2=1}^{#1}",
        2,
        "N",
      ],
      // Effective Sample Size
      ess: [
        "\\mathrm{ESS} = \\frac{1}{\\sum_{#1=1}^{N} \\left( w_{t}^{(#1)} \\right)^2}",
        1,
        "i",
      ],
      // Proposal distribution: \proposal -> q(x_t | x_{t-1}, y_t)
      proposal: [
        "q\\!\\left( x_{#1} \\,\\middle|\\, x_{#2}, y_{#1} \\right)",
        2,
        "t",
      ],
      // Target distribution: \target -> p(x_t | y_{1:t})
      target: [
        "p\\!\\left( x_{#1} \\,\\middle|\\, y_{1:#1} \\right)",
        1,
        "t",
      ],
      // Resampling operator
      resampling: [
        "\\mathcal{R}\\!\\left( \\left\\{ x^{(i)}, w^{(i)} \\right\\}_{i=1}^{#1} \\right)",
        1,
        "N",
      ],
      // Common shorthands
      E: ["\\mathbb{E}"],
      Var: ["\\mathrm{Var}"],
      Cov: ["\\mathrm{Cov}"],
      R: ["\\mathbb{R}"],
      N: ["\\mathcal{N}"],
      iid: ["\\overset{\\text{iid}}{\\sim}"],
      argmin: ["\\operatorname*{arg\\,min}"],
      argmax: ["\\operatorname*{arg\\,max}"],
      given: ["\\,\\middle|\\,"],
    },
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
  svg: {
    fontCache: "global",
    displayAlign: "left",
    displayIndent: "2em",
  },
  loader: {
    load: ["[tex]/ams"],
  },
};

document$.subscribe(() => {
  MathJax.typesetPromise();
});
