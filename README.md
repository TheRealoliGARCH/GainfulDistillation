# The Theory of Gainful Distillation

## The Science of Learning and Intelligence

This repository develops the **Gainful Distillation System (GDS)**: a reciprocal teacher–student learning architecture designed to increase verified collective capability rather than merely reproduce a teacher's outputs.

The initial architecture contains one teacher and five students. The students form a complete graph, K5, augmented by a distinct circumferential overlay. The teacher is connected to every student. The fixed structural topology is kept separate from directed, adaptive knowledge-transmission strengths.

## Central components

- **Adaptive knowledge transmission:** distinct teacher-to-student, student-to-teacher, and peer-to-peer channels.
- **Student Learning Dynamics Estimator (SLDE):** estimates student-, domain-, and method-specific learning responses, with separate tracking of retention, transfer, Breadth, Depth, and uncertainty.
- **Specialist task allocation:** routes tasks to individual students or collaborative coalitions.
- **Independent learning:** protects each student's capacity to investigate beyond teacher and peer information.
- **Resource allocation:** allocates compute according to expected marginal verified gain, covariance, cost, and risk.
- **Stability and diversity controls:** constrain error propagation and reduce epistemic collapse.
- **External validation and governance:** distinguish transmission, investigation, verification, and integration.
- **Independent adjudication:** retain claim provenance and require distinct independent review groups before verification.
- **Robust covariance diagnostics:** compute estimating-equation sandwich covariance when sensitivity and variability matrices are supplied.
- **Band-limited modelling:** provide a Sinc covariance kernel for Gaussian-process models.
- **Proper Sinc-based priors:** provide a non-negative, analytically normalised, Sinc-modulated prior with finite second moment.
- **Bayesian posterior estimation:** provide a stable one-dimensional posterior reference engine using explicit finite-domain midpoint quadrature.
- **Bayesian robust variance:** apply the scalar balanced-inference-loss correction to the posterior variance using observation-level scores and empirical Fisher information.

## Principal research conjecture

The Gainful Distillation Conjecture proposes that, under suitable conditions, an adaptive and independently validated teacher–student network can achieve greater net welfare than its original teacher under an equivalent resource budget.

This is a falsifiable research conjecture, not an established empirical result. The repository distinguishes formal definitions and invariants, proposed hypotheses, and experimentally validated findings.

## Repository structure

| Path | Description |
|---|---|
| `1. The Theory of Gainful Distillation.pdf` | Published theoretical treatise |
| `src/gds_core/graph.py` | Fixed six-agent structural topology |
| `src/gds_core/transmission.py` | Directed adaptive transmission channels |
| `src/gds_core/slde.py` | Online Student Learning Dynamics Estimator |
| `src/gds_core/routing.py` | Probabilistic task routing and coalition shortlists |
| `src/gds_core/resource_allocation.py` | Constrained quadratic resource allocation |
| `src/gds_core/sinc_kernel.py` | Band-limited Sinc covariance kernel for Gaussian-process models |
| `src/gds_core/sinc_prior.py` | Proper non-negative Sinc-modulated univariate and independent-product priors |
| `src/gds_core/posterior.py` | One-dimensional posterior estimation with log-likelihoods, quadrature and posterior summaries |
| `src/gds_core/balanced_loss.py` | Scalar Bayesian balanced-loss variance correction using posterior score-information averaging |
| `experiments/expanded_stress_grid.py` | Multi-distribution, multi-sample-size and multi-prior Monte Carlo stress grid |
| `experiments/normal_mean_misspecification.py` | Reproducible Monte Carlo study under correct and misspecified Normal likelihood variance |
| `src/gds_core/sandwich.py` | General estimating-equation sandwich covariance calculation |
| `src/gds_core/verification.py` | Provenance-aware independent claim adjudication |
| `tests/` | Unit tests for graph, transmission, and SLDE |
| `pyproject.toml` | Python package and test configuration |
| `.github/workflows/python-tests.yml` | Automated Python test workflow |
| `LICENSE` | Repository license |

## GDS-Core: structural conventions

The node order is (T, S1, S2, S3, S4, S5). The student subgraph contains all ten edges of K5. The additional circumferential overlay adds one further unit of multiplicity to each of the five perimeter pairs.

Consequently:
- There are six vertices and fifteen distinct undirected vertex pairs.
- The five teacher–student pairs each have multiplicity one.
- Five student perimeter pairs have multiplicity two; five student diagonals have multiplicity one.
- Total undirected edge multiplicity is twenty.
- The teacher's weighted degree is five; each student's weighted degree is seven.

The structural adjacency matrix is symmetric and fixed. The transmission matrix is directed and dynamic. Its convention is W[source][target]; teacher-to-student and student-to-teacher strengths are independently parameterised. Peer transmission is the structural multiplicity multiplied by a directed peer-channel multiplier.

## SLDE: estimation conventions

SLDE maintains a separate profile for each student × domain × instructional-method combination. The initial response model is:

`performance_change = learning_rate × instructional_dose + innovation`

The learning-rate coefficient is updated by scalar recursive least squares (RLS). A forgetting factor of one gives ordinary RLS; values below one discount older observations. Retention, transfer, Breadth, and Depth are tracked as separate optional bounded measurements, using online means and sample variances. Missing measurements are not imputed as zero.

Performance and auxiliary metric scores are currently expected to be normalised to [0, 1], and instructional dose must be strictly positive. These are implementation conventions, not claims that every real-world learning measure is naturally bounded or directly comparable. Empirical work must establish valid measurement instruments, score comparability, and the causal interpretation of instructional dose.

## Task routing: current conventions

The routing layer accepts a task request and a set of candidate student profiles. Each candidate supplies an estimated task-relevant gain, an uncertainty measure, optional Breadth and Depth fit scores, and an execution-cost estimate. An SLDE profile can be converted into a candidate: the RLS learning-rate estimate supplies expected gain, its reported standard error supplies an exploration signal, and observed Breadth/Depth means supply optional fit measures.

For candidate i and task q, the implemented score is:

`score_i = gain_i + w_B b_q B_i + w_D d_q D_i + w_U u_i - w_C c_i`

where b_q and d_q are task requirements; B_i and D_i are candidate fit estimates; u_i is the uncertainty signal; c_i is estimated execution cost; and the w terms are non-negative configuration weights. When a fit measure is unobserved, the router uses an explicit configurable neutral prior rather than treating missing data as zero.

Scores are converted to probabilities with a temperature-scaled softmax. A configurable exploration floor guarantees a minimum probability for every candidate, subject to `n × floor < 1`. The router returns the full probability distribution and a deterministic top-k shortlist for individual or coalition routing. A downstream scheduler may sample from the returned distribution; the current shortlist itself is not a stochastic assignment.

These are routing heuristics, not validated optimal policies. The uncertainty bonus is an exploration proxy, not a formal value-of-information calculation. Expected gain, fit, uncertainty, and cost must be calibrated to comparable units before deployment. Coalition shortlisting does not yet model complementarity, coordination overhead, or joint performance.

## Resource allocation: current conventions

The resource allocator solves the concave quadratic programme:

`max_r  mu.T r - (lambda/2) r.T Sigma r - (eta/2) ||r||^2`

subject to a total budget and per-student lower and upper allocation shares. Here, mu contains expected marginal gains, Sigma is a symmetric positive-semidefinite covariance matrix, lambda is risk aversion, and eta is a regularisation coefficient. The lower share provides an optional exploration floor; the upper share limits concentration.

The implementation uses projected gradient ascent over a capped simplex, with a conservative step based on an upper bound for the Hessian's spectral norm. A dependency-free Jacobi rotation check validates covariance positive semidefiniteness. When the quadratic curvature is zero, the linear programme is solved directly. The result reports allocations, objective components, budget use, iteration count, convergence status, and a projected-gradient diagnostic.

A result with `converged=False` is an iteration-limited approximation and must not be represented as an optimum. Covariance estimates and expected gains must be calibrated to compatible units and estimated from appropriate held-out or historical data. The allocator does not infer covariance from SLDE standard errors, and it does not yet impose group-level, task-specific, or coalition-complementarity constraints.

## Bayesian and external-validation research layer

Two repository manuscripts motivate this layer:

- *A Novel Approach to Bayesian Estimation using the Sandwich Theorem and the Sinc Function*.
- *A Second Novel Approach to Bayesian Estimation using the Sandwich Theorem and the Sinc Function*.

The implementation separates four mathematically distinct ideas:

1. **Sandwich covariance:** `sandwich.py` computes the general estimating-equation covariance `A^{-1} B A^{-T}`, with an optional division by sample size. It requires model-specific sensitivity and variability inputs. This is not a complete implementation of the Bayesian balanced-loss estimator discussed by Li and Rice.
2. **Sinc spectral modelling:** `sinc_kernel.py` implements the stationary band-limited Sinc covariance kernel for Gaussian-process models.
3. **Proper Sinc-based prior:** `sinc_prior.py` implements a non-negative, normalised prior inspired by the Sinc-modulated prior in the second manuscript. It uses a fourth-power Sinc envelope and bounded cosine modulation. This is an explicit mathematical regularisation of the manuscript's proposal, not a verbatim implementation of its raw expression.
4. **Scalar Bayesian balanced-loss correction:** `balanced_loss.py` implements the one-dimensional specialization of the Li–Rice balanced inference loss. The posterior point estimate remains the posterior mean; the robust variance is the posterior variance multiplied by the posterior expectation of the mean squared observation score divided by the scalar empirical Fisher information. The caller supplies model-specific observation scores and information at each posterior grid point. For scalar models it may supply either the individual score vector or an exact `ScalarScoreSummary` containing its sample size and mean squared score. This assumes independent observations and is not yet a multivariate implementation.
5. **Independent adjudication:** `verification.py` records the origin, claim, evidence references, methods, independent reviews, and decision. An originating agent cannot validate its own finding. Each independence group contributes at most one review. Quorums are configurable governance conventions, not universal statistical thresholds.

### Proper Sinc-modulated prior

Let `sinc(u) = sin(pi*u)/(pi*u)`, with `sinc(0)=1`. For a scalar parameter `theta`, define `x = theta - mu`, bandwidth multiplier `Delta > 0`, scale `sigma > 0`, centre frequency `xi0 >= 0`, and modulation depth `rho in [0, 1]`. The prior is

```
pi(theta) = sinc(Delta*x/sigma)^4
            * (1 + rho*cos(2*pi*xi0*x)) / Z
```

with exact normaliser

```
Z = (sigma/Delta) * (2/3 + rho*Lambda4(xi0*sigma/Delta)).
```

The compactly supported spectral-overlap function is

```
Lambda4(t) = 2/3 - t^2 + |t|^3/2,       |t| <= 1
            (2 - |t|)^3/6,             1 < |t| < 2
            0,                         |t| >= 2.
```

The identity follows from the Fourier transform of `sinc(u)^4`, which is the convolution of four rectangular spectra (equivalently, the convolution of two triangular spectra). The cosine modulation shifts this spectrum to two sidebands. Since `sinc^4 >= 0` and `1 + rho*cos(...) >= 0`, the density is non-negative. The stated normaliser makes its integral exactly one. Its tails are `O(|theta|^-4)`, giving a finite second moment, while its Fourier transform remains compactly supported.

`SincModulatedPrior` evaluates the univariate density and log-density. `IndependentSincPrior` forms a product across coordinates and therefore explicitly assumes prior independence. Correlated multivariate priors require a separate joint construction. The implementation does not claim to implement the manuscript's original unmodified Sinc-times-cosine expression, which need not be non-negative.

### Remaining mathematical qualifications

Pointwise bounds on posterior densities do not alone establish weak convergence to a Dirac measure. A rigorous posterior-concentration argument should use probability measures, neighbourhood probabilities, or an appropriate weak-convergence criterion. The balanced inference loss needs model-specific score and information calculations. The scalar implementation follows Proposition 3.1 of Li and Rice (2024); its Bayes point estimate, correction factor, and variance rule are distinct from the ordinary frequentist sandwich covariance calculation in `sandwich.py`.

The Sinc prior implemented here is a proper prior primitive. The module `posterior.py` now adds a one-dimensional Bayesian posterior reference engine. It accepts a log-likelihood callback, conditions the prior on an explicitly declared finite parameter interval, and uses equally spaced midpoint quadrature with log-sum-exp normalisation. It returns cell probabilities and densities, posterior mean and variance, a central credible interval, and the log marginal likelihood under the prior conditioned on that interval.

The finite interval is an explicit modelling choice, not a silent numerical truncation. The result is conditional on the parameter lying within the declared bounds. Grid-refinement checks are necessary for each application; this deterministic one-dimensional implementation is not a multivariate posterior sampler.

`balanced_loss_variance_1d` accepts a callback returning the per-observation score vector and the scalar empirical average negative Hessian at each posterior grid point. It computes the posterior expectation of the score-information correction and multiplies that by the posterior variance, following the scalar specialization of Li and Rice (2024), Proposition 3.1. The resulting robust standard error is designed to reflect repeated-sampling variability under model misspecification; it is not a Bayesian credible interval. The derivation in Li and Rice assumes independent observations. Dependence, clustered data, and multivariate parameters require appropriate extensions.

The current components remain reference primitives. Formal posterior-concentration diagnostics, multivariate inference, and empirical calibration remain separate research tasks. The code does not assert that the manuscripts' reported empirical improvements have already been independently established.

### Monte Carlo experiment

The experiment `experiments/normal_mean_misspecification.py` is a supplemental high-variance study. It compares ordinary posterior uncertainty with the scalar balanced-loss robust standard error under four data-generating processes: correctly specified Normal errors with SD 1; Normal variance misspecification with SD 2; Student-t(5) errors rescaled to variance 4; and independent heteroskedastic Normal errors with average variance 4. It crosses sample sizes 40, 80 and 160 with four Sinc-prior configurations. Identical simulated datasets are used for all priors within each scenario/sample-size cell (common random numbers).

The standard run contains 48 cells, 100 replications per cell, and 512 midpoint-quadrature points. It records bias, RMSE, posterior and robust-Wald coverage with Monte Carlo standard errors, empirical and oracle variance ratios, and interval widths. The working likelihood is independent `Normal(theta, 1)`. The experiment uses the GDS Sinc prior and is an adaptation of, not an exact replication of, the conjugate Normal-prior example in Li and Rice (2024).

Run it after installing the package:

```bash
python experiments/normal_mean_misspecification.py \
  --sample-sizes 40 80 160 \
  --repetitions 100 \
  --points 512 \
  --seed 2026111401
```

The robust Wald interval is a repeated-sampling interval and must not be described as a posterior credible interval. The ordinary posterior interval and robust Wald interval answer different inferential questions.

### Expanded robustness stress grid

The experiment `experiments/expanded_stress_grid.py` broadens the Normal-scale study across:

- Normal errors with working-model standard deviation 1, 1.5 and 2.
- Standardised Student-t(5) errors.
- Standardised Laplace errors.
- Random-design heteroscedastic errors with conditional variance `0.5 + 3 X^2`, where `X ~ Uniform(-1,1)`.
- A standardised 95%/5% Gaussian contamination mixture.
- Centred and standardised skewed Gamma errors.

It crosses these eight data-generating processes with sample sizes 40, 80 and 160, and six prior configurations: baseline, concentrated, diffuse, spectrally modulated, a truth-centred oracle benchmark, and a deliberately shifted prior-location stress case. The default design uses 200 replications per cell and 2,048 midpoint quadrature points, giving 144 cells in total. Identical simulated datasets are evaluated under all prior configurations within each scenario/sample-size cell (common random numbers), allowing more efficient prior-sensitivity comparisons.

The working likelihood remains `Normal(theta,1)`. For the intercept-only location parameter, the scalar score is `y_i - theta` and the average empirical Fisher information is one. The experiment uses exact sufficient-statistic identities for the log likelihood and mean squared score, so it does not approximate those quantities by subsampling. Each cell reports bias, RMSE, Monte Carlo standard error of bias, posterior and robust Wald coverage with coverage MCSEs, interval widths, empirical sandwich variance comparison, theoretical variance comparison, and posterior boundary-cell mass.

All observations are independent across units. The heteroscedastic case uses iid covariate-outcome pairs; no serial or cluster dependence is simulated. The balanced-loss robust interval and frequentist sandwich interval are Wald-type repeated-sampling intervals, not posterior credible intervals. Coverage and variance results are finite-simulation estimates, not guarantees. The experiment is a GDS adaptation using the Sinc prior, not an exact replication of Li and Rice.

Run it after installing GDS:

```bash
python experiments/expanded_stress_grid.py \
  --sample-sizes 40 80 160 \
  --repetitions 200 \
  --points 2048 \
  --seed 20261115 \
  --output expanded_stress_grid_results.json
```

The `.github/workflows/bayesian-stress-grid.yml` workflow runs the standard design and stores the JSON output as a GitHub Actions artifact.
The completed seed-20261115 run is also archived in `results/expanded_stress_grid_seed_20261115.json`; the interpretation and principal findings are recorded in `results/expanded_stress_grid_analysis.md`.

### Supplemental high-variance and prior-paired grid

The seed-2026111401 study complements the 144-cell stress grid above by deliberately raising the variance in the misspecified Normal, Student-t(5), and heteroskedastic cases. Its 48 results are archived in `results/expanded_bayesian_misspecification_seed_2026111401.json`, and the run is available at [Bayesian Misspecification Study](https://github.com/TheRealoliGARCH/GainfulDistillation/actions/runs/43588473245).

For the centered diffuse prior at sample size 80, the results were:

| Data-generating process | Posterior 95% coverage | Balanced-loss robust Wald 95% coverage | Robust/oracle variance ratio |
|---|---:|---:|---:|
| Normal, SD 1 | 96% | 93% | 0.958 |
| Normal, SD 2 | 61% | 97% | 0.941 |
| Student-t(5), variance 4 | 65% | 95% | 0.953 |
| Heteroskedastic Normal, average variance 4 | 60% | 93% | 0.925 |

The robust variance correction materially improves coverage under these variance-misspecified cases, and its average variance approaches the oracle repeated-sampling variance as the sample size rises from 40 to 160. These findings are specific to the simulated design and have finite Monte Carlo uncertainty.

The prior-sensitivity results expose a separate limitation: at sample size 40 under the correctly specified Normal case, shifted priors produced posterior-mean bias around -0.18 and robust-Wald coverage of only 75–76%, compared with 96% for the centered diffuse prior. A variance correction does not automatically remove prior-induced location bias. GDS therefore needs to diagnose posterior mean bias and prior sensitivity separately from robust variance calibration.


The first recorded run used seed 20261114, sample size 80, 200 replications per scenario, and 1,024 quadrature points. Results are archived at `results/normal_mean_misspecification_seed_20261114.json`.

| True observation SD | Mean posterior SD | Mean robust SE | Mean sandwich SE | Posterior 95% coverage | Robust Wald 95% coverage | Robust/sandwich variance ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 (correct) | 0.1093 | 0.1093 | 0.1119 | 0.925 | 0.930 | 0.955 |
| 2.0 (misspecified) | 0.1092 | 0.2149 | 0.2211 | 0.700 | 0.945 | 0.945 |

Monte Carlo standard errors for the posterior and robust coverage estimates, respectively, were 0.0186 and 0.0180 in the correctly specified scenario, and 0.0324 and 0.0161 in the misspecified scenario.

In this run, the ordinary posterior interval materially under-covered when the working likelihood variance was wrong, while the balanced-loss robust Wald interval was close to nominal coverage and the average robust variance was within roughly 5.5% of the frequentist sandwich variance. These are results for this particular simulation design and seed, not a general performance guarantee. More repetitions, alternative data-generating processes, prior-sensitivity analysis, and independent replication remain necessary.

## Development and testing

Python 3.10 or later is required:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

The Python workflow runs tests on pushes and pull requests that change source code, tests, or project configuration.

## Research discipline

1. Evaluate all architectures under comparable compute and data budgets.
2. Measure generalisation on held-out tasks and distributions.
3. Treat student learning profiles as estimates with uncertainty, not fixed labels.
4. Verify novelty independently; disagreement alone is not innovation.
5. Report capability, cost, energy, risk, coordination overhead, and distributional effects.
6. Preserve audit trails for data, models, interventions, validation, and updates.
7. Report negative and null results as well as positive findings.

## Implementation status

**Implemented:** GDS-Core structural graph, directed transmission-channel representation, online SLDE, probabilistic task routing, constrained quadratic resource allocation, Sinc covariance kernel, proper Sinc-modulated prior, one-dimensional Bayesian posterior reference engine, scalar Bayesian balanced-loss variance correction, frequentist sandwich covariance calculation, independent verification adjudication, and reproducible Normal-mean and expanded robustness Monte Carlo experiments; core components and experiment entry points have automated tests.

**Next:** extend posterior computation and balanced-loss correction to multivariate parameters; develop posterior-concentration and prior-predictive diagnostics; address prior-induced bias; replicate the NHANES application; and evaluate GDS end-to-end under matched compute and data budgets.

The PDF treatise provides the theoretical specification and empirical research programme. Claims of performance superiority remain subject to implementation, replication, and independent validation.
