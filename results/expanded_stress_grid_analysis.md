# GDS Expanded Bayesian Robust-Inference Stress Grid

## Run record

- Workflow: [Bayesian Robustness Stress Grid](https://github.com/TheRealoliGARCH/GainfulDistillation/actions/runs/43587199084)
- Seed: `20261115`
- Data-generating processes: 8
- Sample sizes: 40, 80, 160
- Prior configurations: 6
- Replications per cell: 200
- Total cells: 144
- Total posterior fits: 28,800
- Midpoint quadrature points: 2,048
- Parameter interval: [-8, 8]
- True location parameter: 0.5
- Working likelihood: Normal(theta, 1)

The raw cell-level results are archived in [`expanded_stress_grid_seed_20261115.json`](expanded_stress_grid_seed_20261115.json).

This is a GDS adaptation using the proper Sinc-modulated prior; it is not an exact numerical replication of Li and Rice (2024). The balanced-loss correction follows the scalar specialization of Proposition 3.1 in Li and Rice, *A Bayesian “Sandwich” for Variance Estimation*, Statistical Science 39(4), 589–600, DOI: 10.1214/24-STS935.

## Design

The eight data-generating processes are:

1. Normal errors with SD 1 (correctly specified working model).
2. Normal errors with SD 1.5.
3. Normal errors with SD 2.
4. Student-t(5) errors standardised to unit variance.
5. Laplace errors standardised to unit variance.
6. Random-design heteroscedastic errors with conditional variance 0.5 + 3 X^2, X ~ Uniform(-1, 1).
7. A 95%/5% Gaussian contamination mixture standardised to unit marginal variance.
8. Centred, standardised skewed Gamma errors.

The six proper Sinc-prior configurations are baseline, concentrated, diffuse, spectrally modulated, truth-centred (an oracle benchmark in this simulation), and shifted away from the true parameter.

The same simulated dataset is evaluated under all six priors within each scenario/sample-size cell (common random numbers). Observations are independent across units. The heteroscedastic case uses iid random covariate-outcome pairs; no serial or cluster dependence is included.

Three intervals are compared:

- Ordinary Bayesian posterior credible interval.
- Bayesian balanced-loss robust Wald interval, centred at the posterior mean.
- Frequentist sandwich Wald interval, centred at the sample mean/MLE.

The robust intervals are repeated-sampling intervals, not posterior credible intervals. The score and average empirical Fisher information under the working Normal(theta, 1) likelihood are s_i(theta) = y_i - theta and I_n(theta) = 1. The experiment uses exact sufficient-statistic expressions for the likelihood and mean squared score.

## Aggregate results by prior

The following figures are unweighted averages over the 24 scenario/sample-size cells for each prior. Each cell has 200 replications.

| Prior | Mean absolute posterior-mean bias | Mean RMSE | Mean posterior 95% coverage | Mean balanced-loss robust Wald coverage | Mean frequentist sandwich Wald coverage | Mean robust/theoretical variance ratio |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.026 | 0.133 | 90.4% | 94.9% | 95.0% | 0.942 |
| Concentrated | 0.139 | 0.177 | 65.4% | 75.5% | 95.0% | 0.743 |
| Diffuse | 0.008 | 0.138 | 89.8% | 95.0% | 95.0% | 0.986 |
| Spectrally modulated | 0.049 | 0.133 | 89.5% | 94.1% | 95.0% | 0.893 |
| Truth-centred oracle | 0.007 | 0.131 | 90.4% | 95.5% | 95.0% | 0.947 |
| Shifted away | 0.110 | 0.163 | 81.2% | 86.9% | 95.0% | 0.855 |

The frequentist coverage column is the same across prior specifications because the frequentist interval is computed from the same simulated data and does not depend on the Bayesian prior.

## Baseline-prior results by data-generating process

Coverage figures below are averaged over the three sample sizes.

| Data-generating process | Posterior credible coverage | Balanced-loss robust Wald coverage | Frequentist sandwich Wald coverage |
|---|---:|---:|---:|
| Normal, SD 1 | 95.7% | 95.3% | 95.8% |
| Normal, SD 1.5 | 83.3% | 96.0% | 96.2% |
| Normal, SD 2 | 71.8% | 95.5% | 95.3% |
| Student-t(5), unit variance | 95.3% | 95.5% | 94.0% |
| Laplace, unit variance | 94.5% | 93.5% | 93.8% |
| Heteroscedastic, mean variance 1.5 | 91.2% | 96.2% | 96.0% |
| Contaminated Normal, unit variance | 96.2% | 95.2% | 95.0% |
| Skewed Gamma, unit variance | 94.8% | 92.0% | 93.5% |

The most pronounced working-likelihood misspecification result is the Normal SD 2 scenario: the ordinary posterior credible interval covers the true parameter in 71.8% of replications, compared with 95.5% for the balanced-loss robust Wald interval and 95.3% for the frequentist sandwich interval.

For the Normal SD 1.5 scenario, the corresponding coverages are 83.3%, 96.0% and 96.2%.

## Prior sensitivity is a separate source of failure

The balanced-loss correction adjusts posterior variance using a posterior expectation of the score-information lack-of-fit term. It does not remove bias in the posterior mean caused by an overly concentrated or badly located prior.

This is visible in the aggregate results:

- The concentrated prior has mean absolute posterior-mean bias of 0.139 and mean robust-Wald coverage of 75.5%.
- The deliberately shifted prior has mean absolute bias of 0.110 and mean robust-Wald coverage of 86.9%.
- The baseline and diffuse priors achieve 94.9% and 95.0% mean robust-Wald coverage.
- The truth-centred prior is an oracle reference only; it is not a deployable prior unless that location can be justified independently.

For the concentrated prior, mean robust-Wald coverage improves from 66.1% at n=40 to 75.6% at n=80 and 84.9% at n=160, but remains below nominal coverage at the largest sample size. The shifted-away prior improves from 81.7% to 87.5% to 91.6%, respectively. These patterns are consistent with prior-induced shrinkage bias diminishing as likelihood information accumulates, but not disappearing quickly at these sample sizes.

## Interpretation and limitations

1. Under the baseline, diffuse and truth-centred priors, the scalar balanced-loss Bayesian Wald interval tracks the frequentist sandwich Wald interval reasonably closely in this design.
2. Under Normal variance misspecification, the ordinary posterior credible interval undercovers materially, while the robust Bayesian and frequentist sandwich intervals remain close to nominal coverage.
3. The correction does not fix prior-induced point-estimate bias. A robust variance multiplier cannot be treated as a general debiasing method.
4. The concentrated and shifted priors demonstrate why prior sensitivity must be part of external validation, rather than an optional cosmetic diagnostic.
5. Coverage estimates are finite Monte Carlo estimates. Each cell has 200 replications; cell-level coverage MCSEs are reported in the JSON. Small differences of a few percentage points should not be over-interpreted.
6. This experiment does not cover dependent, clustered, time-series or high-dimensional settings. The scalar balanced-loss derivation used here assumes independent observations.
7. The results are specific to this Sinc prior family, parameter interval, likelihood and simulation design. They do not establish universal superiority of the Sinc prior or balanced-loss procedure.

## Next research steps

- Increase replications and add coverage confidence intervals for selected cells.
- Examine prior hyperparameter calibration and prior predictive diagnostics.
- Extend the posterior and balanced-loss correction to multivariate parameters.
- Investigate clustered and serially dependent observations with a derivation appropriate to those dependence structures.
- Reproduce the NHANES application with documented data extraction and independent replication.
