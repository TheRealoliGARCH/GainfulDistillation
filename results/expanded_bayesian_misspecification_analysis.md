# Supplemental High-Variance Bayesian Robustness Study

## Run record

- Workflow: [Bayesian Misspecification Study](https://github.com/TheRealoliGARCH/GainfulDistillation/actions/runs/43588473245)
- Seed: `2026111401`
- Data-generating processes: 4
- Sample sizes: 40, 80, 160
- Sinc-prior configurations: 4
- Replications per cell: 100
- Total cells: 48
- Total posterior fits: 4,800
- Midpoint quadrature points: 512
- Parameter interval: [-8, 8]
- True location parameter: 0.5
- Working likelihood: independent Normal(theta, 1)

Raw results are archived in [`expanded_bayesian_misspecification_seed_2026111401.json`](expanded_bayesian_misspecification_seed_2026111401.json).

This GDS experiment supplements the larger 144-cell stress grid. It is an adaptation using the GDS Sinc prior, not an exact replication of the conjugate Normal-prior worked example in Li and Rice (2024), *A Bayesian “Sandwich” for Variance Estimation*, Statistical Science 39(4), 589–600, DOI 10.1214/24-STS935.

## Design

The four data-generating processes are:

1. Normal errors with SD 1 (correctly specified working model).
2. Normal errors with SD 2 (variance misspecification).
3. Student-t(5) errors rescaled to variance 4 (heavy tails and variance misspecification).
4. Independent heteroskedastic Normal errors with average conditional variance 4.

The four Sinc priors are centered diffuse, shifted diffuse, shifted concentrated, and spectrally modulated diffuse. Each dataset is evaluated under all four priors using common random numbers, so prior comparisons are paired on identical simulated observations.

The working likelihood is independent Normal(theta, 1). For this location model, the individual score is s_i(theta) = y_i - theta and the average empirical Fisher information is I_n(theta) = 1. The scalar balanced-loss correction uses the exact mean squared score. The ordinary posterior credible interval and the robust Wald interval are reported separately.

## Centered-diffuse prior: results at n = 80

| Data-generating process | Posterior 95% coverage | Balanced-loss robust Wald 95% coverage | Mean robust SE | Oracle SE | Robust/oracle variance ratio |
|---|---:|---:|---:|---:|---:|
| Normal, SD 1 | 96% | 93% | 0.1092 | 0.1118 | 0.958 |
| Normal, SD 2 | 61% | 97% | 0.2161 | 0.2236 | 0.941 |
| Student-t(5), variance 4 | 65% | 95% | 0.2166 | 0.2236 | 0.953 |
| Heteroskedastic Normal, average variance 4 | 60% | 93% | 0.2140 | 0.2236 | 0.925 |

The corresponding robust-to-empirical-frequentist-sandwich variance ratios were 0.954, 0.946, 0.945 and 0.944, respectively.

Under the three variance-four scenarios, the ordinary posterior credible interval under-covered materially (60–65% coverage in this run). The robust Wald interval was much closer to its nominal 95% level (93–97%). The average robust variance was about 4.7–7.5% below the oracle variance at n = 80.

## Sample-size behaviour under the centered-diffuse prior

Each entry below reports posterior coverage / robust Wald coverage / robust-to-oracle variance ratio.

| Data-generating process | n = 40 | n = 80 | n = 160 |
|---|---:|---:|---:|
| Normal, SD 1 | 98% / 96% / 0.890 | 96% / 93% / 0.958 | 94% / 94% / 0.994 |
| Normal, SD 2 | 67% / 94% / 0.874 | 61% / 97% / 0.941 | 65% / 97% / 0.960 |
| Student-t(5), variance 4 | 72% / 95% / 0.910 | 65% / 95% / 0.953 | 65% / 96% / 0.990 |
| Heteroskedastic Normal, average variance 4 | 63% / 92% / 0.890 | 60% / 93% / 0.925 | 65% / 94% / 0.980 |

The robust variance ratio generally moves towards one as the sample size increases. The ordinary posterior interval remains substantially under-covering in the variance-four scenarios. Since each cell has only 100 replications, coverage estimates have nontrivial Monte Carlo error; for example, at 95% coverage the binomial Monte Carlo standard error is approximately 2.2 percentage points.

## Prior sensitivity and the remaining bias problem

The balanced-loss correction adjusts posterior variance. It does not, by itself, debias the posterior mean when the prior is poorly located or too concentrated.

Under the correctly specified Normal SD 1 scenario at n = 40:

| Prior | Mean posterior-mean bias | Posterior 95% coverage | Robust Wald 95% coverage |
|---|---:|---:|---:|
| Centered diffuse | -0.055 | 98% | 96% |
| Shifted diffuse (location -1) | -0.188 | 74% | 75% |
| Shifted concentrated (location -0.25) | -0.183 | 74% | 76% |
| Spectrally modulated diffuse | -0.077 | 96% | 94% |

At n = 160, the shifted-prior robust coverages improved to 90% and 89%, respectively, but remained below nominal coverage. This illustrates why prior sensitivity must be evaluated separately from variance calibration. A robust variance multiplier is not a general correction for prior-induced location bias.

## Interpretation and limitations

1. In this scalar independent-observation design, the balanced-loss variance correction materially improves repeated-sampling coverage under substantial likelihood-variance misspecification.
2. The robust variance estimates approach both the oracle and empirical sandwich variance as sample size increases.
3. The ordinary Bayesian credible interval and the balanced-loss robust Wald interval answer different inferential questions. The robust Wald interval is not a posterior credible interval.
4. Prior location and concentration can induce finite-sample bias that the variance correction does not remove.
5. The simulation uses independent observations. It does not validate clustered, serially dependent or multivariate settings.
6. There are 100 replications per cell. These results are exploratory finite-simulation evidence, not a general performance guarantee.
7. The result is specific to this Sinc prior family, working likelihood, parameter interval and simulation design. Independent replication and broader model classes remain necessary.

## Next research steps

- Increase replications and report coverage confidence intervals for selected cells.
- Develop explicit prior-predictive and posterior-bias diagnostics.
- Extend posterior computation and balanced-loss correction to multivariate parameters.
- Derive suitable score aggregation for clustered and dependent observations.
- Replicate the NHANES age–systolic-blood-pressure application with documented data extraction and independent validation.
