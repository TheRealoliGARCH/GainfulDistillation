# Paired Prior-Location and Diffuse-Contamination Stress Grid

## Executive finding

Diffuse-contamination mixtures are **not a universal bias-control improvement** in this experiment. They reduce the influence of the nominal prior centre, but they also weaken regularisation. Across all tested settings, the mixtures were essentially neutral for point-estimation loss under the correctly specified Normal model, but increased mean absolute error and mean squared error under variance misspecification, heavy tails, and heteroskedasticity.

The balanced-loss variance correction continued to produce substantially better repeated-sampling coverage than the ordinary posterior credible interval in the misspecified scenarios. That is a distinct result: improving variance calibration does not imply that broadening the prior improves the posterior point estimate.

## Design

The study uses a paired common-random-number Monte Carlo design with:

- Four data-generating processes: Normal SD 1 (correctly specified); Normal SD 2 (variance misspecified); Student-t(5) rescaled to variance 4; and independent heteroskedastic Normal observations with average variance 4.
- Sample sizes 40, 80 and 160.
- True parameter 0.5.
- Nominal prior-location offsets -1, -0.5, +0.5 and +1 relative to the true parameter.
- Diffuse-contamination weights 5%, 10% and 20%.
- Diffuse-component scale multipliers 3, 5 and 10.
- A nominal prior at each of the four displaced locations and a correctly centred diffuse reference prior.

For each displaced location, the nine mixture combinations share the nominal component's location. The diffuse component is wider and unmodulated. It therefore changes prior concentration without correcting a wrongly located centre. There are 41 prior variants per data-generating process/sample-size combination and 492 result cells overall. Each cell contains 100 replications and 2,048 midpoint-quadrature points. The experiment uses 1,200 distinct simulated datasets; each dataset is evaluated under every prior variant in its DGP/sample-size block.

The working likelihood is independent Normal(theta, 1). The heteroskedastic case is independent but non-identically distributed. There is no serial or cluster dependence. The scalar balanced-loss implementation is used only within its independent-observation scope.

The posterior parameter interval is [-16, 16]. The minimum reported prior mass retained inside this interval is 0.999214 (99.9214%), so finite-bound truncation is small even for the widest tested mixture component. The fixed seed is 2026111402.

## Paired comparison method

Each mixture is compared with the nominal prior at the same prior-location offset using the **same simulated datasets**. For each matched replication the experiment calculates the change in:

- absolute estimation error, |posterior mean - true parameter|;
- squared estimation error;
- posterior credible-interval coverage; and
- balanced-loss robust-Wald coverage.

A negative error-loss change favours the mixture; a positive change favours the nominal prior. The paired Monte Carlo standard error is the sample standard deviation of the replication-level paired differences divided by the square root of the replication count. Design-average standard errors combine the cell-level variances and give equal weight to the listed design conditions.

The reported MAE and MSE changes are therefore paired estimates, not differences between two independently simulated arms.

## Results by data-generating process

The following table averages all nine mixture settings, all four prior-location offsets and all three sample sizes within each data-generating process. Positive paired changes mean that diffuse contamination increased estimation loss relative to the matching nominal prior.

| Data-generating process | Paired MAE change (MCSE) | Paired MSE change (MCSE) | Interpretation |
|---|---:|---:|---|
| Normal, SD 1 | -0.000003 (0.000070) | -0.000002 (0.000020) | Essentially neutral |
| Normal, SD 2 | +0.004230 (0.000170) | +0.002954 (0.000180) | Higher point-estimation loss |
| Student-t(5), variance 4 | +0.004228 (0.000170) | +0.003201 (0.000180) | Higher point-estimation loss |
| Heteroskedastic Normal, average variance 4 | +0.004702 (0.000200) | +0.003718 (0.000210) | Higher point-estimation loss |

The increase in loss under the three high-variance or misspecified scenarios is large relative to its paired Monte Carlo standard error. Under correctly specified Normal SD 1, the average change is indistinguishable from zero at this simulation resolution.

## Sensitivity to mixture weight and diffuse scale

The table below averages each fixed mixture setting over the 48 combinations of data-generating process, sample size and prior-location offset. The percentage changes are relative to the average nominal-prior MAE and MSE across those matched conditions.

| Contamination weight | Scale multiplier | Paired MAE change (MCSE) | MAE change | Paired MSE change (MCSE) | MSE change | Replications with lower absolute error |
|---:|---:|---:|---:|---:|---:|---:|
| 5% | 10x | +0.00168 (0.00016) | +1.10% | +0.00141 (0.00019) | +3.34% | 57.8% |
| 5% | 5x | +0.00232 (0.00019) | +1.51% | +0.00185 (0.00022) | +4.36% | 57.8% |
| 5% | 3x | +0.00252 (0.00019) | +1.64% | +0.00195 (0.00021) | +4.60% | 57.7% |
| 10% | 10x | +0.00258 (0.00020) | +1.69% | +0.00204 (0.00023) | +4.82% | 57.8% |
| 10% | 5x | +0.00340 (0.00024) | +2.22% | +0.00256 (0.00026) | +6.05% | 57.6% |
| 10% | 3x | +0.00362 (0.00025) | +2.36% | +0.00266 (0.00025) | +6.28% | 57.6% |
| 20% | 10x | +0.00385 (0.00026) | +2.51% | +0.00286 (0.00027) | +6.76% | 57.6% |
| 20% | 5x | +0.00476 (0.00031) | +3.11% | +0.00342 (0.00029) | +8.06% | 57.3% |
| 20% | 3x | +0.00487 (0.00032) | +3.18% | +0.00345 (0.00029) | +8.15% | 57.1% |

All nine fixed mixture combinations increased average paired MAE and MSE over the full design. Larger contamination weights generally increased the loss penalty. A larger diffuse-scale multiplier partially moderated that penalty at a given weight, but did not reverse it.

The fraction of replications in which the mixture produced a smaller absolute error was nevertheless about 57–58%. This is not contradictory: the mixture improved the absolute error in a modest majority of replications, but the worsened cases tended to contribute larger errors, raising the average MAE and MSE.

## Sample-size behaviour

Averaging across all data-generating processes, location offsets and mixture settings:

| Sample size | Paired MAE change (MCSE) | Paired MSE change (MCSE) |
|---:|---:|---:|
| 40 | +0.00686 (0.00022) | +0.00569 (0.00024) |
| 80 | +0.00228 (0.00008) | +0.00139 (0.00007) |
| 160 | +0.00072 (0.00003) | +0.00033 (0.00002) |

The penalty becomes smaller as sample size rises, consistent with the likelihood becoming more influential relative to the prior. It remains positive in this finite design at all three sample sizes.

## Coverage and robust variance

Averaging across the full prior-location and mixture grid, ordinary posterior 95% credible-interval coverage and balanced-loss robust-Wald 95% coverage were:

| Data-generating process | Posterior credible-interval coverage | Robust-Wald coverage | Robust/oracle variance ratio |
|---|---:|---:|---:|
| Normal, SD 1 | 95.3% | 94.9% | 0.937 |
| Normal, SD 2 | 72.5% | 96.3% | 0.947 |
| Student-t(5), variance 4 | 67.5% | 93.1% | 0.944 |
| Heteroskedastic Normal, average variance 4 | 64.7% | 93.2% | 0.957 |

Under the three misspecified/high-variance designs, the ordinary posterior credible interval substantially under-covered, while the robust-Wald interval was much closer to nominal coverage. The robust-Wald interval remains a repeated-sampling interval, not a Bayesian credible interval. The heteroskedastic and heavy-tailed coverage estimates remain below 95% in this run and should not be described as exact nominal calibration.

Descriptively, the correctly centred diffuse reference had average MAE 0.15186, compared with 0.15319 for the nominal displaced priors and 0.15648 averaged over mixture variants. The centred reference is an oracle-location benchmark in this simulation, not an operational prior-selection rule.

## Interpretation for GDS

1. **Prior diffusion is not automatically beneficial.** In the correctly specified unit-variance Normal design it was essentially neutral on average. Under Normal variance misspecification, Student-t tails and heteroskedasticity it increased paired point-estimation loss.
2. **Prior concentration and likelihood uncertainty must be considered together.** In the high-variance designs, the nominal prior provided shrinkage that helped control estimation error. Weakening that shrinkage increased estimator variability more than it reduced prior-induced location bias.
3. **The variance correction and prior robustification solve different problems.** The balanced-loss variance correction improved repeated-sampling coverage under likelihood misspecification. The diffuse prior mixture did not improve point-estimation loss in those same scenarios.
4. **The mixture does not correct location misspecification.** All diffuse components retain their nominal centre. A wider wrongly centred prior remains wrongly centred.
5. **Do not select the weight from this same grid and call it optimal.** The nine configurations are a sensitivity grid. A prospective choice would require an independent validation design and an explicit loss function trading bias, variance, coverage and predictive adequacy.

The GDS recommendation is therefore to retain diffuse contamination as an auditable sensitivity option, not a default automatic adjustment. A prior-predictive conflict warning should trigger review of assumptions and evidence; it should not mechanically broaden or recenter the prior.

## Limitations and next validation

This remains a scalar Monte Carlo study with one true parameter value, one working likelihood, three sample sizes and 100 replications per cell. The heavy-tailed and heteroskedastic cases are independent observations; serial, clustered, multivariate and endogenous designs are not covered. The results do not establish a universal ranking of prior families or mixture weights.

The next validation should test the diagnostic decision itself prospectively: specify prior-location and dispersion policies before generating held-out datasets, then compare their point loss, predictive calibration and interval coverage under independent replication. The policy should be allowed to retain the nominal prior when diffusion is not supported by the evidence.
