# Diffuse-Contamination Prior Bias-Control Study

## Question

Can an explicit diffuse-contamination component reduce finite-sample posterior-mean bias caused by a badly located but proper Sinc prior, without using the observed data to recenter the prior?

The study compares a shifted nominal prior with mixtures that retain the shifted component and add a broader, same-centre Sinc component. A correctly centred diffuse prior is included as a reference benchmark.

## Design

- Data: independent observations from Normal(theta, 1), with true theta = 0.5.
- Working likelihood: correctly specified independent Normal(theta, 1).
- Sample sizes: 40, 80 and 160.
- Replications: 200 per prior/sample-size cell.
- Posterior approximation: 1,024 midpoint-quadrature cells on [-8, 8].
- Seed: 20261114.
- Prior comparisons: common simulated datasets within each sample-size cell.
- Prior variants: shifted nominal; 5%, 10% and 20% diffuse-contamination mixtures; correctly centred diffuse reference.

The nominal prior is centred at -1 with scale 2. Each mixture retains that component and adds an unmodulated Sinc prior centred at -1 with scale 10. The mixture is therefore a concentration-robustification device; it does not correct the prior's location. The centred reference prior is centred at zero with scale 2.

The reported posterior coverage is for the ordinary 95% Bayesian credible interval. The robust-Wald coverage is for the posterior mean plus or minus 1.96 times the balanced-loss robust standard error. These are different inferential objects.

## Results

Each table reports repeated-sample posterior-mean bias (Monte Carlo standard error), RMSE, ordinary posterior 95% credible-interval coverage, and balanced-loss robust-Wald 95% coverage.

### Sample size 40

| Prior | Bias (MCSE) | RMSE | Posterior coverage | Robust-Wald coverage |
|---|---:|---:|---:|---:|
| Shifted nominal | -0.1862 (0.0085) | 0.2213 | 74.5% | 76.0% |
| 5% diffuse mixture | -0.1112 (0.0121) | 0.2038 | 85.5% | 85.5% |
| 10% diffuse mixture | -0.0860 (0.0124) | 0.1950 | 91.0% | 87.5% |
| 20% diffuse mixture | -0.0613 (0.0123) | 0.1837 | 94.0% | 92.5% |
| Centred diffuse reference | -0.0524 (0.0100) | 0.1511 | 95.5% | 96.0% |

### Sample size 80

| Prior | Bias (MCSE) | RMSE | Posterior coverage | Robust-Wald coverage |
|---|---:|---:|---:|---:|
| Shifted nominal | -0.0908 (0.0067) | 0.1311 | 89.0% | 88.5% |
| 5% diffuse mixture | -0.0407 (0.0087) | 0.1292 | 91.5% | 91.0% |
| 10% diffuse mixture | -0.0258 (0.0087) | 0.1257 | 92.0% | 91.0% |
| 20% diffuse mixture | -0.0124 (0.0086) | 0.1213 | 93.5% | 91.5% |
| Centred diffuse reference | -0.0121 (0.0076) | 0.1083 | 94.5% | 94.0% |

### Sample size 160

| Prior | Bias (MCSE) | RMSE | Posterior coverage | Robust-Wald coverage |
|---|---:|---:|---:|---:|
| Shifted nominal | -0.0474 (0.0051) | 0.0866 | 92.0% | 91.0% |
| 5% diffuse mixture | -0.0203 (0.0060) | 0.0867 | 92.5% | 92.5% |
| 10% diffuse mixture | -0.0120 (0.0060) | 0.0852 | 93.0% | 94.0% |
| 20% diffuse mixture | -0.0049 (0.0059) | 0.0834 | 94.5% | 94.5% |
| Centred diffuse reference | -0.0052 (0.0055) | 0.0784 | 94.5% | 95.5% |

## Interpretation

1. **The diffuse component attenuates prior-induced bias in this design.** Relative to the shifted nominal prior, the 20% mixture reduces absolute bias by approximately 67% at n = 40, 86% at n = 80, and 90% at n = 160.
2. **Coverage improves as the diffuse weight increases.** At n = 40, ordinary posterior coverage rises from 74.5% under the shifted nominal prior to 94.0% under the 20% mixture. The robust-Wald coverage rises from 76.0% to 92.5%.
3. **The mixture does not outperform a correctly located prior in this experiment.** The centred diffuse reference has lower RMSE and equal or better coverage in each sample-size comparison. The correct response to a location error is not merely to broaden the same incorrectly centred prior.
4. **Variance correction and mean-bias control remain distinct.** The balanced-loss correction changes the variance estimate; it does not, by itself, move the posterior mean to remove prior-induced location bias.
5. **The 20% setting is only the strongest of the three tested weights.** It is not an estimated universal optimum. Weight and diffuse scale should be specified prospectively and tested across other likelihoods and data-generating processes.

The prior-conflict warning rate also declines as the diffuse component is added. That rate is a scale-relative diagnostic: its standardised prior-reference gap divides by the square root of prior variance plus reference variance. Increasing prior spread can therefore reduce the warning rate without changing the prior centre. It must not be read as evidence that the location conflict has been repaired. The empirical bias and RMSE results are the more direct evidence in this Monte Carlo design.

## Limitations and next tests

This is a one-parameter, independent-observation experiment with a correctly specified Gaussian likelihood. It does not establish performance under heavy tails, heteroskedasticity, serial or clustered dependence, model misspecification, or multivariate inference. Each cell has 200 replications, so coverage estimates retain Monte Carlo uncertainty. The finite bounds and quadrature grid are explicit parts of the numerical experiment.

The next scalar checks should vary the nominal prior's location, diffuse scale multiplier, and contamination weight across the heavy-tailed and heteroskedastic scenarios already used in GDS. A useful decision rule should combine prior-predictive compatibility, retained prior mass, bias, RMSE, and coverage rather than selecting a mixture weight from one simulation alone. Only after those checks should GDS move to multivariate balanced-loss inference.
