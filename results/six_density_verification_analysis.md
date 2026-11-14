# GDS First Research Milestone: Six-Density Verification

**Status:** High-accuracy numerical verification completed. Rigorous analytic or interval-arithmetic bounds for the strict peak inequalities remain a follow-up task.

## 1. Source conventions

This audit examines the six piecewise densities in *Six Probability Density Functions*, pages 1, 3, 5, 7, 9 and 11; the associated plots appear on pages 2, 4, 6, 8, 10 and 12.

The Mahaloy definition says only that `sinc` is the sinc function. Its printed normaliser and plotted zero locations are consistent with the **unnormalised** convention `sinc(u) = sin(u)/u`, with `sinc(0)=1`. This audit uses that convention explicitly. The paper should specify it because the normalised convention `sin(pi*u)/(pi*u)` gives a different function.

The Fresnel functions are interpreted as defined in the paper:
- `C(z) = integral_0^z cos(pi*t^2/2) dt`;
- `S(z) = integral_0^z sin(pi*t^2/2) dt`.

The imaginary error function is `erfi(z) = erf(i*z)/i`.

## 2. Normalising constants

Writing each density as its kernel divided by `D`, the constants printed in the paper are:

- **Ghosh:** `D_G = 2*(1 + C(sqrt(6))/sqrt(6))`.
- **Mahaloy:**
  `D_M = 1/4 * [-sin(8) - 8*sqrt(pi)*C(2) + 8*sqrt(pi)*C(sqrt(2)) + 4*sqrt(pi)*C(4/sqrt(pi))]`.
- **Madras:** `D_D = sqrt(2)*S(sqrt(2))`.
- **Temple bell:** `D_T = 2*Gamma(5/4) = Gamma(1/4)/2`.
- **Rocket cone:** `D_R = sqrt(pi)*erf(1)`.
- **Cup:** `D_C = sqrt(pi)*erfi(1)`.

The Ghosh, Madras, Rocket cone and Cup identities follow by substitution in their defining integrals. The Temple bell identity follows from the Gamma integral and `Gamma(5/4)=Gamma(1/4)/4`. For Mahaloy, the printed Fresnel expression was checked against direct integration of the absolute sinc kernel, splitting at the two positive-side interior zeros.

## 3. Numerical results

All six functions are even, so their means are zero. The following values were computed using adaptive Simpson quadrature; the reported integral is an independent numerical integration of each implemented density.

| Density | Normaliser `D` | Numeric integral | Peak density | Variance | Kurtosis |
|---|---:|---:|---:|---:|---:|
| Ghosh | 2.413671104813 | 1.000000000000 | 0.828613308587 | 0.264917290354 | 2.241045968808 |
| Mahaloy | 1.029851026428 | 1.000000000000 | 0.971014228600 | 0.134795111147 | 2.914695231469 |
| Madras | 1.009709188227 | 1.000000000000 | 0.990384173641 | 0.433146815363 | 1.275253587922 |
| Temple bell | 1.812804954111 | 1.000000000000 | 0.551631325660 | 0.337989120034 | 2.188439615230 |
| Rocket cone | 1.493648265625 | 1.000000000000 | 0.669501664491 | 0.253704101804 | 2.085895218156 |
| Cup | 2.925303491814 | 1.000000000000 | 0.929230705828 | 0.429230705828 | 1.548991521369 |

The peak values are the kernel maxima divided by their normalisers: respectively `2/D_G`, `1/D_M`, `1/D_D`, `1/D_T`, `1/D_R`, and `e/D_C`.

## 4. Main finding: all six peaks are below one

The five strict inequalities stated in the paper are reproduced numerically. The Cup density, for which the paper gives only `0 <= f(x) <= 1`, also has a strict maximum:

`sup_x f_Cup(x) = e/(sqrt(pi)*erfi(1)) = 0.929230705828 < 1`.

Its normaliser exceeds `e` by approximately `0.207021663355`. Thus the Cup statement can be strengthened to a strict upper bound, subject to a formal analytic or interval-arithmetic enclosure of this constant comparison.

Madras is closest to the ceiling, with peak approximately `0.990384173641`.

**Numerical-certificate qualification:** the implementation uses adaptive Simpson quadrature with target tolerance `2e-11`. It is not interval arithmetic and is not yet a machine-checked proof of the strict constant inequalities. S2's next task is to derive transparent analytic bounds or rigorous interval enclosures, especially for the Mahaloy and Madras normalisers and for the Cup comparison with `e`.

## 5. Support and moments

The five distributions other than Temple bell have compact support `[-1,1]), so all their polynomial moments exist. Temple bell has real-line support, but its `exp(-x^4)` tails also give finite moments of every polynomial order. All six are symmetric, have zero mean, and have positive variance.

## 6. Reproducibility

- `experiments/six_density_verification.py` uses only the Python standard library.
- `tests/test_six_density_verification.py` checks normalisation, strict peak bounds, the Mahaloy sinc convention, the strengthened Cup bound, and reference moments.
- The Mahaloy absolute-value integral is split at `sqrt(pi/8)` and `sqrt(pi/4)` on the positive half-axis.
- The Temple bell integral is truncated at `x=12` on the positive half-axis; the omitted `exp(-x^4)` tail is negligible at the reported precision.

## 7. Next assignments

- **S1:** study Bayesian-prior interpretations and distinguish prior shape from likelihood information.
- **S2:** derive analytic proofs or rigorous interval bounds for all peak inequalities and normalising constants.
- **S3:** maintain the independent implementation and add cross-implementation tests.
- **S4:** compare moments, entropy, tail behaviour and sampling efficiency under a common protocol.
- **S5:** challenge the sinc convention, endpoint definitions, absolute-value handling, truncation and inequalities independently.

This is the first reproducible checkpoint, not the end of the investigation.
