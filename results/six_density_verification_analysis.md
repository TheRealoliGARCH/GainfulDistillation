# GDS First Research Milestone: Six-Density Verification (SciPy)

**Verification engine:** SciPy `scipy.integrate.quad` (QUADPACK), with special-function values from `scipy.special`. No NumPy API is called directly. This SciPy audit supersedes the earlier hand-written adaptive-Simpson prototype.

## 1. Formula transcription and conventions

The six piecewise expressions were transcribed from the rendered formula pages of *Six Probability Density Functions*: Ghosh (p. 1), Mahaloy (p. 3), Madras (p. 5), Temple bell (p. 7), Rocket cone (p. 9), and Cup (p. 11). The first five have support `[-1, 1]`; Temple bell has support on the real line.

The Mahaloy formula writes `sinc(8*x^2)` without specifying the convention. This audit uses the unnormalised convention `sinc(u) = sin(u)/u`, with `sinc(0) = 1`. This is consistent with both the printed Fresnel-C normalising constant and the zero locations visible in the graph on p. 4. The integral of the absolute kernel is split at its two positive interior zeros, `sqrt(pi/8)` and `sqrt(pi/4)`. The normalised convention `sin(pi*u)/(pi*u)` would define a different density and would not reproduce the printed constant/plot.

The Fresnel functions follow the paper's definitions:
- `C(z) = integral_0^z cos(pi*t^2/2) dt`;
- `S(z) = integral_0^z sin(pi*t^2/2) dt`.

## 2. Normalising constants

Let `D` denote the denominator in each density. The printed formulae give:

- Ghosh: `D_G = 2*(1 + C(sqrt(6))/sqrt(6))`;
- Mahaloy: `D_M = 1/4 * [-sin(8) - 8*sqrt(pi)*C(2) + 8*sqrt(pi)*C(sqrt(2)) + 4*sqrt(pi)*C(4/sqrt(pi))]`;
- Madras: `D_D = sqrt(2)*S(sqrt(2))`;
- Temple bell: `D_T = 2*Gamma(5/4)`;
- Rocket cone: `D_R = sqrt(pi)*erf(1)`;
- Cup: `D_C = sqrt(pi)*erfi(1)`.

SciPy's Fresnel outputs and special functions were used to evaluate these constants. Independently, QUADPACK integrated each unnormalised kernel over its full support (using symmetry and the Mahaloy zero splits). The numerical kernel integral and its formula-based denominator agree to the reported floating-point precision for all six cases.

## 3. Numerical results

Integrals and moments were evaluated with `epsabs = epsrel = 1e-12` and `limit = 500`. For Temple bell, QUADPACK integrated over `[0, infinity)` rather than using a finite truncation. Since all six functions are even, their means are zero.

| Density | Normaliser `D` | Integral of normalised density | Peak density | Variance | Kurtosis |
|---|---:|---:|---:|---:|---:|
| Ghosh | 2.413671104813002 | 0.9999999999999998 | 0.828613308587024 | 0.264917290353926 | 2.24104596880736 |
| Mahaloy | 1.029851026428492 | 1.0000000000000000 | 0.971014228599630 | 0.134795111147494 | 2.91469523146811 |
| Madras | 1.009709188227373 | 1.0000000000000002 | 0.990384173640711 | 0.433146815362601 | 1.27525358792227 |
| Temple bell | 1.812804954110954 | 1.0000000000000002 | 0.551631325660419 | 0.337989120033642 | 2.18843961522648 |
| Rocket cone | 1.493648265624854 | 1.0000000000000002 | 0.669501664491044 | 0.253704101803684 | 2.08589521815590 |
| Cup | 2.925303491814363 | 1.0000000000000000 | 0.929230705827751 | 0.429230705827751 | 1.54899152136945 |

The absolute QUADPACK error estimates for the raw-kernel integrals are at most about `1.7e-12` in this run. They are numerical error estimates, not formal interval certificates.

## 4. Peak calculations and the Cup inequality

The kernel maxima on their supports are respectively `2, 1, 1, 1, 1, e`. Thus the peak densities are those maxima divided by their normalisers. All six computed maxima are strictly below one. Madras is closest to one, at approximately `0.990384173640711`.

The Cup case admits a simple analytic strengthening of the paper's weak inequality. Its denominator is `D_C = integral_{-1}^{1} exp(x^2) dx`. For `|x| <= 1`, `exp(x^2) >= 1 + x^2 + x^4/2`, so `D_C >= 2 + 2/3 + 1/5 = 43/15 > e`. Consequently `sup f_Cup = e/D_C < 1`; SciPy evaluates this supremum as approximately `0.929230705827751`. Therefore the Cup density also satisfies a strict upper bound, not merely the weak bound printed in the paper.

For the other five, this audit is a high-accuracy numerical verification of the printed normalisers and peak inequalities. It is not claiming that floating-point quadrature error estimates alone constitute formal mathematical proofs of those inequalities. S2's follow-up is to provide analytic bounds or interval-arithmetic certificates for their normaliser-versus-kernel-maximum comparisons.

## 5. Reproducibility

- Implementation: `experiments/six_density_verification.py`.
- Tests: `tests/test_six_density_verification.py`.
- Archived SciPy output: `results/six_density_verification.json`.
- Runtime dependency for this research check: SciPy. The core GDS package remains free of mandatory runtime dependencies; SciPy is included in the optional development/test dependencies.

The test suite checks all six normalised integrals, peak bounds, the Mahaloy convention and equality between its printed normaliser and direct kernel integration, the strengthened Cup bound, and reference moments.

## 6. Next independent checks

- S1: examine the six densities as possible Bayesian priors, separating prior shape from likelihood information.
- S2: derive analytic or rigorous interval bounds for the first five strict peak inequalities.
- S3: maintain the independent SciPy implementation and cross-check endpoints and support handling.
- S4: compare moments, entropy, tail behaviour, and sampling methods under one protocol.
- S5: adversarially test transcription, sinc conventions, endpoint definitions, absolute-value handling, and alternative interpretations.

This is a reproducible numerical checkpoint, not the end of the investigation.
