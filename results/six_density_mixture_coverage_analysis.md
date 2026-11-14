# Geometric Coverage of All Convex Mixtures of the Six PDFs

## Definition

A weighted combination is interpreted as a probability mixture:
`f_w(x) = sum_i w_i f_i(x)`, with `w_i >= 0` and `sum_i w_i = 1`.

For fixed `x`, the attainable values are exactly the convex hull of the six
scalar values `f_i(x)`, namely the interval between their pointwise lower
and upper envelopes:

`m(x) = min_i f_i(x)`, `M(x) = max_i f_i(x)`.

The covered set in the requested rectangle is
`{(x,y): -1 <= x <= 1, m(x) <= y <= M(x)}`. It is the union of the
graphs of all possible convex-mixture PDFs, not the area under one PDF.

## Result

The rectangle `[-1,1] x [0,1]` has area 2. Each source density is even, so
the covered area is twice the envelope-span integral on `[0,1]`.

| Quantity | Value |
|---|---:|
| Positive-half envelope-span integral | 0.732565995308064 |
| Covered area | 1.465131990616127 |
| Rectangle area | 2.000000000000000 |
| **Coverage** | **73.256599530806%** |
| Uncovered area | 0.534868009383873 |
| Uncovered share | 26.743400469194% |

## Upper and lower envelopes

The envelope on `x in [0,1]` changes at seven crossings:

| Interval | Upper envelope | Lower envelope |
|---|---|---|
| [0.000000000, 0.358287138] | Mahaloy | Madras |
| [0.358287138, 0.411212075] | Mahaloy | Cup |
| [0.411212075, 0.451996671] | Mahaloy | Ghosh |
| [0.451996671, 0.615103967] | Madras | Ghosh |
| [0.615103967, 0.830995341] | Madras | Mahaloy |
| [0.830995341, 0.863248702] | Ghosh | Mahaloy |
| [0.863248702, 0.958508940] | Cup | Mahaloy |
| [0.958508940, 1.000000000] | Cup | Ghosh |

The two Mahaloy zeros, at `sqrt(pi/8)` and `sqrt(pi/4)`, were used as
integration breakpoints because the absolute-sinc kernel has cusps there.

## Numerical method

- The source densities and normalisers are taken from
  `experiments/six_density_verification.py`.
- Pairwise curve crossings are bracketed on 16,384 equal subintervals of
  `[0,1]` and refined by SciPy `brentq`.
- Between crossings, midpoint classification identifies the upper and lower
  envelopes. Adjacent intervals with unchanged envelope identities are merged.
- The envelope width is integrated using SciPy `scipy.integrate.quad`
  (QUADPACK), with explicit splits at the Mahaloy absolute-value cusps.
- The calculation was repeated at 8,192 scan intervals; all eight envelope
  segments and their boundaries were stable to the regression tolerances.
- No NumPy API is imported or called.

The QUADPACK error estimate for the positive-half integral is approximately
`8.1e-15`. This is a numerical diagnostic, not a formal interval-arithmetic
bound. Pairwise tangencies that do not change the ordering cannot change
the envelope area.

## Interpretation

The result is conditional on the convex-mixture definition. Signed linear
combinations or unnormalised nonnegative weights define different sets and
would require a different coverage calculation. Under the standard
probability-mixture interpretation, coverage is **73.2566%**.
