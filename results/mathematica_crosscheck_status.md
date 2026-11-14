# Mathematica symbolic cross-check status

The Mathematica cross-check harness uses Wolfram Language `Integrate[]`
throughout. It does not use `NIntegrate[]`.

The script derives the six kernel integrals symbolically, checks them against
the printed Fresnel/Gamma/error-function constants, derives the second and
fourth moments symbolically, and then uses decimal evaluation only to compare
the exact expressions with the previously computed SciPy/QUADPACK values.
The Mahaloy absolute-sinc kernel is handled by splitting at its two interior
zeros before invoking `Integrate[]`.

**Execution status:** no Mathematica or Wolfram Engine runtime is available in
the current environment. The script is committed but has not been executed
here. Consequently, no Mathematica output is claimed. The symbolic cross-check
becomes complete only after a Wolfram Engine / Mathematica execution produces
and validates the JSON output.

Run from the repository root:

    wolframscript -file experiments/six_density_mathematica_crosscheck.wl

The Cup strict upper bound is also derived analytically in the script:
`Integral[-1,1] (1+x^2+x^4/2) dx = 43/15 > e`.
