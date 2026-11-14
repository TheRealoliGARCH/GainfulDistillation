# Mathematica cross-check status

A standalone Wolfram Language cross-check harness is provided at
experiments/six_density_mathematica_crosscheck.wl.

It independently evaluates all six printed normalising constants, integrates
all six unnormalised kernels and normalised densities with NIntegrate,
calculates the first, second and fourth central moments, computes peak
heights, and compares the results against the archived SciPy/QUADPACK values.
The Mahaloy absolute-sinc integral is split at its interior zeros.

**Execution status:** the current GDS automation environment does not contain
a Mathematica / Wolfram Engine runtime. Therefore the Wolfram Language
program has been prepared but has not been executed here. No Mathematica
output is fabricated or represented as a completed cross-validation.

Run from the repository root in Mathematica or Wolfram Engine:

    wolframscript -file experiments/six_density_mathematica_crosscheck.wl

On successful execution, the script writes
results/six_density_mathematica_crosscheck.json. The JSON records the Wolfram
version, precision settings, numerical results, and tolerance-based
comparisons with SciPy. The cross-validation should be marked complete only
after this output has been generated and inspected.
