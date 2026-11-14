(* Symbolic Wolfram Language cross-check for the six densities.
   Run from repository root:
   wolframscript -file experiments/six_density_mathematica_crosscheck.wl

   All integrations below use Integrate[]. N[] is used only to display
   decimal values after symbolic integration; NIntegrate[] is not used.

   Mahaloy convention: sinc(u)=Sin[u]/u, with value 1 at u=0.
   This convention reproduces the printed Fresnel normaliser and graph zeros. *)

ClearAll["Global`*"];
$Assumptions = Element[x, Reals];

printedNormalisers = <|
  "Ghosh" -> 2 (1 + FresnelC[Sqrt[6]]/Sqrt[6]),
  "Mahaloy" -> 1/4 (-Sin[8] - 8 Sqrt[Pi] FresnelC[2] +
      8 Sqrt[Pi] FresnelC[Sqrt[2]] + 4 Sqrt[Pi] FresnelC[4/Sqrt[Pi]]),
  "Madras" -> Sqrt[2] FresnelS[Sqrt[2]],
  "Temple bell" -> 2 Gamma[5/4],
  "Rocket cone" -> Sqrt[Pi] Erf[1],
  "Cup" -> Sqrt[Pi] Erfi[1]
|>;

(* The five kernels other than Mahaloy are nonnegative on their support.
   For Mahaloy, split at the two positive zeros of Sin[8 x^2]. *)
kernels = <|
  "Ghosh" -> (1 + Cos[3 Pi x^2]),
  "Madras" -> Sin[Pi x^2],
  "Temple bell" -> Exp[-x^4],
  "Rocket cone" -> Exp[-x^2],
  "Cup" -> Exp[x^2]
|>;

mahaloyKernel = Sin[8 x^2]/(8 x^2);
mahaloyA = Sqrt[Pi/8];
mahaloyB = Sqrt[Pi/4];

mahaloyPiecewiseIntegral[p_Integer?NonNegative] :=
  2 (
    Integrate[x^p mahaloyKernel, {x, 0, mahaloyA},
      Assumptions -> 0 <= x <= mahaloyA] -
    Integrate[x^p mahaloyKernel, {x, mahaloyA, mahaloyB},
      Assumptions -> mahaloyA <= x <= mahaloyB] +
    Integrate[x^p mahaloyKernel, {x, mahaloyB, 1},
      Assumptions -> mahaloyB <= x <= 1]
  );

kernelMoment[name_String, p_Integer?NonNegative] :=
  Which[
    name === "Mahaloy",
      mahaloyPiecewiseIntegral[p],
    name === "Temple bell",
      Integrate[x^p Exp[-x^4], {x, -Infinity, Infinity},
        Assumptions -> Element[x, Reals]],
    True,
      Integrate[x^p kernels[name], {x, -1, 1},
        Assumptions -> -1 <= x <= 1]
  ];

(* Independent symbolic integrations of each unnormalised kernel. *)
symbolicKernelIntegrals = Association@Table[
  name -> FullSimplify[kernelMoment[name, 0]],
  {name, Keys[printedNormalisers]}
];

(* Symbolic equality between the direct integral and each printed constant. *)
normaliserIdentities = Association@Table[
  name -> FullSimplify[
    symbolicKernelIntegrals[name] - printedNormalisers[name] == 0
  ],
  {name, Keys[printedNormalisers]}
];

(* Evenness gives mean zero. These are exact Integrate[] moment calculations. *)
symbolicMoments = Association@Table[
  name -> Module[{d = printedNormalisers[name], m2, m4},
    m2 = FullSimplify[kernelMoment[name, 2]/d];
    m4 = FullSimplify[kernelMoment[name, 4]/d];
    <|
      "Mean" -> 0,
      "Variance" -> m2,
      "FourthMoment" -> m4,
      "Kurtosis" -> FullSimplify[m4/m2^2]
    |>
  ],
  {name, Keys[printedNormalisers]}
];

peakNumerators = <|
  "Ghosh" -> 2,
  "Mahaloy" -> 1,
  "Madras" -> 1,
  "Temple bell" -> 1,
  "Rocket cone" -> 1,
  "Cup" -> E
|>;

(* Decimal evaluation is applied only after exact integration/simplification. *)
peakValues = Association@Table[
  name -> N[peakNumerators[name]/printedNormalisers[name], 30],
  {name, Keys[printedNormalisers]}
];

densityIntegralIdentities = Association@Table[
  name -> FullSimplify[
    symbolicKernelIntegrals[name]/printedNormalisers[name] == 1
  ],
  {name, Keys[printedNormalisers]}
];

(* A fully analytic strict-bound argument for Cup:
   exp(x^2) >= 1 + x^2 + x^4/2 on [-1,1], hence its
   normaliser is at least 43/15, which is greater than e. *)
cupLowerBound = Integrate[1 + x^2 + x^4/2, {x, -1, 1}];
cupStrictBound = FullSimplify[cupLowerBound > E];

(* Compare symbolic moments with the independent SciPy/QUADPACK values. *)
scipyReference = <|
  "Ghosh" -> {0.2649172903539256, 2.2410459688073603},
  "Mahaloy" -> {0.13479511114749426, 2.9146952314681136},
  "Madras" -> {0.4331468153626013, 1.2752535879222653},
  "Temple bell" -> {0.3379891200336424, 2.1884396152264762},
  "Rocket cone" -> {0.2537041018036845, 2.085895218155902},
  "Cup" -> {0.4292307058277509, 1.548991521369453}
|>;

momentComparisons = Association@Table[
  name -> Module[{ref = scipyReference[name], vals},
    vals = N[
      {symbolicMoments[name]["Variance"],
       symbolicMoments[name]["Kurtosis"]}, 25
    ];
    <|
      "VarianceDifferenceFromSciPy" -> N[vals[[1]] - ref[[1]], 15],
      "KurtosisDifferenceFromSciPy" -> N[vals[[2]] - ref[[2]], 15],
      "BothWithin1e-9" ->
        (Abs[vals[[1]] - ref[[1]]] < 10^-9 &&
         Abs[vals[[2]] - ref[[2]]] < 10^-9)
    |>
  ],
  {name, Keys[printedNormalisers]}
];

report = <|
  "Engine" -> "Wolfram Language symbolic Integrate",
  "WolframVersion" -> $Version,
  "IntegrationMethod" -> "Integrate only; no NIntegrate",
  "MahaloySincConvention" -> "Sin[u]/u, with value 1 at u=0",
  "MahaloySignSplits" -> N[{mahaloyA, mahaloyB}, 20],
  "SymbolicKernelIntegrals" -> symbolicKernelIntegrals,
  "PrintedNormalisers" -> printedNormalisers,
  "NormaliserIdentities" -> normaliserIdentities,
  "DensityIntegralIdentities" -> densityIntegralIdentities,
  "AllNormaliserIdentitiesProved" ->
    And @@ (TrueQ /@ Values[normaliserIdentities]),
  "AllDensityIntegralsProved" ->
    And @@ (TrueQ /@ Values[densityIntegralIdentities]),
  "SymbolicMoments" -> symbolicMoments,
  "PeakValuesDecimal" -> peakValues,
  "CupLowerBound" -> cupLowerBound,
  "CupLowerBoundExceedsE" -> cupStrictBound,
  "MomentComparisonsWithSciPy" -> momentComparisons,
  "AllMomentComparisonsWithin1e-9" ->
    And @@ (TrueQ[#["BothWithin1e-9"]] & /@ Values[momentComparisons])
|>;

Print["Normaliser identities:"];
Print[Dataset[normaliserIdentities]];
Print["Density-integral identities:"];
Print[Dataset[densityIntegralIdentities]];
Print["Symbolic moments:"];
Print[Dataset[symbolicMoments]];
Print["Peak values:"];
Print[Dataset[peakValues]];
Print["Summary flags:"];
Print[InputForm[KeyDrop[report, {
  "SymbolicKernelIntegrals", "PrintedNormalisers", "SymbolicMoments"
}]]];

root = DirectoryName[DirectoryName[ExpandFileName[$InputFileName]]];
resultsDirectory = FileNameJoin[{root, "results"}];
If[!DirectoryQ[resultsDirectory], CreateDirectory[resultsDirectory]];
Export[
  FileNameJoin[{resultsDirectory, "six_density_mathematica_crosscheck.json"}],
  report,
  "RawJSON"
];
Print["Exported symbolic results to results/six_density_mathematica_crosscheck.json"];
