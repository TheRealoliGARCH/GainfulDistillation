(* Wolfram Language cross-check for the six densities in the supplied paper.
   Run from repository root:
   wolframscript -file experiments/six_density_mathematica_crosscheck.wl

   Mahaloy convention: sinc(u)=Sin[u]/u, with value 1 at u=0.
   This convention matches the printed Fresnel normaliser and plot zeros. *)

ClearAll["Global`*"];
workingPrecision = 60;
accuracyGoal = 35;
precisionGoal = 35;
quadOptions = {
  WorkingPrecision -> workingPrecision, AccuracyGoal -> accuracyGoal,
  PrecisionGoal -> precisionGoal, MaxRecursion -> 35,
  Method -> {"GlobalAdaptive", "SymbolicProcessing" -> 0}
};

sincUnnormalised[u_?NumericQ] := If[TrueQ[u == 0], 1, Sin[u]/u];

normalisers = <|
  "Ghosh" -> 2 (1 + FresnelC[Sqrt[6]]/Sqrt[6]),
  "Mahaloy" -> 1/4 (-Sin[8] - 8 Sqrt[Pi] FresnelC[2] +
      8 Sqrt[Pi] FresnelC[Sqrt[2]] + 4 Sqrt[Pi] FresnelC[4/Sqrt[Pi]]),
  "Madras" -> Sqrt[2] FresnelS[Sqrt[2]],
  "Temple bell" -> 2 Gamma[5/4],
  "Rocket cone" -> Sqrt[Pi] Erf[1],
  "Cup" -> Sqrt[Pi] Erfi[1]
|>;

kernels = <|
  "Ghosh" -> Function[{x}, Cos[3 Pi x^2] + Cos[2 Pi]],
  "Mahaloy" -> Function[{x}, Abs[sincUnnormalised[8 x^2]]],
  "Madras" -> Function[{x}, Abs[Sin[Pi x^2]]],
  "Temple bell" -> Function[{x}, Exp[-x^4]],
  "Rocket cone" -> Function[{x}, Exp[-x^2]],
  "Cup" -> Function[{x}, Exp[x^2]]
|>;

mahaloyCuts = {-1, -Sqrt[Pi/4], -Sqrt[Pi/8], 0,
  Sqrt[Pi/8], Sqrt[Pi/4], 1};

integrateOnDomain[name_String, integrand_] := Which[
  name === "Temple bell",
  NIntegrate[integrand[x], {x, -Infinity, Infinity}, Sequence @@ quadOptions],
  name === "Mahaloy",
  Total@Table[
    NIntegrate[integrand[x], {x, mahaloyCuts[[j]], mahaloyCuts[[j + 1]]},
      Sequence @@ quadOptions],
    {j, 1, Length[mahaloyCuts] - 1}
  ],
  True,
  NIntegrate[integrand[x], {x, -1, 1}, Sequence @@ quadOptions]
];

scipyReference = <|
  "Ghosh" -> <|"Normaliser" -> 2.413671104813002,
    "Variance" -> 0.2649172903539256, "Kurtosis" -> 2.2410459688073603|>,
  "Mahaloy" -> <|"Normaliser" -> 1.0298510264284924,
    "Variance" -> 0.13479511114749426, "Kurtosis" -> 2.9146952314681136|>,
  "Madras" -> <|"Normaliser" -> 1.009709188227373,
    "Variance" -> 0.4331468153626013, "Kurtosis" -> 1.2752535879222653|>,
  "Temple bell" -> <|"Normaliser" -> 1.812804954110954,
    "Variance" -> 0.3379891200336424, "Kurtosis" -> 2.1884396152264762|>,
  "Rocket cone" -> <|"Normaliser" -> 1.4936482656248538,
    "Variance" -> 0.2537041018036845, "Kurtosis" -> 2.085895218155902|>,
  "Cup" -> <|"Normaliser" -> 2.925303491814363,
    "Variance" -> 0.4292307058277509, "Kurtosis" -> 1.548991521369453|>
|>;

peakNumerators = <|
  "Ghosh" -> 2, "Mahaloy" -> 1, "Madras" -> 1,
  "Temple bell" -> 1, "Rocket cone" -> 1, "Cup" -> E
|>;

names = Keys[normalisers];
mathematicaResults = Association@Table[
  name -> Module[
    {d = normalisers[name], k = kernels[name], rawIntegral, densityIntegral,
     firstMoment, secondMoment, fourthMoment, variance, kurtosis, peak,
     reference, comparison},
    rawIntegral = integrateOnDomain[name, Function[{x}, k[x]]];
    densityIntegral = integrateOnDomain[name, Function[{x}, k[x]/d]];
    firstMoment = integrateOnDomain[name, Function[{x}, x k[x]/d]];
    secondMoment = integrateOnDomain[name, Function[{x}, x^2 k[x]/d]];
    fourthMoment = integrateOnDomain[name,
      Function[{x}, (x - firstMoment)^4 k[x]/d]];
    variance = secondMoment - firstMoment^2;
    kurtosis = fourthMoment/variance^2;
    peak = peakNumerators[name]/d;
    reference = scipyReference[name];
    comparison = <|
      "NormaliserMatchesSciPy" ->
        Abs[N[d, 40] - SetPrecision[reference["Normaliser"], 40]] < 10^-9,
      "VarianceMatchesSciPy" ->
        Abs[N[variance, 40] - SetPrecision[reference["Variance"], 40]] < 10^-9,
      "KurtosisMatchesSciPy" ->
        Abs[N[kurtosis, 40] - SetPrecision[reference["Kurtosis"], 40]] < 10^-9
    |>;
    <|
      "NormalisingConstant" -> N[d, 35],
      "DirectKernelIntegral" -> N[rawIntegral, 35],
      "NormalisedDensityIntegral" -> N[densityIntegral, 35],
      "FirstMoment" -> N[firstMoment, 25],
      "Variance" -> N[variance, 35],
      "FourthCentralMoment" -> N[fourthMoment, 35],
      "Kurtosis" -> N[kurtosis, 35],
      "PeakDensity" -> N[peak, 35],
      "PeakStrictlyBelowOne" -> TrueQ[peak < 1],
      "KernelIntegralMinusPrintedConstant" -> N[rawIntegral - d, 20],
      "SciPyComparison" -> comparison
    |>
  ],
  {name, names}
];

cupAnalyticBound = 43/15 > E;
cupNumericalPeak = peakNumerators["Cup"]/normalisers["Cup"];

summary = <|
  "Engine" -> "Wolfram Mathematica / Wolfram Language",
  "WolframVersion" -> $Version,
  "SystemID" -> $SystemID,
  "WorkingPrecision" -> workingPrecision,
  "AccuracyGoal" -> accuracyGoal,
  "PrecisionGoal" -> precisionGoal,
  "MahaloySincConvention" -> "Sin[u]/u, with value 1 at u=0",
  "MahaloyInteriorZeroSplits" -> N[mahaloyCuts, 20],
  "AllSixNormalisedIntegralsCloseToOne" ->
    AllTrue[Values[mathematicaResults],
      Abs[#["NormalisedDensityIntegral"] - 1] < 10^-25 &],
  "AllSixPeaksStrictlyBelowOne" ->
    AllTrue[Values[mathematicaResults], TrueQ[#["PeakStrictlyBelowOne"]] &],
  "AllSixSciPyComparisonsPass" ->
    AllTrue[Values[mathematicaResults], And @@ Values[#["SciPyComparison"]] &],
  "CupStrictBoundProvedByElementaryInequality" -> cupAnalyticBound,
  "CupPeak" -> N[cupNumericalPeak, 35],
  "Results" -> mathematicaResults
|>;

Print[Dataset[mathematicaResults]];
Print[InputForm[KeyDrop[summary, "Results"]]];

repoRoot = DirectoryName[DirectoryName[ExpandFileName[$InputFileName]]];
resultsDirectory = FileNameJoin[{repoRoot, "results"}];
If[!DirectoryQ[resultsDirectory], CreateDirectory[resultsDirectory]];
outputPath = FileNameJoin[{resultsDirectory, "six_density_mathematica_crosscheck.json"}];
Export[outputPath, summary, "RawJSON"];
Print["Exported Mathematica results to: ", outputPath];
