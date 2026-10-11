The real ATen / Native pair through the shared-frontier coordinator, one model
and one sample. Gated on PT2_DATA; run via `make pt2.runtest`.

Every unit's two outputs agree within binary32 tolerance:

  $ for m in test_convnext2 mobilenetv3_small_050; do
  >   echo "== $m"
  >   ./pair_run.exe "$PT2_DATA/$m/$m.pt2" "$PT2_DATA/$m/inputs.pt" 2>/dev/null
  > done
  == test_convnext2
  extracted 142 graphs
  execution=complete numerical=agreement executed=71 units
  comparisons: 71 pass, 0 mismatch, 0 other
  == mobilenetv3_small_050
  extracted 314 graphs
  execution=complete numerical=agreement executed=157 units
  comparisons: 157 pass, 0 mismatch, 0 other

The comparison is not vacuous: at zero tolerance the engines' last-bit
differences are reported, and the run still completes (a numerical
disagreement is a result, not a failure).

  $ TOL=0 ./pair_run.exe "$PT2_DATA/test_convnext2/test_convnext2.pt2" "$PT2_DATA/test_convnext2/inputs.pt" 2>/dev/null | grep -c "numerical=disagreement"
  1

Each engine's segmented route (every unit fed that engine's own committed
values, via either donor) is calibrated against its whole-graph run, bitwise:

  $ for d in left right; do
  >   CALIBRATE=1 DONOR=$d ./pair_run.exe "$PT2_DATA/mobilenetv3_small_050/mobilenetv3_small_050.pt2" "$PT2_DATA/mobilenetv3_small_050/inputs.pt" 2>/dev/null | grep calibration
  > done
  calibration: 1 final output(s) bitwise equal the whole-graph run
  calibration: 1 final output(s) bitwise equal the whole-graph run

The same models against the four-axis evaluator (after the canonical passes,
composed ATen -> Native -> canonical Native -> Native4D). The units are the
planner's coarsening of the passes' own correspondence:

  $ for m in test_convnext2 mobilenetv3_small_050 mobilenetv2_050 regnetx_002; do
  >   echo "== $m"
  >   RIGHT=native4d ./pair_run.exe "$PT2_DATA/$m/$m.pt2" "$PT2_DATA/$m/inputs.pt" 2>/dev/null
  > done
  == test_convnext2
  extracted 14 graphs
  execution=complete numerical=agreement executed=7 units
  comparisons: 7 pass, 0 mismatch, 0 other
  == mobilenetv3_small_050
  extracted 28 graphs
  execution=complete numerical=agreement executed=14 units
  comparisons: 14 pass, 0 mismatch, 0 other
  == mobilenetv2_050
  extracted 4 graphs
  execution=complete numerical=agreement executed=2 units
  comparisons: 2 pass, 0 mismatch, 0 other
  == regnetx_002
  extracted 4 graphs
  execution=complete numerical=agreement executed=2 units
  comparisons: 2 pass, 0 mismatch, 0 other

The canonical Native graph and its pass map exist without the four-axis
lowering, which runs only when forced (a model Native4D refuses keeps them):

  $ ./canonical_run.exe "$PT2_DATA/mobilenetv2_050/mobilenetv2_050.pt2" "$PT2_DATA/mobilenetv2_050/inputs.pt" 2>/dev/null
  canonical: built, lowering forced: false
  canonicalize map: 40 unit(s)
  lowering: ok, 100 node(s)
  lowering forced: true

Not vacuous here either, and each donor's segmented route matches the
four-axis whole-graph run bitwise:

  $ RIGHT=native4d TOL=0 ./pair_run.exe "$PT2_DATA/test_convnext2/test_convnext2.pt2" "$PT2_DATA/test_convnext2/inputs.pt" 2>/dev/null | grep -c "numerical=disagreement"
  1
  $ for d in left right; do
  >   RIGHT=native4d CALIBRATE=1 DONOR=$d ./pair_run.exe "$PT2_DATA/mobilenetv3_small_050/mobilenetv3_small_050.pt2" "$PT2_DATA/mobilenetv3_small_050/inputs.pt" 2>/dev/null | grep calibration
  > done
  calibration: 1 final output(s) bitwise equal the whole-graph run
  calibration: 1 final output(s) bitwise equal the whole-graph run

Regrouping the three maps gives the same composed meaning, on real passes:

  $ ASSOC=1 RIGHT=native4d ./pair_run.exe "$PT2_DATA/test_convnext2/test_convnext2.pt2" "$PT2_DATA/test_convnext2/inputs.pt" 2>/dev/null | head -1
  (ATen.canonicalize).lower vs ATen.(canonicalize.lower): same meaning (493 facts)

The same pair with no .pt2: the graph and weight map are the submodule's model
directory and the weights the pinned Hub checkpoint (`fixture_pair_run
safetensors`). The cluster count and the verdict equal the .pt2 run's, and at
zero tolerance the engines still differ (the check is not vacuous):

  $ ./fixture_pair_run.exe safetensors "$PT2_SAFETENSORS_MODELS/mobilenetv2_050" "$PT2_DATA/mobilenetv2_050/inputs.pt" 0 aten native 2>/dev/null | grep -E "clusters|^execution|^comparisons"
  left aten | right native | donor left | tolerance 0.0001 | 152 clusters
  execution=complete numerical=agreement executed=152 units
  comparisons: 152 pass, 0 mismatch, 0 other
  $ ./pair_run.exe "$PT2_DATA/mobilenetv2_050/mobilenetv2_050.pt2" "$PT2_DATA/mobilenetv2_050/inputs.pt" 2>/dev/null | grep -E "^execution|^comparisons"
  execution=complete numerical=agreement executed=152 units
  comparisons: 152 pass, 0 mismatch, 0 other
  $ TOL=0 ./fixture_pair_run.exe safetensors "$PT2_SAFETENSORS_MODELS/mobilenetv2_050" "$PT2_DATA/mobilenetv2_050/inputs.pt" 0 aten native 2>/dev/null | grep -c "numerical=disagreement"
  1
