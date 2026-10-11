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

Any two graph views and either orientation, on a small model (`fixture_pair_run`
takes variants; `native:sequential32` accumulates dots in binary32). Each pair's
units come from the composed or reversed correspondences of the importer, the
canonical passes and the four-axis lowering:

  $ m="$PT2_SAFETENSORS_MODELS/test_convnext2"; i="$PT2_DATA/test_convnext2/inputs.pt"
  $ for p in "aten native.canonical" "native.canonical native" "native4d native.canonical" "native native4d"; do
  >   ./fixture_pair_run.exe safetensors "$m" "$i" 0 $p 2>/dev/null | grep -E "clusters|^comparisons"
  > done
  left aten | right native.canonical | donor left | tolerance 0.0001 | 8 clusters
  comparisons: 8 pass, 0 mismatch, 0 other
  left native.canonical | right native | donor left | tolerance 0.0001 | 13 clusters
  comparisons: 13 pass, 0 mismatch, 0 other
  left native4d | right native.canonical | donor left | tolerance 0.0001 | 40 clusters
  comparisons: 40 pass, 0 mismatch, 0 other
  left native | right native4d | donor left | tolerance 0.0001 | 12 clusters
  comparisons: 12 pass, 0 mismatch, 0 other

One view against itself, one cluster per node. The same view twice agrees
exactly at zero tolerance, while two dot accumulations differ in the last bits
(the check is not vacuous):

  $ for p in "native.canonical native.canonical" "native4d native4d"; do
  >   TOL=0 ./fixture_pair_run.exe safetensors "$m" "$i" 0 $p 2>/dev/null | grep -E "^comparisons"
  > done
  comparisons: 49 pass, 0 mismatch, 0 other
  comparisons: 40 pass, 0 mismatch, 0 other
  $ TOL=0 ./fixture_pair_run.exe safetensors "$m" "$i" 0 aten native.canonical:sequential32 2>/dev/null | grep -c "numerical=disagreement"
  1

A pair with no mapping is refused, not run:

  $ ./fixture_pair_run.exe safetensors "$m" "$i" 0 native aten 2>/dev/null; echo "exit $?"
  refused: ATen on the right needs the mapping from Native to ATen
  exit 3

The Native evaluator under the storage and scheduling configurations inference
runs with (`native:WORD,...`: `release`, `arena`, `arena=BYTES`, `separate`,
`shared`, and `scheduled` to reorder the body first). Each is bitwise equal to
the default (every configuration is checked bitwise on a hermetic body in
`native_storage_test`), and the report says what each resolved to:

  $ for c in arena,scheduled separate; do
  >   TOL=0 ./fixture_pair_run.exe safetensors "$m" "$i" 0 native native:$c 2>/dev/null | grep -E "^comparisons|storage:"
  > done
  comparisons: 106 pass, 0 mismatch, 0 other
  native:arena,scheduled storage: 106 body(ies): 106 in an arena (0 reusing storage), 0 role-planned, 0 release-only
  comparisons: 106 pass, 0 mismatch, 0 other
  native:separate storage: 106 body(ies): 0 in an arena (0 reusing storage), 106 role-planned, 0 release-only

A single node has nothing to reuse. Against ATen the canonical graph is large
bodies, some of which have intermediates to share:

  $ ./fixture_pair_run.exe safetensors "$m" "$i" 0 aten native.canonical:arena 2>/dev/null | grep -E "^comparisons|storage:"
  comparisons: 8 pass, 0 mismatch, 0 other
  native.canonical:arena storage: 8 body(ies): 8 in an arena (4 reusing storage), 0 role-planned, 0 release-only

A budget a later body exceeds refuses at preparation, before any unit runs and
so with no comparison:

  $ ./fixture_pair_run.exe safetensors "$m" "$i" 0 native native.canonical:arena=100000 2>/dev/null | grep -E "^execution|^comparisons|^failure|storage:"
  execution=refused numerical=not_established executed=0 units
  failure: preparation_refused phase=preparation unit=cu00000 text=engine: arena: the run needs 307200 bytes but the budget is 100000
  comparisons: 0 pass, 0 mismatch, 0 other
  native.canonical:arena=100000 storage: 1 body(ies): 0 in an arena (0 reusing storage), 0 role-planned, 0 release-only
