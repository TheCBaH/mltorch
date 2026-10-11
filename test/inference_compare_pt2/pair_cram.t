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
