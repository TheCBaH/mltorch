The embedded CompCert route against the direct assembly route on the canonical
Native graph of real models, through the shared-frontier coordinator. Gated on
MLTORCH_COMPCERT (the embedded CompCert for this host) and PT2_DATA.

Every unit's two outputs agree, under either donor:

  $ M="$PT2_DATA/test_convnext2/test_convnext2.pt2"
  $ I="$PT2_DATA/test_convnext2/inputs.pt"
  $ for d in left right; do DONOR=$d ./compiled_run.exe "$M" "$I" 2>/dev/null; done
  extracted 98 graphs
  execution=complete numerical=agreement executed=49 units
  comparisons: 49 pass, 0 mismatch, 0 other
  extracted 98 graphs
  execution=complete numerical=agreement executed=49 units
  comparisons: 49 pass, 0 mismatch, 0 other

Each route's segmented run equals its own whole-graph run, bitwise:

  $ for d in left right; do CALIBRATE=1 DONOR=$d ./compiled_run.exe "$M" "$I" 2>/dev/null | grep calibration; done
  calibration: 1 final output(s) bitwise equal the whole-graph run
  calibration: 1 final output(s) bitwise equal the whole-graph run

A zero tolerance compares the same values (the routes are bitwise equal here,
so nothing is reported), and a larger model with many units agrees too:

  $ N="$PT2_DATA/mobilenetv3_small_050/mobilenetv3_small_050.pt2"
  $ TOL=0 ./compiled_run.exe "$N" "$PT2_DATA/mobilenetv3_small_050/inputs.pt" 2>/dev/null
  extracted 318 graphs
  execution=complete numerical=agreement executed=159 units
  comparisons: 159 pass, 0 mismatch, 0 other

No host compiler is involved: with no tool on the search path the CompCert
route still compiles and runs.

  $ env -i PATH=/nonexistent MLTORCH_COMPCERT=aarch64 ./compiled_run.exe "$M" "$I" 2>/dev/null
  extracted 98 graphs
  execution=complete numerical=agreement executed=49 units
  comparisons: 49 pass, 0 mismatch, 0 other
