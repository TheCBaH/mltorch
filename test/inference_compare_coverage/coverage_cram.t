Every row of the coverage manifest against test_convnext2, sample 0. Opt-in
(MLTORCH_COVERAGE, PT2_DATA): about four minutes. A row whose engine is not
built here is reported pending with the reason, so the lines below are the ones
that hold wherever this runs; `make model.compare.coverage` prints them all.

  $ m="$PT2_SAFETENSORS_MODELS/test_convnext2"; i="$PT2_DATA/test_convnext2/inputs.pt"
  $ ./coverage_walk.exe coverage.manifest ../inference_compare_pt2/fixture_pair_run.exe "$m" "$i" > walk.out; echo "exit $?"
  exit 0
  $ grep -c "FAIL" walk.out
  0
  [1]
  $ grep -E "^(self-native|sequential32-bitwise|native4d-arena|arena-over-budget):" walk.out
  self-native: ok
  sequential32-bitwise: ok
  arena-over-budget: ok
  native4d-arena: ok
