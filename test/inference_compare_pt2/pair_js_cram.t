The JavaScript engines as variants (`VIEW:loop-js` and `ssa-js`, the Loop IR
bundle's and the SSA backend's generated JavaScript, and `VIEW:jsoo`, the Native
evaluator compiled by js_of_ocaml), each run by node through a helper process
that speaks the protocol of the wire library. Gated on PT2_DATA and MLTORCH_JS.

The two generated-JavaScript routes against the direct Native evaluator, within
the default tolerance, and against each other bit for bit:

  $ m="$PT2_SAFETENSORS_MODELS/test_convnext2"; i="$PT2_DATA/test_convnext2/inputs.pt"
  $ for e in loop-js ssa-js jsoo; do
  >   ./fixture_pair_run.exe safetensors "$m" "$i" 0 native native.canonical:$e 2>/dev/null | grep -E "^execution|^comparisons"
  > done
  execution=complete numerical=agreement executed=13 units
  comparisons: 13 pass, 0 mismatch, 0 other
  execution=complete numerical=agreement executed=13 units
  comparisons: 13 pass, 0 mismatch, 0 other
  execution=complete numerical=agreement executed=13 units
  comparisons: 13 pass, 0 mismatch, 0 other
  $ TOL=0 ./fixture_pair_run.exe safetensors "$m" "$i" 0 native.canonical:loop-js native.canonical:ssa-js 2>/dev/null | grep -E "^execution|^comparisons"
  execution=complete numerical=agreement executed=49 units
  comparisons: 49 pass, 0 mismatch, 0 other

JavaScript's math library is not the C library's, so against the direct
evaluator at zero tolerance the last bit differs in the clusters with a GELU:
the comparison is not vacuous.

  $ TOL=0 ./fixture_pair_run.exe safetensors "$m" "$i" 0 native native.canonical:loop-js 2>/dev/null | grep -c "numerical=disagreement"
  1

The four-axis evaluator against the Native evaluator under js_of_ocaml:

  $ ./fixture_pair_run.exe safetensors "$m" "$i" 0 native4d native.canonical:jsoo 2>/dev/null | grep -E "^execution|^comparisons"
  execution=complete numerical=agreement executed=40 units
  comparisons: 40 pass, 0 mismatch, 0 other
