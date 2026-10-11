The real ATen / Native pair driven by the dedicated command, from an exported
experiment directory to a replayed run. Gated on PT2_DATA (`make pt2.runtest`).

  $ IC=../../bin/inference_compare.exe
  $ M="$PT2_DATA/test_convnext2/test_convnext2.pt2"
  $ I="$PT2_DATA/test_convnext2/inputs.pt"
  $ mkdir exp ev
  $ $IC export-pt2 --model "$M" --inputs "$I" --out exp
  wrote exp/experiment.json
  $ $IC validate --experiment exp/experiment.json
  valid: 71 unit(s), 71 comparison site(s), coverage complete

A run agrees, is written as immutable evidence, and replays identically:

  $ $IC run --experiment exp/experiment.json --evidence-root ev --run-id r1
  run ev/run-r1
  execution=complete numerical=agreement units=71
  $ $IC replay --run ev/run-r1
  replay full: identical to the record

A model that is not the pinned file is refused before anything runs:

  $ sed 's/"sha256":"[0-9a-f]*"/"sha256":"00"/' exp/experiment.json > exp/bad.json
  $ $IC plan --experiment exp/bad.json 2>&1 | sed 's|[^ ]*test_convnext2.pt2|MODEL|' | cut -c1-36
  error: left engine: MODEL: sha256 is

The same model against the four-axis evaluator, after the canonical passes:

  $ mkdir exp4 ev4
  $ $IC export-pt2 --model "$M" --inputs "$I" --right native4d --out exp4
  wrote exp4/experiment.json
  $ $IC validate --experiment exp4/experiment.json
  valid: 7 unit(s), 7 comparison site(s), coverage complete
  $ $IC run --experiment exp4/experiment.json --evidence-root ev4 --run-id r1
  run ev4/run-r1
  execution=complete numerical=agreement units=7
  $ $IC replay --run ev4/run-r1
  replay full: identical to the record

Two-pass certification of the Native engine against ATen as the reference. The
segmented ATen run does not reproduce ATen's own complete graph bit for bit, so
demanding exact partition fidelity stops the workflow at the complete-graph
stage with no golden; declaring the contract certifies, and the golden is
qualified as within it:

  $ mkdir cert-exact cert-declared cert-aten
  $ $IC export-pt2 --model "$M" --inputs "$I" --certify exact --out cert-exact >/dev/null
  $ $IC certify --workflow cert-exact/validation.json --evidence-root cert-exact/ev
  not certified: full_graph_confirmation: partition fidelity of the reference engine: output out0 is not bitwise equal
  [1]
  $ ls cert-exact/ev/cert-pt2-pair
  stages
  $ $IC export-pt2 --model "$M" --inputs "$I" --certify declared --out cert-declared >/dev/null
  $ $IC certify --workflow cert-declared/validation.json --evidence-root cert-declared/ev
  certified cert-declared/ev/cert-pt2-pair (within the declared contract)
  $ ls cert-declared/ev/cert-pt2-pair/golden
  golden.json
  output-0.bin
  outputs.json

With ATen under test the same exact demand fails on the tested engine itself:

  $ $IC export-pt2 --model "$M" --inputs "$I" --certify exact --tested left --out cert-aten >/dev/null
  $ $IC certify --workflow cert-aten/validation.json --evidence-root cert-aten/ev
  not certified: full_graph_confirmation: partition fidelity of the tested engine: output out0 is not bitwise equal
  [1]
