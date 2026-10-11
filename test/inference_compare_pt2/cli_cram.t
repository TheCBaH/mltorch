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
