End-to-end behaviour of the inference_compare command on a synthetic
three-unit experiment (units A, B, D over exact int64 arithmetic).

  $ R=../../..
  $ gen() { $R/test/inference_compare/cli/fixture_gen.exe "$@"; }
  $ cmp() { $R/bin/inference_compare.exe "$@"; }

Two identical engines agree completely; the run is validated, planned and then
executed, and the evidence directory is finalized.

  $ gen agree agree
  $ cmp validate --experiment agree/experiment.json
  valid: 3 unit(s), 3 comparison site(s), coverage complete
  $ cmp plan --experiment agree/experiment.json --out agree/plan.json | sed 's/sha256:[0-9a-f]*/sha256:DIGEST/'
  plan sha256:DIGEST written to agree/plan.json
  $ cmp run --experiment agree/experiment.json --evidence-root agree/evidence --run-id r1
  run agree/evidence/run-r1
  execution=complete numerical=agreement units=3
  $ echo $?
  0
  $ ls agree/evidence/run-r1
  completion.json
  events.jsonl
  inputs
  manifest.json
  payloads
  plan.json
  policy_bindings.json
  retained.json
  summary.json

A deviating engine is a completed numerical disagreement (exit 1), with the
mismatch localized to unit A and recorded as such; downstream units still agree
because both engines consume the same committed value.

  $ gen deviate deviate
  $ cmp run --experiment deviate/experiment.json --evidence-root deviate/evidence --run-id r1
  run deviate/evidence/run-r1
  execution=complete numerical=disagreement units=3
  [1]
  $ grep -o '"comparison":"[a-z_]*"\|"verdict":"[a-z_]*"' deviate/evidence/run-r1/summary.json
  "comparison":"cmp_h"
  "verdict":"mismatch"
  "comparison":"cmp_m"
  "verdict":"pass"
  "comparison":"cmp_y"
  "verdict":"pass"

The same run directory name cannot be reused.

  $ cmp run --experiment deviate/experiment.json --evidence-root deviate/evidence --run-id r1
  error: run directory deviate/evidence/run-r1 already exists (use --run-id)
  [2]

An invalid mapping (a node no unit claims) is rejected before anything runs.

  $ gen unmapped unmapped
  $ cmp validate --experiment unmapped/experiment.json 2>&1 | sed 's/sha256:[0-9a-f]*/sha256:DIGEST/'
  error: mapping: missing_owner side=left node=aten/v1@sha256:DIGEST:/n3
  $ cmp run --experiment unmapped/experiment.json --evidence-root unmapped/evidence 2>&1 | sed 's/sha256:[0-9a-f]*/sha256:DIGEST/'
  error: mapping: missing_owner side=left node=aten/v1@sha256:DIGEST:/n3
  $ ls unmapped
  experiment.json
  left.graph.json
  mapping.json
  right.graph.json
  selection.json
  w.bin
  x.bin

Transfer corruption stops the run before either engine executes anything: the
evidence is still finalized, with the failure recorded.

  $ gen corrupt corrupt
  $ cmp run --experiment corrupt/experiment.json --evidence-root corrupt/evidence --run-id r1
  run corrupt/evidence/run-r1
  execution=failed numerical=not_established units=0
  failure: input_transfer_failed
  [3]
  $ grep -c graph_completed corrupt/evidence/run-r1/events.jsonl
  0
  [1]

A refused preparation is a refusal, not a result.

  $ gen refuse refuse
  $ cmp run --experiment refuse/experiment.json --evidence-root refuse/evidence --run-id r1
  run refuse/evidence/run-r1
  execution=refused numerical=not_established units=0
  failure: preparation_refused
  [3]
