Retention, replay and evidence integrity on the synthetic residual experiment.

  $ R=../../..
  $ gen() { $R/test/inference_compare/cli/fixture_gen.exe "$@"; }
  $ cmp() { $R/bin/inference_compare.exe "$@"; }
  $ scrub() { sed 's/sha256:[0-9a-f]*/sha256:DIGEST/g; s|sha256/[0-9a-f]\{64\}|sha256/DIGEST|g'; }

Retaining every unit keeps the common inputs, raw outputs and candidates, each
once (identical bytes share one file).

  $ gen all deviate --retention all
  $ cmp run --experiment all/experiment.json --evidence-root all/evidence --run-id r1
  run all/evidence/run-r1
  execution=complete numerical=disagreement units=3
  [1]
  $ ls all/evidence/run-r1/payloads/sha256 | wc -l | tr -d ' '
  6
  $ grep -o '"omissions":\[[^]]*\]' all/evidence/run-r1/retained.json
  "omissions":[]

A single unit replays from its retained common inputs and reproduces exactly
the recorded comparison and digests, including the mismatch at A.

  $ cmp replay --run all/evidence/run-r1 --unit A
  replay unit A: identical to the record
  note: re-executed from the retained common inputs of the unit
  $ cmp replay --run all/evidence/run-r1 --unit B
  replay unit B: identical to the record
  note: re-executed from the retained common inputs of the unit
  $ cmp replay --run all/evidence/run-r1 --unit Z
  error: unit Z is not in the recorded plan
  [2]

The full trajectory replays from the pinned initial payloads and recorded donors.

  $ cmp replay --run all/evidence/run-r1
  replay full: identical to the record

With no retention nothing but the pinned identities survives, and a unit replay
says so rather than inventing inputs.

  $ gen none deviate
  $ cmp run --experiment none/experiment.json --evidence-root none/evidence --run-id r1
  run none/evidence/run-r1
  execution=complete numerical=disagreement units=3
  [1]
  $ ls none/evidence/run-r1/payloads/sha256 2>/dev/null | wc -l | tr -d ' '
  0
  $ cmp replay --run none/evidence/run-r1 --unit B
  error: unit B: retained input(s) h, w were never retained
  [4]
  $ cmp replay --run none/evidence/run-r1
  replay full: identical to the record

Retaining only the first mismatch keeps that unit alone.

  $ gen first deviate --retention first_mismatch
  $ cmp run --experiment first/experiment.json --evidence-root first/evidence --run-id r1 > /dev/null
  [1]
  $ cmp replay --run first/evidence/run-r1 --unit A | head -1
  replay unit A: identical to the record
  $ cmp replay --run first/evidence/run-r1 --unit B
  error: unit B: retained input(s) h, w were never retained
  [4]

A best-effort disk budget records what it could not keep; a required one fails
the run before anything is published.

  $ gen budget deviate --retention all --disk-bytes 30
  $ cmp run --experiment budget/experiment.json --evidence-root budget/evidence --run-id r1 > /dev/null
  [1]
  $ grep -o '"reason":"[a-z_]*"' budget/evidence/run-r1/retained.json | sort | uniq -c | tr -s ' '
   16 "reason":"budget_exhausted"
  $ gen strict deviate --retention all --disk-bytes 30 --required
  $ cmp run --experiment strict/experiment.json --evidence-root strict/evidence --run-id r1
  run strict/evidence/run-r1
  execution=failed numerical=not_established units=0
  failure: evidence_failed
  [3]

Corrupted, tampered or interrupted evidence is never replayed.

  $ cp -r all/evidence/run-r1 tamper_events
  $ echo '{"extra":1}' >> tamper_events/events.jsonl
  $ cmp replay --run tamper_events --unit A
  error: events.jsonl does not match its pinned digest
  [4]
  $ cp -r all/evidence/run-r1 interrupted
  $ rm interrupted/completion.json
  $ cmp replay --run interrupted --unit A
  error: interrupted has no completion record: the run was interrupted
  [4]
  $ cp -r all/evidence/run-r1 tamper_graph
  $ sed 's/n3_r/n3_x/g' tamper_graph/inputs/right.graph.json > tmp.json && mv tmp.json tamper_graph/inputs/right.graph.json
  $ cmp replay --run tamper_graph --unit A 2>&1 | scrub
  error: incompatible identity: right_graph differs from the recorded input
  $ cp -r all/evidence/run-r1 tamper_bindings
  $ sed 's/exact_integer/exact_boolean/g' tamper_bindings/policy_bindings.json > tmp.json && mv tmp.json tamper_bindings/policy_bindings.json
  $ cmp replay --run tamper_bindings --unit A
  error: policy_bindings.json does not match its pinned digest
  [4]
  $ cp -r all/evidence/run-r1 tamper_payload
  $ for f in tamper_payload/payloads/sha256/*; do printf 'garbage-garbage-garbage' > $f; done
  $ cmp replay --run tamper_payload --unit A 2>&1 | scrub
  error: sha256/DIGEST does not match its pinned digest

Identical payloads are stored, and charged to the budget, once: 120 bytes hold
five of the six distinct 24-byte payloads.

  $ gen dedupe deviate --retention all --disk-bytes 120
  $ cmp run --experiment dedupe/experiment.json --evidence-root dedupe/evidence --run-id r1 > /dev/null
  [1]
  $ ls dedupe/evidence/run-r1/payloads/sha256 | wc -l | tr -d ' '
  5
