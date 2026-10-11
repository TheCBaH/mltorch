The two-pass certification of a synthetic three-unit experiment: stage gating,
and an immutable golden record only when every stage passes.

  $ R=../../..
  $ gen() { $R/test/inference_compare/cli/fixture_gen.exe "$@"; }
  $ cmp() { $R/bin/inference_compare.exe "$@"; }

Identical engines certify exactly. Each stage is persisted before the next, the
passes and the complete-graph run are ordinary run directories, and the golden
record holds the tested engine's own complete output:

  $ gen ok agree --certify
  $ cmp certify --workflow ok/validation.json --evidence-root ok/evidence
  certified ok/evidence/cert-exp (exact)
  $ lsd() { for d in "$@"; do echo "$d:"; ls "$d" | sed 's/^/  /'; done; }
  $ lsd ok/evidence ok/evidence/cert-exp ok/evidence/cert-exp/stages ok/evidence/cert-exp/golden
  ok/evidence:
    cert-exp
    run-exp-full
    run-exp-p1
    run-exp-p2
  ok/evidence/cert-exp:
    golden
    stages
  ok/evidence/cert-exp/stages:
    1-reference_trajectory.json
    2-tested_trajectory.json
    3-full_graph_confirmation.json
    4-required_end_to_end_checks.json
  ok/evidence/cert-exp/golden:
    golden.json
    output-0.bin
    outputs.json
  $ grep -o '"qualification":"[a-z_]*"' ok/evidence/cert-exp/golden/golden.json
  "qualification":"exact"

A certification is not repeated into the same directory:

  $ cmp certify --workflow ok/validation.json --evidence-root ok/evidence
  not certified: contract: certification directory: ok/evidence/cert-exp: File exists
  [2]

A first-pass disagreement ends the workflow: pass 2 never runs and no golden
exists, but the failed pass's evidence is kept.

  $ gen bad deviate --certify
  $ cmp certify --workflow bad/validation.json --evidence-root bad/evidence
  not certified: reference_trajectory: a local comparison disagrees (evidence: bad/evidence/run-exp-p1)
  [1]
  $ lsd bad/evidence bad/evidence/cert-exp
  bad/evidence:
    cert-exp
    run-exp-p1
  bad/evidence/cert-exp:

The contract can require the complete tested output to match a frozen
reference. A matching reference certifies; a different one does not, although
every local check passed, and no golden is written:

  $ gen acc agree --certify --required-acceptance
  $ cp ok/evidence/cert-exp/golden/output-0.bin acc/frozen.bin
  $ cmp certify --workflow acc/validation.json --evidence-root acc/evidence
  certified acc/evidence/cert-exp (exact)
  $ gen wrong agree --certify --required-acceptance
  $ head -c 24 /dev/zero > wrong/frozen.bin
  $ cmp certify --workflow wrong/validation.json --evidence-root wrong/evidence
  not certified: required_end_to_end_checks: full-graph acceptance against the frozen reference: output out0 fails its final policy
  [1]
  $ ls wrong/evidence/cert-exp
  stages

A peer-only comparison, a contract that requires an acceptance no reference
freezes, and a changed contract are refused before anything runs:

  $ gen peers agree --certify --peers
  $ cmp certify --workflow peers/validation.json --evidence-root peers/evidence
  not certified: contract: a peer-only comparison cannot certify correctness: there is no trusted side
  [2]
  $ gen req agree --certify --required-acceptance --no-frozen
  $ cmp certify --workflow req/validation.json --evidence-root req/evidence
  not certified: contract: the contract requires a full-graph acceptance and the workflow freezes no reference output for some final
  [2]
  $ sed 's/"required_full_graph_acceptance":false/"required_full_graph_acceptance":true/' ok/contract.json > ok/contract2.json
  $ sed 's/contract.json/contract2.json/' ok/validation.json > ok/validation2.json
  $ cmp certify --workflow ok/validation2.json --evidence-root ok/evidence2 --id other
  not certified: contract: the workflow's contract digest is not the contract's
  [2]
  $ ls ok/evidence2 2>&1 | head -2
  ls: cannot access 'ok/evidence2': No such file or directory
