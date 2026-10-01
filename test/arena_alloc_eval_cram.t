Evaluate the arena allocators on a two-model corpus: every strategy, the order
search, the portfolio and the reference minimum, for both normalized dialects.
No weights, no inputs: each dialect's allocation script comes from a dry run.

  $ for m in regnetx_002 test_convnext2; do
  >   mkdir -p corpus/$m/models && cp ${m}_model.json corpus/$m/models/model.json
  > done
  $ E="../bin/arena_alloc_eval.exe --models-dir corpus --reference-max-states 20000 --reference-max-depth 256"
  $ $E --iterations 0,25 --expected-models 2 --output eval.jsonl --summary eval.md --models-output models.jsonl --artifacts art 2>/dev/null
  $ grep -o '"model":"[^"]*","dialect":"[^"]*","type":"dialect","status":"[^"]*"' eval.jsonl
  "model":"regnetx_002","dialect":"native","type":"dialect","status":"evaluated"
  "model":"regnetx_002","dialect":"native4d","type":"dialect","status":"evaluated"
  "model":"test_convnext2","dialect":"native","type":"dialect","status":"evaluated"
  "model":"test_convnext2","dialect":"native4d","type":"dialect","status":"evaluated"
  $ grep '"type":"coverage"' eval.jsonl
  {"type":"coverage","dialect":"native","expected":2,"attempted":2,"evaluated":2,"refused":0,"prerequisite":0,"failed":0}
  {"type":"coverage","dialect":"native4d","expected":2,"attempted":2,"evaluated":2,"refused":0,"prerequisite":0,"failed":0}
  $ grep -o '"dialect":"[^"]*","type":"reference","kind":"[^"]*"' eval.jsonl | sort | uniq -c
        2 "dialect":"native","type":"reference","kind":"float32"
        2 "dialect":"native4d","type":"reference","kind":"float32"

Per kind: four constructive rows, four improved rows per budget and seed, and
one portfolio row per budget and seed, so 4 + 4*2 + 2 = 14.

  $ grep -c '"type":"strategy"' eval.jsonl
  56
  $ grep '^Models completed' eval.md
  Models completed: 2/2. Result: complete.
  $ ls art | wc -l
  4

One row per evaluated model and dialect, after the row naming the columns:
the arena of each, in bytes, summed over kinds.

  $ grep -c '"type":"model_arena"' models.jsonl
  4
  $ head -1 models.jsonl
  {"type":"model_arena_columns","pools":["greedy_by_area","greedy_by_area+improve@0/1","greedy_by_area+improve@25/1","greedy_by_lifetime","greedy_by_lifetime+improve@0/1","greedy_by_lifetime+improve@25/1","greedy_by_size","greedy_by_size+improve@0/1","greedy_by_size+improve@25/1","greedy_by_size_best_fit","greedy_by_size_best_fit+improve@0/1","greedy_by_size_best_fit+improve@25/1","portfolio@0/1","portfolio@25/1"]}

A corpus that does not hold the expected number of models is refused before
anything runs.

  $ $E --expected-models 3 --output other.jsonl
  arena_alloc_eval: expected 3 models, found 2
  [1]
  $ test -e other.jsonl || echo "no output"
  no output

Resuming needs the same configuration: a different budget grid is refused
rather than mixed into the earlier rows, and the rows stay as they were.

  $ cp eval.jsonl before.jsonl
  $ $E --iterations 0 --expected-models 2 --output eval.jsonl --resume
  arena_alloc_eval: cannot resume: the existing output was written with a different configuration, revision or corpus
  [1]
  $ cmp eval.jsonl before.jsonl && echo unchanged
  unchanged

The reference limits have no default: an unbounded exhaustive search is never
launched by omission.

  $ ../bin/arena_alloc_eval.exe --models-dir corpus --expected-models 2 --output x.jsonl 2>&1 | grep missing
  arena_alloc_eval: required option --reference-max-states is missing
