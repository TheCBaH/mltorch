Native lowering frontier over the six tracked release models, from their
committed payload-free model.json (no PT2_DATA gate needed -- same as
mobilenetv2_050/test_convnext2's existing rules). Pins the outcome AFTER
`gelu.default`/`sigmoid.default`/scalar `mul.Tensor`/`mul.Scalar` support,
after `select.int`/`unsqueeze.default` and `cat.default`/`stack.default`
first landed, and after the design-goal fix gave `select.int`
and `stack.default` their own single `Select`/`Stack` Native nodes (each had
decomposed into a `Slice`+`Reshape` pair, resp. N `Reshape`s + `Concat`):
a regression here means either the frontier moved backward (a real bug) or
moved forward silently (this test not updated to match). This cram reports
only capability/blocker status, not node counts, so the internal node-shape
change is invisible here by design -- see `native_bridge_test.ml`'s
"builds a single Select/Stack node" cases for that.

`csatv2` moved past its `select.int` block to `stack.default`, and now past
that too: its `stage:initial_native` frontier is
`torch.ops.aten.clone.default: memory_format is not supported` (section 3 item
8, not yet landed). This is a MALFORMED rejection, not an `unsupported_operator`
capability -- the importer refuses the whole graph rather than degrading one
capability's status -- so the script's `else` branch fires and the row reads
"blocked: ..." instead of the two `stage:` lines every other model prints; that
is the harness noticing a different *kind* of rejection, not a broken script.
`unsqueeze.default` never showed up as a named blocker for any of the six
models either before or after either change -- its coverage is
`native_bridge_test.ml`'s dedicated verify/dispatch cases, not this cram.

`test_convnext2` is the one surprise this test caught, in two stages. It
serializes `gelu.default` with `approximate="tanh"`, not `"none"` -- an
earlier scope decision implemented only the exact, erf-based `"none"` form, so
this model was blocked on the `Unsupported_option` rejection. `Gelu` now
carries an `approximate` field (`Exact | Tanh`) and both dialects' `Compute`
implement PyTorch's tanh formula (proved against real ATen in
`native_bridge_test.ml`'s "verify: gelu (tanh)..." and, at the Native4D
lowering, structurally in `native4d/verify_test.ml`'s "gelu tanh" cluster), so
`test_convnext2` is now fully available at both stages.

  $ for m in mobilenetv2_050 regnetx_002 test_convnext2 efficientnet_b0 fastvit_sa12 csatv2; do
  >   if ../bin/native_graph.exe visualize --model ${m}_model.json --output ${m}-session.json 2>${m}.err; then
  >     python3 -c "
  > import json
  > s = json.load(open('${m}-session.json'))
  > rows = {c['key']: c['status'] for c in s['capabilities']}
  > diags = {d['code']: d['message'] for d in s['diagnostics']}
  > def show(key):
  >     st = rows[key]
  >     if st['state'] == 'available':
  >         print('${m} %s available %s' % (key, st['payload']['kind']))
  >     else:
  >         detail = diags.get(st['reason'], '')
  >         suffix = (': ' + detail) if detail else ''
  >         print('${m} %s unavailable %s%s' % (key, st['reason'], suffix))
  > show('stage:initial_native')
  > show('stage:native4d')"
  >   else
  >     echo "${m} blocked: $(cat ${m}.err)"
  >   fi
  > done
  mobilenetv2_050 stage:initial_native available graph
  mobilenetv2_050 stage:native4d available graph
  regnetx_002 stage:initial_native available graph
  regnetx_002 stage:native4d unavailable outside_dialect_domain: node n373: convolution has 3 groups, which is neither 1 nor depthwise
  test_convnext2 stage:initial_native available graph
  test_convnext2 stage:native4d available graph
  efficientnet_b0 stage:initial_native available graph
  efficientnet_b0 stage:native4d available graph
  fastvit_sa12 stage:initial_native available graph
  fastvit_sa12 stage:native4d unavailable outside_dialect_domain: node n604: axis T is outside the N/H/W/C dialect
  csatv2 stage:initial_native unavailable unsupported_operator: unsupported PT2 operator: torch.ops.aten.index.Tensor
  csatv2 stage:native4d unavailable prerequisite_unavailable

`regnetx_002`'s `stage:native4d` was already `unavailable outside_dialect_domain`
before this change (a 3-group convolution, neither 1 nor depthwise -- unrelated
to any op this change touches); the plan's premise that it was "already
available" at that stage was mistaken. `stage:initial_native` is the row this
change actually guards there, and it is unchanged. `fastvit_sa12`'s
`stage:native4d` stays `unavailable outside_dialect_domain` too, pre-existing
and unrelated to Gelu/Sigmoid/Mul_scalar (this model uses SDPA, which
`native4d_design.md` section 8 documents as an intentional domain rejection) --
not a regression from this change.
