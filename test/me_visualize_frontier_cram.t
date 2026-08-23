Native lowering frontier over the six tracked release models, from their
committed payload-free model.json (no PT2_DATA gate needed -- same as
mobilenetv2_050/test_convnext2's existing rules). Pins the outcome AFTER
`gelu.default`/`sigmoid.default`/scalar `mul.Tensor`/`mul.Scalar` support: a
regression here means either the frontier moved backward (a real bug) or moved
forward silently (this test not updated to match).

`test_convnext2` is the one surprise this test caught: it serializes
`gelu.default` with `approximate="tanh"`, not `"none"` -- the scope decision
(confirmed with the user) implements only the exact, erf-based `"none"` form,
so this model stays blocked, now on the `Unsupported_option` rejection rather
than on a missing `Gelu` op entirely. [`Unsupported_option`] classifies as
`Fatal` in [Me_classify.lowering] (the same as a non-default `alpha` or
`ceil_mode=true` elsewhere), so `visualize` exits non-zero rather than
producing a session with an `unavailable` capability row -- exactly as
`me_visualize_unsupported_cram.t` documents for a malformed graph.

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
  test_convnext2 blocked: native_graph: malformed PT2 graph: torch.ops.aten.gelu.default: approximate="tanh" is not supported (only "none")
  efficientnet_b0 stage:initial_native available graph
  efficientnet_b0 stage:native4d available graph
  fastvit_sa12 stage:initial_native available graph
  fastvit_sa12 stage:native4d unavailable outside_dialect_domain: node n604: axis T is outside the N/H/W/C dialect
  csatv2 stage:initial_native unavailable unsupported_operator: unsupported PT2 operator: torch.ops.aten.select.int
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
