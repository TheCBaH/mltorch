Evaluate `torch.ops.aten.unbind.int` — the first bound op with a
variable-length `Tensor[]` return — through the real ATen kernel only.

`--eval` prints ATen's results and does not compare them against anything. That
is this file's job and it stays that way now that a native unbind exists: what
it proves is that the whole `Tensor[]` path agrees on the result COUNT, which
needs no second implementation. The ATen-versus-native comparison lives where
the oracle is — `test/native_bridge_test.ml`'s verify cases,
`test/aten_spec_run_test.ml`'s `eval_print` golden, and the generated bridge
walk in `test/native_walk_test.ml`.

This is the end-to-end proof that the whole Tensor[] path agrees on the result
count: `Aten_spec_run.outputs_for` predicts it from `self`'s shape at the
normalized `dim` and synthesizes that many output names, ATen returns however
many it returns, and `Interp_decode.bind_tensor_list` pairs the two through
`Err.List.map2`. A disagreement anywhere is an arity error here, not a silent
pass. The output lines themselves come from `Interp_decode.output_names`, which
flattens the single `as_tensors` output — filtering for `Argument.Tensor` alone
would print no `->` lines at all.

Five fixtures, one per `(dim, out_rank)` configuration recorded for
`unbind.int` in `ops-aten.yaml` at v0.0.3. **`out_rank` is the rank of each
RETURNED tensor**, so an input is one rank higher — unbind drops the dimension
it splits.

Only `2_dim_absent_out_rank4_vit.json` is provenance-backed: `[3,1,3,101,32]` is
the exact input of the first unbind node in `test_vit4`, from its producer
chain (`view_1 [1,101,3,3,32]` -> `permute_1 dims [2,0,3,1,4]` -> unbind),
consistent with the matrix recording `test_vit4 unbind.int: {3: 9}`. The other
four are representative synthetic shapes at the required rank and dim, because
`ops-aten.yaml` records configurations, not input shapes.

That fixture also **omits `dim` entirely**, matching the real exported node —
the serializer leaves a defaulted argument out. That exercises the generated
codec's `~dec_absent:(Int 0)` path, which the hand-built interpreter tests in
`native_bridge_test.ml` cannot reach.

  $ for f in data/unbind/*.json; do ../bin/aten_spec_verify.exe --eval "$f"; done
  === data/unbind/0_dim_neg1_out_rank2.json ===
  [node] torch.ops.aten.unbind.int(self=f32[2,4,3]~normal(mean=0,variance=1), dim=-1)
    -> out0: [2,4]
    -> out1: [2,4]
    -> out2: [2,4]
    status: ok
  === data/unbind/1_dim0_out_rank3.json ===
  [node] torch.ops.aten.unbind.int(self=f32[2,3,4,5]~normal(mean=0,variance=1), dim=0)
    -> out0: [3,4,5]
    -> out1: [3,4,5]
    status: ok
  === data/unbind/2_dim_absent_out_rank4_vit.json ===
  [node] torch.ops.aten.unbind.int(self=f32[3,1,3,101,32]~normal(mean=0,variance=1), dim=0)
    -> out0: [1,3,101,32]
    -> out1: [1,3,101,32]
    -> out2: [1,3,101,32]
    status: ok
  === data/unbind/3_dim0_out_rank5.json ===
  [node] torch.ops.aten.unbind.int(self=f32[2,2,3,2,3,2]~normal(mean=0,variance=1), dim=0)
    -> out0: [2,3,2,3,2]
    -> out1: [2,3,2,3,2]
    status: ok
  === data/unbind/4_dim2_out_rank4.json ===
  [node] torch.ops.aten.unbind.int(self=f32[2,3,4,3,2]~normal(mean=0,variance=1), dim=2)
    -> out0: [2,3,3,2]
    -> out1: [2,3,3,2]
    -> out2: [2,3,3,2]
    -> out3: [2,3,3,2]
    status: ok
