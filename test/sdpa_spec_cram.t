Evaluate `torch.ops.aten.scaled_dot_product_attention.default` through the
real ATen kernel only (op8-impl.md commit 3).

No model in this repo's zoo serializes this target (F1 of op8-impl.md): the
string appears only inside `from_node` provenance of the four `vit_*` graphs,
whose actual nodes are its twelve-primitive decomposition. These five
hand-written fixtures are therefore the row's only serialized evidence,
following `unbind_spec_cram.t`'s precedent for the same reason.

`--eval` prints ATen's result and does not compare it against anything --
that is `test/native/compute_test.ml` (hand-computed), `test/native_bridge_test.ml`
(the ATen-live-tensor importer path) and `test/native_interp/sdpa_test.ml`
(the serialized-metadata importer path). This file only proves that a real
`.pt2`-shaped op-spec, at every mask/scale form the flash oracle admits (F4),
decodes and runs through the archive commit 2 built.

Five fixtures, one per admissible form: no mask (default scale), a rank-2
`[Wq,Wk]` mask, a rank-4 `[D,H,Wq,Wk]` mask, a rank-4 mask broadcasting `D`
and `H` together (`[1,1,Wq,Wk]`), and an explicit scale. `D=2,H=3` and
`Wq=5,Wk=6` are asymmetric on purpose, so a transposed axis or a swapped
D/H would show up as a shape or a status failure rather than passing by
coincidence.

  $ for f in data/sdpa/*.json; do ../bin/aten_spec_verify.exe --eval "$f"; done
  === data/sdpa/0_no_mask_default_scale.json ===
  [node] torch.ops.aten.scaled_dot_product_attention.default(query=f32[2,3,5,8]~normal(mean=0,variance=1), key=f32[2,3,6,8]~normal(mean=0,variance=1), value=f32[2,3,6,8]~normal(mean=0,variance=1), attn_mask=none, dropout_p=0., is_causal=false, scale=none, enable_gqa=false)
    -> out0: [2,3,5,8]
    status: ok
  === data/sdpa/1_mask_2d.json ===
  [node] torch.ops.aten.scaled_dot_product_attention.default(query=f32[2,3,5,8]~normal(mean=0,variance=1), key=f32[2,3,6,8]~normal(mean=0,variance=1), value=f32[2,3,6,8]~normal(mean=0,variance=1), attn_mask=f32[5,6]~normal(mean=0,variance=0.01), dropout_p=0., is_causal=false, scale=none, enable_gqa=false)
    -> out0: [2,3,5,8]
    status: ok
  === data/sdpa/2_mask_4d_full.json ===
  [node] torch.ops.aten.scaled_dot_product_attention.default(query=f32[2,3,5,8]~normal(mean=0,variance=1), key=f32[2,3,6,8]~normal(mean=0,variance=1), value=f32[2,3,6,8]~normal(mean=0,variance=1), attn_mask=f32[2,3,5,6]~normal(mean=0,variance=0.01), dropout_p=0., is_causal=false, scale=none, enable_gqa=false)
    -> out0: [2,3,5,8]
    status: ok
  === data/sdpa/3_mask_4d_broadcast_d_h.json ===
  [node] torch.ops.aten.scaled_dot_product_attention.default(query=f32[2,3,5,8]~normal(mean=0,variance=1), key=f32[2,3,6,8]~normal(mean=0,variance=1), value=f32[2,3,6,8]~normal(mean=0,variance=1), attn_mask=f32[1,1,5,6]~normal(mean=0,variance=0.01), dropout_p=0., is_causal=false, scale=none, enable_gqa=false)
    -> out0: [2,3,5,8]
    status: ok
  === data/sdpa/4_explicit_scale.json ===
  [node] torch.ops.aten.scaled_dot_product_attention.default(query=f32[2,3,5,8]~normal(mean=0,variance=1), key=f32[2,3,6,8]~normal(mean=0,variance=1), value=f32[2,3,6,8]~normal(mean=0,variance=1), attn_mask=none, dropout_p=0., is_causal=false, scale=0.0500000007451, enable_gqa=false)
    -> out0: [2,3,5,8]
    status: ok
