(* the scaled_dot_product_attention.default arm. Split from the former native_bridge_test.ml; promote with [dune promote test/native_bridge/sdpa_test.ml]. *)

open Helpers

(* ---- the scaled_dot_product_attention.default arm ------------------------ *)

(* Mask present vs absent, structurally: [Graph_ir]'s [Sdpa] carries
   [mask : Tensor_ref.t option], and an arm that materialised a zero tensor
   for an absent mask would build a graph [Native_interp] cannot also build
   for the same node (op8-impl.md commit 3). Query=[1,0], key0=[1,0] (dot=1),
   key1=[0,1] (dot=0); explicit scale=1 so score=dot; value0=[10,20],
   value1=[30,40] -- the same hand-computed case as compute_test.ml's "two
   keys, unequal scores", cross-checking the bridge's decode against the same
   known-correct numbers rather than a fresh derivation. *)
let%expect_test "dispatch: sdpa builds the graph, mask present vs absent" =
  let q = float_tensor [ 1; 1; 1; 2 ] [ 1.; 0. ] in
  let k = float_tensor [ 1; 1; 2; 2 ] [ 1.; 0.; 0.; 1. ] in
  let v = float_tensor [ 1; 1; 2; 2 ] [ 10.; 20.; 30.; 40. ] in
  let mask = float_tensor [ 1; 2 ] [ 0.; 0. ] in
  let state label bindings inputs =
    Format.printf "%s@." label;
    dispatch_print_with_graph ~print_graph:true
      ~target:"torch.ops.aten.scaled_dot_product_attention.default" ~bindings
      ~inputs ~noutputs:1
  in
  state "mask absent:"
    [ ("query", q); ("key", k); ("value", v) ]
    [
      in_tensor "query";
      in_tensor "key";
      in_tensor "value";
      in_none "attn_mask";
      in_float "dropout_p" 0.0;
      in_bool "is_causal" false;
      in_float "scale" 1.0;
      in_bool "enable_gqa" false;
    ];
  state "mask present:"
    [ ("query", q); ("key", k); ("value", v); ("attn_mask", mask) ]
    [
      in_tensor "query";
      in_tensor "key";
      in_tensor "value";
      in_tensor "attn_mask";
      in_float "dropout_p" 0.0;
      in_bool "is_causal" false;
      in_float "scale" 1.0;
      in_bool "enable_gqa" false;
    ];
  [%expect
    {|
    mask absent:
    graph
    inputs:
      [t0 f32 [C=2] ->[n0], t1 f32 [W=2 C=2] ->[n0], t2 f32 [W=2 C=2] ->[n0]]
    nodes:
      n0: [t3 f32 [C=2]] =
        sdpa query=t0 key=t1 value=t2 mask=none params={scale=explicit(1)}
    outputs: [t3 f32 [C=2] <-n0]
    tensor f32 [C=2] {15.3788, 25.3788}
    mask present:
    graph
    inputs:
      [t0 f32 [C=2] ->[n0], t1 f32 [W=2 C=2] ->[n0], t2 f32 [W=2 C=2] ->[n0],
       t3 f32 [C=2] ->[n0]]
    nodes:
      n0: [t4 f32 [C=2]] =
        sdpa query=t0 key=t1 value=t2 mask=t3 params={scale=explicit(1)}
    outputs: [t4 f32 [C=2] <-n0]
    tensor f32 [C=2] {15.3788, 25.3788} |}]

(* Default scale: E=4, so 1/sqrt(4) = 0.5. Same numbers as
   compute_test.ml's "Direct: sdpa -- default scale is 1/sqrt(head_dim)". *)
let%expect_test "dispatch: sdpa defaults scale to 1/sqrt(head_dim)" =
  let q = float_tensor [ 1; 1; 1; 4 ] [ 1.; 1.; 1.; 1. ] in
  let k = float_tensor [ 1; 1; 2; 4 ] [ 1.; 1.; 1.; 1.; 0.; 0.; 0.; 0. ] in
  let v = float_tensor [ 1; 1; 2; 4 ] [ 1.; 1.; 1.; 1.; 2.; 2.; 2.; 2. ] in
  dispatch_print ~target:"torch.ops.aten.scaled_dot_product_attention.default"
    ~bindings:[ ("query", q); ("key", k); ("value", v) ]
    ~inputs:[ in_tensor "query"; in_tensor "key"; in_tensor "value" ]
    ~noutputs:1;
  [%expect "tensor f32 [C=4] {1.1192, 1.1192, 1.1192, 1.1192}"]

let%expect_test "dispatch: sdpa rejects dropout_p, is_causal, enable_gqa" =
  let q = float_tensor [ 1; 1; 1; 1 ] [ 1. ] in
  let k = float_tensor [ 1; 1; 1; 1 ] [ 1. ] in
  let v = float_tensor [ 1; 1; 1; 1 ] [ 1. ] in
  let bindings = [ ("query", q); ("key", k); ("value", v) ] in
  let base = [ in_tensor "query"; in_tensor "key"; in_tensor "value" ] in
  dispatch_print ~target:"torch.ops.aten.scaled_dot_product_attention.default"
    ~bindings
    ~inputs:(base @ [ in_float "dropout_p" 0.5 ])
    ~noutputs:1;
  dispatch_print ~target:"torch.ops.aten.scaled_dot_product_attention.default"
    ~bindings
    ~inputs:(base @ [ in_bool "is_causal" true ])
    ~noutputs:1;
  dispatch_print ~target:"torch.ops.aten.scaled_dot_product_attention.default"
    ~bindings
    ~inputs:(base @ [ in_bool "enable_gqa" true ])
    ~noutputs:1;
  [%expect
    {|
    error: sdpa: dropout_p=0.5 is not supported (only 0)
    error: sdpa: is_causal=true is not supported
    error: sdpa: enable_gqa=true is not supported |}]

let%expect_test
    "dispatch: sdpa rejects a boolean mask (never reinterpreted as f32)" =
  let q = float_tensor [ 1; 1; 1; 1 ] [ 1. ] in
  let k = float_tensor [ 1; 1; 1; 1 ] [ 1. ] in
  let v = float_tensor [ 1; 1; 1; 1 ] [ 1. ] in
  let mask = T.create ~dtype:Stype.Bool [ 1; 1 ] in
  dispatch_print ~target:"torch.ops.aten.scaled_dot_product_attention.default"
    ~bindings:[ ("query", q); ("key", k); ("value", v); ("attn_mask", mask) ]
    ~inputs:
      [
        in_tensor "query";
        in_tensor "key";
        in_tensor "value";
        in_tensor "attn_mask";
      ]
    ~noutputs:1;
  [%expect
    {| error: sdpa: a boolean attn_mask is not supported; only an additive f32 mask is (never reinterpreted from bool) |}]

(* Rank is checked on the raw ATen tensor, before conversion erases it
   (op8-impl.md F13): a rank-3 [1,Wq,Wk] mask normalizes to the SAME Native
   shape as an admissible rank-4 [1,1,Wq,Wk] one, so only a raw-rank check
   catches it. *)
let%expect_test "dispatch: sdpa rejects wrong-rank query/key/value/mask" =
  let bad3 = float_tensor [ 1; 1; 1 ] [ 1. ] in
  let ok4 = float_tensor [ 1; 1; 1; 1 ] [ 1. ] in
  dispatch_print ~target:"torch.ops.aten.scaled_dot_product_attention.default"
    ~bindings:[ ("query", bad3); ("key", ok4); ("value", ok4) ]
    ~inputs:[ in_tensor "query"; in_tensor "key"; in_tensor "value" ]
    ~noutputs:1;
  dispatch_print ~target:"torch.ops.aten.scaled_dot_product_attention.default"
    ~bindings:[ ("query", ok4); ("key", bad3); ("value", ok4) ]
    ~inputs:[ in_tensor "query"; in_tensor "key"; in_tensor "value" ]
    ~noutputs:1;
  dispatch_print ~target:"torch.ops.aten.scaled_dot_product_attention.default"
    ~bindings:[ ("query", ok4); ("key", ok4); ("value", bad3) ]
    ~inputs:[ in_tensor "query"; in_tensor "key"; in_tensor "value" ]
    ~noutputs:1;
  let mask3 = float_tensor [ 1; 1; 1 ] [ 0. ] in
  dispatch_print ~target:"torch.ops.aten.scaled_dot_product_attention.default"
    ~bindings:
      [ ("query", ok4); ("key", ok4); ("value", ok4); ("attn_mask", mask3) ]
    ~inputs:
      [
        in_tensor "query";
        in_tensor "key";
        in_tensor "value";
        in_tensor "attn_mask";
      ]
    ~noutputs:1;
  [%expect
    {|
    error: sdpa query must be rank-4, got rank-3
    error: sdpa key must be rank-4, got rank-3
    error: sdpa value must be rank-4, got rank-3
    error: sdpa: sdpa attn_mask has rank 3, expected 2 or 4 |}]

(* Real-ATen grounding for head broadcasting (query H=2, key/value H=1,
   `enable_gqa=false`): `.ai/attention_design.md`'s own gate table
   (`check_batch_size_and_num_heads_dense`) requires EQUAL heads for the
   fused/flash kernel unless GQA is enabled, so this configuration is
   already off the flash kernel in real ATen -- it falls to the `math`
   backend, which this op's arithmetic already mirrors structurally (§6).
   [verify_print] runs the SAME node through real ATen and through
   [Op_bridge]+[Eval_direct], comparing at [Verify.atol_for_target]'s
   existing single `1e-5` entry (silence means agreement) -- the first
   evidence this project's one atol also holds off the flash kernel, not
   only on it. *)
let%expect_test
    "verify: sdpa head broadcasting agrees with real ATen (math backend, \
     unequal heads without GQA)" =
  let q =
    float_tensor [ 1; 2; 1; 2 ] [ 1.; 0.; (* head 0 *) 0.; 1. (* head 1 *) ]
  in
  let k = float_tensor [ 1; 1; 2; 2 ] [ 1.; 0.; 0.; 1. ] in
  let v = float_tensor [ 1; 1; 2; 2 ] [ 10.; 20.; 30.; 40. ] in
  verify_print ~target:"torch.ops.aten.scaled_dot_product_attention.default"
    ~bindings:[ ("query", q); ("key", k); ("value", v) ]
    ~inputs:
      [
        in_tensor "query";
        in_tensor "key";
        in_tensor "value";
        in_none "attn_mask";
        in_float "dropout_p" 0.0;
        in_bool "is_causal" false;
        in_float "scale" 1.0;
        in_bool "enable_gqa" false;
      ];
  [%expect {| aten and native agree |}]

(* Same grounding, batch axis (D in the Native frame, ATen's leading dim):
   query has two real batch elements, key/value share ONE (broadcast). Also
   off the flash kernel (`check_batch_size_and_num_heads_dense` requires
   equal BATCH size unconditionally, GQA or not), so this is math-backend
   too. *)
let%expect_test
    "verify: sdpa batch broadcasting agrees with real ATen (math backend, \
     unequal batch)" =
  let q =
    float_tensor [ 2; 1; 1; 2 ] [ 1.; 0.; (* batch 0 *) 0.; 1. (* batch 1 *) ]
  in
  let k = float_tensor [ 1; 1; 2; 2 ] [ 1.; 0.; 0.; 1. ] in
  let v = float_tensor [ 1; 1; 2; 2 ] [ 10.; 20.; 30.; 40. ] in
  verify_print ~target:"torch.ops.aten.scaled_dot_product_attention.default"
    ~bindings:[ ("query", q); ("key", k); ("value", v) ]
    ~inputs:
      [
        in_tensor "query";
        in_tensor "key";
        in_tensor "value";
        in_none "attn_mask";
        in_float "dropout_p" 0.0;
        in_bool "is_causal" false;
        in_float "scale" 1.0;
        in_bool "enable_gqa" false;
      ];
  [%expect {| aten and native agree |}]

let%expect_test "dispatch: sdpa rejects a negative or non-finite explicit scale"
    =
  let q = float_tensor [ 1; 1; 1; 1 ] [ 1. ] in
  let k = float_tensor [ 1; 1; 1; 1 ] [ 1. ] in
  let v = float_tensor [ 1; 1; 1; 1 ] [ 1. ] in
  let bindings = [ ("query", q); ("key", k); ("value", v) ] in
  let base = [ in_tensor "query"; in_tensor "key"; in_tensor "value" ] in
  dispatch_print ~target:"torch.ops.aten.scaled_dot_product_attention.default"
    ~bindings
    ~inputs:(base @ [ in_float "scale" (-1.0) ])
    ~noutputs:1;
  dispatch_print ~target:"torch.ops.aten.scaled_dot_product_attention.default"
    ~bindings
    ~inputs:(base @ [ in_float "scale" Float.nan ])
    ~noutputs:1;
  dispatch_print ~target:"torch.ops.aten.scaled_dot_product_attention.default"
    ~bindings
    ~inputs:(base @ [ in_float "scale" Float.infinity ])
    ~noutputs:1;
  [%expect
    {|
    error: sdpa: explicit scale=-1 is negative; ATen handles this specially and this row does not implement it
    error: sdpa: explicit scale=nan is not finite
    error: sdpa: explicit scale=inf is not finite |}]
