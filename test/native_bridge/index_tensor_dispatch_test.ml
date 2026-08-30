(* `torch.ops.aten.index.Tensor`: [Op_bridge]'s dispatch arm, restricted to
   the one evidenced shape family -- [indices] has exactly one live entry, of
   ATen rank 1, every other position [None] -- and its locked typed
   rejections (`.ai/index_tensor_design.md` rounds 3/9). No real ATen call is
   made ([Tensor?[]] has no [lib/aten_gen] C-shim support), so these fixtures
   are hand-derived rather than [Interp_verify]-compared, the same choice
   already made for `addcmul.default`/`group_norm.default`. *)

open Helpers

let%expect_test
    "dispatch: index.Tensor gathers along dim 0, a non-last live position" =
  let self = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let idx = i64_tensor [ 2 ] [ 1L; 0L ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.index.Tensor"
    ~bindings:[ ("self", self); ("idx0", idx) ]
    ~inputs:
      [ in_tensor "self"; in_optional_tensors "indices" [ `T "idx0"; `None ] ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0], t1 i64 [C=2] ->[n0]]
    nodes:
      n0: [t2 f32 [W=2 C=3]] = index_tensor self=t0 index=t1 params={axis=W}
    outputs: [t2 f32 [W=2 C=3] <-n0]
    tensor f32 [W=2 C=3] {4, 5, 6, 1, 2, 3} |}]

let%expect_test "dispatch: index.Tensor rejects a wrong-length indices list" =
  let self = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let idx = i64_tensor [ 2 ] [ 1L; 0L ] in
  dispatch_print ~target:"torch.ops.aten.index.Tensor"
    ~bindings:[ ("self", self); ("idx0", idx) ]
    ~inputs:[ in_tensor "self"; in_optional_tensors "indices" [ `T "idx0" ] ]
    ~noutputs:1;
  [%expect
    {| error: index.Tensor: indices has 1 entries, expected 2 (self's rank) |}]

let%expect_test "dispatch: index.Tensor rejects two live entries" =
  let self = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let idx0 = i64_tensor [ 2 ] [ 1L; 0L ] in
  let idx1 = i64_tensor [ 3 ] [ 2L; 1L; 0L ] in
  dispatch_print ~target:"torch.ops.aten.index.Tensor"
    ~bindings:[ ("self", self); ("idx0", idx0); ("idx1", idx1) ]
    ~inputs:
      [
        in_tensor "self"; in_optional_tensors "indices" [ `T "idx0"; `T "idx1" ];
      ]
    ~noutputs:1;
  [%expect
    {| error: index.Tensor: indices has more than one live entry, at positions [0, 1] |}]

let%expect_test
    "dispatch: index.Tensor rejects a boolean-mask entry (wrong dtype)" =
  let self = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let mask = T.create ~dtype:Stype.Bool [ 2 ] in
  dispatch_print ~target:"torch.ops.aten.index.Tensor"
    ~bindings:[ ("self", self); ("mask", mask) ]
    ~inputs:
      [ in_tensor "self"; in_optional_tensors "indices" [ `T "mask"; `None ] ]
    ~noutputs:1;
  [%expect {| error: index.Tensor: indices[0] must be Long, got Bool |}]

let%expect_test "dispatch: index.Tensor rejects a live entry of ATen rank 2" =
  let self = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let idx = i64_tensor [ 2; 1 ] [ 1L; 0L ] in
  dispatch_print ~target:"torch.ops.aten.index.Tensor"
    ~bindings:[ ("self", self); ("idx0", idx) ]
    ~inputs:
      [ in_tensor "self"; in_optional_tensors "indices" [ `T "idx0"; `None ] ]
    ~noutputs:1;
  [%expect {| error: index.Tensor: indices[0] must be rank 1, got rank 2 |}]

let%expect_test
    "dispatch: index.Tensor rejects an all-None indices list (no live entry)" =
  let self = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  dispatch_print ~target:"torch.ops.aten.index.Tensor"
    ~bindings:[ ("self", self) ]
    ~inputs:[ in_tensor "self"; in_optional_tensors "indices" [ `None; `None ] ]
    ~noutputs:1;
  [%expect {| error: index.Tensor: indices has no live (non-None) entry |}]
