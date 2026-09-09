(* `torch.ops.aten.index.Tensor`: [Op_bridge]'s dispatch arm, restricted to
   the evidenced shape family -- [indices] has at most [self]'s ATen rank
   many entries, exactly one live entry of ATen rank at least 1, every other
   listed position [None] -- and its locked typed rejections
   (`.ai/index_tensor_design.md` rounds 3/9, and the multi-entry-gather
   landing's [index_rank] generalization). No real ATen call is made
   ([Tensor?[]] has no [lib/aten_gen] C-shim support), so these fixtures are
   hand-derived rather than [Interp_verify]-compared, the same choice
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
      n0: [t2 f32 [W=2 C=3]] =
        index_tensor self=t0 index=t1 params={axis=W index_rank=1}
    outputs: [t2 f32 [W=2 C=3] <-n0]
    tensor f32 [W=2 C=3] {4, 5, 6, 1, 2, 3} |}]

(* Real ATen implicitly full-slices any dims [indices] doesn't mention, so a
   list SHORTER than [self]'s rank is not itself a fault -- this is exactly
   `mvitv2_tiny`/`maxxvitv2_nano_rw_256`'s own encoding
   (`.ai/index_tensor_design.md`): a length-1 [indices] against a rank-2
   [self]. Same result as gathering with an explicit trailing [None]. *)
let%expect_test
    "dispatch: index.Tensor accepts an indices list shorter than self's rank" =
  let self = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let idx = i64_tensor [ 2 ] [ 1L; 0L ] in
  dispatch_print ~target:"torch.ops.aten.index.Tensor"
    ~bindings:[ ("self", self); ("idx0", idx) ]
    ~inputs:[ in_tensor "self"; in_optional_tensors "indices" [ `T "idx0" ] ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=3] {4, 5, 6, 1, 2, 3} |}]

let%expect_test
    "dispatch: index.Tensor rejects an indices list longer than self's rank" =
  let self = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let idx = i64_tensor [ 2 ] [ 1L; 0L ] in
  dispatch_print ~target:"torch.ops.aten.index.Tensor"
    ~bindings:[ ("self", self); ("idx0", idx) ]
    ~inputs:
      [
        in_tensor "self";
        in_optional_tensors "indices" [ `T "idx0"; `None; `None ];
      ]
    ~noutputs:1;
  [%expect
    {| error: index.Tensor: indices has 3 entries, more than self's rank 2 |}]

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

(* `mvitv2_tiny`/`maxxvitv2_nano_rw_256`'s own shape family
   (`.ai/index_tensor_design.md`): a live entry of ATen rank 2 -- accepted
   since [axis] is [self]'s own outermost dim here, so there is nothing of
   [self]'s to lose to index's extra axis. Output is
   [index.shape ++ self.shape[1:]]: [idx]'s own [[1],[0]] selects self's
   rows 1 then 0, each kept as its own leading axis rather than collapsed. *)
let%expect_test "dispatch: index.Tensor accepts a live entry of ATen rank 2" =
  let self = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let idx = i64_tensor [ 2; 1 ] [ 1L; 0L ] in
  dispatch_print ~target:"torch.ops.aten.index.Tensor"
    ~bindings:[ ("self", self); ("idx0", idx) ]
    ~inputs:
      [ in_tensor "self"; in_optional_tensors "indices" [ `T "idx0"; `None ] ]
    ~noutputs:1;
  [%expect {| tensor f32 [H=2 W=1 C=3] {4, 5, 6, 1, 2, 3} |}]

(* `Self_collision` (`.ai/index_tensor_design.md`): a rank-3 [self] with real
   content on every axis, gathered at a MIDDLE position with a rank-2 live
   entry -- index's extra axis would have to borrow [self]'s own leading
   axis H (extent 2, not 1), which real ATen never discards. Reachable
   through the importer itself, unlike [Rank_overflow]
   (`test/native/graph_direct_index_tensor_test.ml`'s own proof), since a
   real ATen graph can name any dim position, not only self's outermost. *)
let%expect_test
    "dispatch: index.Tensor rejects a rank-2 entry that would overwrite self's \
     own axis" =
  let self =
    float_tensor [ 2; 3; 4 ] (List.init 24 (fun i -> float_of_int (i + 1)))
  in
  let idx = i64_tensor [ 2; 1 ] [ 1L; 0L ] in
  dispatch_print ~target:"torch.ops.aten.index.Tensor"
    ~bindings:[ ("self", self); ("idx0", idx) ]
    ~inputs:
      [
        in_tensor "self";
        in_optional_tensors "indices" [ `None; `T "idx0"; `None ];
      ]
    ~noutputs:1;
  [%expect
    {| error: index.Tensor: a multi-axis index at axis W would overwrite self's own axis H, which must have extent 1 (got 2) |}]

let%expect_test
    "dispatch: index.Tensor rejects an all-None indices list (no live entry)" =
  let self = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  dispatch_print ~target:"torch.ops.aten.index.Tensor"
    ~bindings:[ ("self", self) ]
    ~inputs:[ in_tensor "self"; in_optional_tensors "indices" [ `None; `None ] ]
    ~noutputs:1;
  [%expect {| error: index.Tensor: indices has no live (non-None) entry |}]
