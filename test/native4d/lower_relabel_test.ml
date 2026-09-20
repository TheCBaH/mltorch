(* [Lower_relabel] beyond split attention: permute, expand, clone, the scalar
   and activation ops, batched_matmul and the reductions, all with D read as N.
   Every case lowers a Native graph and compares each output element-for-element
   with Native's own evaluation, on values that are not exact in f32, so a
   mis-conjugated permutation or a wrong reduction axis is a mismatch. *)

open Axis

let s = Fixtures.s
let perm_of = Fixtures.perm_of
let inexact = Lower_region_test.inexact
let compare_graphs = Lower_region_test.compare_graphs ~fill:inexact
let build name b = Graph_builder.build ~name ~outputs:Fun.id b
let ok g = Err.or_raise ~pp_error:Graph_builder.pp_error g
let sum_d = { Reduce.Sum.dims = [ D ]; keepdim = false }

(* [x] is [H=4 W=3 C=6]; the reshape gives it a real D. *)
let onto_d = s 1 1 4 3 3 2

let permuted_group perm =
  build "permute"
    (let open Graph_builder in
     let* x = input ~shape:(s 1 1 1 4 3 6) () in
     let* r = reshape { Reshape.Reshape.shape = onto_d } x in
     let* p = permute (perm_of perm) r in
     let* q = silu p in
     let* out = sum sum_d q in
     Graph_builder.return [ out ])
  |> ok

let%expect_test "permute of a D-carrying value, D moved and moved back" =
  List.iter
    (fun (name, perm) ->
      compare_graphs name ~x_shapes:[ s 1 1 1 4 3 6 ] (permuted_group perm))
    [
      ("D<->H", [ (D, H); (H, D) ]);
      ("D<->W", [ (D, W); (W, D) ]);
      ("D->H->W->D", [ (D, H); (H, W); (W, D) ]);
    ];
  [%expect
    {|
    D<->H: 1 outputs, 4 nodes, identical=true
    D<->W: 1 outputs, 4 nodes, identical=true
    D->H->W->D: 1 outputs, 4 nodes, identical=true |}]

(* The permute is where the extent enters or leaves D: only one side is
   relabelled. *)
let%expect_test "permute that introduces or removes the D extent" =
  let enter =
    build "enter"
      (let open Graph_builder in
       let* x = input ~shape:(s 1 1 1 4 3 6) () in
       let* p = permute (perm_of [ (D, H); (H, D) ]) x in
       let* q = relu p in
       let* out = sum sum_d q in
       Graph_builder.return [ out ])
    |> ok
  in
  compare_graphs "enter" ~x_shapes:[ s 1 1 1 4 3 6 ] enter;
  let leave =
    build "leave"
      (let open Graph_builder in
       let* x = input ~shape:(s 1 1 1 4 3 6) () in
       let* r = reshape { Reshape.Reshape.shape = s 1 1 4 1 3 6 } x in
       let* q = mul_scalar 1.5 r in
       let* p = permute (perm_of [ (D, H); (H, D) ]) q in
       Graph_builder.return [ p ])
    |> ok
  in
  compare_graphs "leave" ~x_shapes:[ s 1 1 1 4 3 6 ] leave;
  [%expect
    {|
    enter: 1 outputs, 3 nodes, identical=true
    leave: 1 outputs, 3 nodes, identical=true |}]

(* Outlook attention: a batch of D windows, each [H=5 W=9 C=9] x [.. C=7]. *)
let matmul_graph =
  build "batched_matmul"
    (let open Graph_builder in
     let* a = input ~shape:(s 1 1 1 5 9 54) () in
     let* b = input ~shape:(s 1 1 1 5 9 42) () in
     let* a = reshape { Reshape.Reshape.shape = s 1 1 6 5 9 9 } a in
     let* b = reshape { Reshape.Reshape.shape = s 1 1 6 5 9 7 } b in
     let* m = batched_matmul a b in
     let* m = add_scalar 0.25 m in
     let* out = sum sum_d m in
     Graph_builder.return [ out ])
  |> ok

let%expect_test "batched_matmul over a D batch" =
  compare_graphs "outlook"
    ~x_shapes:[ s 1 1 1 5 9 54; s 1 1 1 5 9 42 ]
    matmul_graph;
  [%expect {| outlook: 1 outputs, 5 nodes, identical=true |}]

let%expect_test "expand and clone of a D-carrying value" =
  let g =
    build "expand_clone"
      (let open Graph_builder in
       let* x = input ~shape:(s 1 1 1 3 2 2) () in
       let* w = input ~shape:(s 1 1 1 12 2 2) () in
       let* w = reshape { Reshape.Reshape.shape = s 1 1 4 3 2 2 } w in
       let* e = expand { Pointwise.Expand.size = s 1 1 4 3 2 2 } x in
       let* c = clone e in
       let* m = mul c w in
       let* out = sum sum_d m in
       Graph_builder.return [ out ])
    |> ok
  in
  compare_graphs "expand" ~x_shapes:[ s 1 1 1 3 2 2; s 1 1 1 12 2 2 ] g;
  [%expect {| expand: 1 outputs, 4 nodes, identical=true |}]

(* Reductions: a reduced D, or a kept one on the way to a later reduction. *)
let%expect_test "amax, mean and sum over D-carrying values" =
  let g =
    build "reductions"
      (let open Graph_builder in
       let* x = input ~shape:(s 1 1 1 4 3 6) () in
       let* r = reshape { Reshape.Reshape.shape = onto_d } x in
       let* hi = amax { Reduce.Amax.dims = [ H ]; keepdim = true } r in
       let* mean_h = mean { Reduce.Mean.dims = [ H; C ]; keepdim = true } r in
       let* both = sub hi mean_h in
       let* pooled = mean { Reduce.Mean.dims = [ D ]; keepdim = false } both in
       let* top = amax { Reduce.Amax.dims = [ D; H ]; keepdim = false } r in
       let* total = sum { Reduce.Sum.dims = [ D ]; keepdim = true } r in
       Graph_builder.return [ pooled; top; total ])
    |> ok
  in
  compare_graphs "reductions" ~x_shapes:[ s 1 1 1 4 3 6 ] g;
  [%expect {| reductions: 3 outputs, 7 nodes, identical=true |}]

(* Refusals: a group that meets anything whose meaning does not survive the
   move is left alone, and the ordinary path names the blocker. *)
let%expect_test "refused groups" =
  let refuse name build_group =
    compare_graphs name
      ~x_shapes:[ s 1 1 1 4 3 6 ]
      (build name build_group |> ok)
  in
  (* keepdim=false over H shifts the D extent onto a different axis. *)
  refuse "mean over H, keepdim false"
    (let open Graph_builder in
     let* x = input ~shape:(s 1 1 1 4 3 6) () in
     let* r = reshape { Reshape.Reshape.shape = onto_d } x in
     let* m = mean { Reduce.Mean.dims = [ H ]; keepdim = false } r in
     let* out = sum sum_d m in
     Graph_builder.return [ out ]);
  (* Sdpa's batch is D alone, not N. *)
  refuse "sdpa"
    (let open Graph_builder in
     let* x = input ~shape:(s 1 1 1 4 3 6) () in
     let* r = reshape { Reshape.Reshape.shape = onto_d } x in
     let* a =
       sdpa { Attention.Sdpa.scale = Default } ~query:r ~key:r ~value:r ()
     in
     let* out = sum sum_d a in
     Graph_builder.return [ out ]);
  (* The extent leaves the group on a graph output. *)
  refuse "output carries D"
    (let open Graph_builder in
     let* x = input ~shape:(s 1 1 1 4 3 6) () in
     let* r = reshape { Reshape.Reshape.shape = onto_d } x in
     let* p = permute (perm_of [ (D, W); (W, D) ]) r in
     Graph_builder.return [ p ]);
  [%expect
    {|
    mean over H, keepdim false: node n2: axis D is outside the N/H/W/C dialect
    sdpa: node n1: scaled-dot-product attention has extent > 1 on D, its batch axis, which the N/H/W/C dialect has no name for
    output carries D: node n1: axis D is outside the N/H/W/C dialect |}]

let%expect_test "the widened groups' maps verify" =
  Lower_region_test.verified_graph "permute" (permuted_group [ (D, H); (H, D) ]);
  Lower_region_test.verified_graph "outlook" matmul_graph;
  [%expect
    {|
    permute: 8 clusters: 1 proved (structural) [sampled 8], 1 unproved (over max_rounds) [sampled 8], 6 vacuous
    outlook: 11 clusters: 2 proved (structural) [sampled 8], 1 unproved (over max_rounds) [sampled 8], 8 vacuous |}]
