(* view/reshape and Aten_shape.resolve_view_size. Split from the former native_bridge_test.ml; promote with [dune promote test/native_bridge/reshape_test.ml]. *)

open Helpers

(* [view.default] and [_unsafe_view.default] share one dispatch arm
   (op3-impl.md commit 4, Part IV #2), so every case worth proving for one is
   worth proving for both -- [_unsafe_view] cannot be allowed to hide behind
   [view.default]'s coverage. *)
let view_targets =
  [ "torch.ops.aten.view.default"; "torch.ops.aten._unsafe_view.default" ]

let dispatch_view_both ~x ~size =
  List.iter
    (fun target ->
      dispatch_print ~target
        ~bindings:[ ("self", x) ]
        ~inputs:[ in_tensor "self"; in_ints "size" size ]
        ~noutputs:1)
    view_targets

(* op3-impl.md F1, the zero-guard half: [Aten_shape.resolve_view_size]
   validates every non-[-1] entry through [Dim.extent_checked] before any
   division runs, so [0] alongside [-1] is refused rather than dividing by the
   zero it would otherwise fold [known] to. Before commit 1 the bridge's own
   [resolve_view_size] had NO zero guard at all and raised [Division_by_zero]
   here -- unlike [Native_interp], which already special-cased the [-1] case
   (commit 9390ae6). *)
let%expect_test "dispatch: view.default rejects a target sizing both 0 and -1" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_view_both ~x ~size:[ 0; -1 ];
  [%expect
    {|
    error: extent must be >= 1, got 0
    error: extent must be >= 1, got 0 |}]

(* The oversized-TARGET case, not the source: [Aten_shape.resolve_view_size]'s
   divide-first fold rejects it (its [known] already exceeds the source's tiny
   [numel] partway through) before [Aten_shape.of_aten] ever converts a
   resolved list into a native shape, so no large allocation happens -- unlike
   an oversized SOURCE, which would have to reach the bridge through a real
   (unconstructible-as-a-fixture) ATen handle; see [Tensor_bridge.of_aten]'s
   own preflight, tested at the primitive level in test/native/vec6_test.ml. *)
let%expect_test "dispatch: view.default rejects a target past the numel ceiling"
    =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  (* numel 6 either way: a small, ordinary target still lowers, establishing
     that the rejection below is about the OVERSIZED case and not a general
     regression. *)
  dispatch_view_both ~x ~size:[ 2; 3 ];
  dispatch_view_both ~x ~size:[ 65536; 65536 ];
  [%expect
    {|
    tensor f32 [W=2 C=3] {0, 1, 2, 3, 4, 5}
    tensor f32 [W=2 C=3] {0, 1, 2, 3, 4, 5}
    error: view size [65536, 65536] does not match 6 elements
    error: view size [65536, 65536] does not match 6 elements |}]

(* Rank-changing targets, under both overloads: rank-increasing (a flat
   6-vector split into [2,3]) and rank-decreasing (a rank-3 volume flattened
   to [24]). Row-major values, so a wrong element order fails visibly rather
   than by shape alone. *)
let%expect_test
    "dispatch: view/_unsafe_view rank-increasing and rank-decreasing targets" =
  let flat = float_tensor [ 6 ] (List.init 6 float_of_int) in
  dispatch_view_both ~x:flat ~size:[ 2; 3 ];
  let vol = float_tensor [ 2; 3; 4 ] (List.init 24 float_of_int) in
  dispatch_view_both ~x:vol ~size:[ 24 ];
  [%expect
    {|
    tensor f32 [W=2 C=3] {0, 1, 2, 3, 4, 5}
    tensor f32 [W=2 C=3] {0, 1, 2, 3, 4, 5}
    tensor f32 [C=24] {0, 1, 2, 3, 4, 5, 6, 7, ...}
    tensor f32 [C=24] {0, 1, 2, 3, 4, 5, 6, 7, ...} |}]

(* One inferred [-1] in leading, middle and trailing position, on a source
   with no repeated extents (2,3,4) so a wrong resolved position is visible in
   the printed shape, not just the values. *)
let%expect_test
    "dispatch: view/_unsafe_view resolve a leading, middle and trailing -1" =
  let vol = float_tensor [ 2; 3; 4 ] (List.init 24 float_of_int) in
  dispatch_view_both ~x:vol ~size:[ -1; 2; 3 ];
  dispatch_view_both ~x:vol ~size:[ 3; -1; 2 ];
  dispatch_view_both ~x:vol ~size:[ 3; 2; -1 ];
  [%expect
    {|
    tensor f32 [H=4 W=2 C=3] {0, 1, 2, 3, 4, 5, 6, 7, ...}
    tensor f32 [H=4 W=2 C=3] {0, 1, 2, 3, 4, 5, 6, 7, ...}
    tensor f32 [H=3 W=4 C=2] {0, 1, 2, 3, 4, 5, 6, 7, ...}
    tensor f32 [H=3 W=4 C=2] {0, 1, 2, 3, 4, 5, 6, 7, ...}
    tensor f32 [H=3 W=2 C=4] {0, 1, 2, 3, 4, 5, 6, 7, ...}
    tensor f32 [H=3 W=2 C=4] {0, 1, 2, 3, 4, 5, 6, 7, ...} |}]

(* A rank-6 target is the boundary: exactly 6 axes is accepted (the frame's
   width), rank-7 is `Rank_out_of_range`. *)
let%expect_test "dispatch: view/_unsafe_view accepts rank 6, rejects rank 7" =
  let flat = float_tensor [ 6 ] (List.init 6 float_of_int) in
  dispatch_view_both ~x:flat ~size:[ 1; 1; 1; 1; 1; 6 ];
  dispatch_view_both ~x:flat ~size:[ 1; 1; 1; 1; 1; 1; 6 ];
  [%expect
    {|
    tensor f32 [C=6] {0, 1, 2, 3, 4, 5}
    tensor f32 [C=6] {0, 1, 2, 3, 4, 5}
    error: rank 7 out of [0, 6]
    error: rank 7 out of [0, 6] |}]

(* op3-impl.md F1's three accepted-then-wrong-answer holes, now three typed
   rejections instead: more than one inferred dim, a target whose declared
   product disagrees with the source's element count, and a target that does
   not evenly divide it. *)
let%expect_test "dispatch: view/_unsafe_view reject F1's three invalid targets"
    =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_view_both ~x ~size:[ -1; -1 ];
  dispatch_view_both ~x ~size:[ 4; 2 ];
  dispatch_view_both ~x ~size:[ 4; -1 ];
  [%expect
    {|
    error: view size [-1, -1] has more than one inferred (-1) dimension
    error: view size [-1, -1] has more than one inferred (-1) dimension
    error: view size [4, 2] does not match 6 elements
    error: view size [4, 2] does not match 6 elements
    error: view size [4, -1] does not divide 6 elements
    error: view size [4, -1] does not divide 6 elements |}]

(* A declared 0 with no [-1] present -- a distinct case from the
   [-1]-alongside-[0] one above, caught by [Dim.extent_checked] before the
   divisibility check that follows it even runs. *)
let%expect_test "dispatch: view/_unsafe_view reject a target with a zero extent"
    =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_view_both ~x ~size:[ 0; 6 ];
  [%expect
    {|
    error: extent must be >= 1, got 0
    error: extent must be >= 1, got 0 |}]

(* The rejected witness cannot be the PRODUCT [prefix * extent]: [Dim.extent]
   bounds an extent only BELOW, so on this (63-bit) backend a single axis can
   sit near [max_int], and [prefix * extent] here would itself overflow
   [int64] -- reintroducing exactly the wrap this design exists to prevent.
   Native-only on purpose (unlike every other numel fixture in this group):
   the constant is not representable in js_of_ocaml's 32-bit [int], and this
   file's stanza is the one inline suite with plain [(inline_tests)], no [js]
   mode (test/dune). *)
let%expect_test
    "vec6: numel_bounded reports the witness pair, never their \
     (unrepresentable) product" =
  let s = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1_073_741_824 ~c:max_int in
  (match Vec6.numel_bounded ~limit:Kernel.Limits.Hard.numel s with
  | Ok n -> Format.printf "Ok %Ld@." n
  | Error e -> (
      match Err.Error.kind e with
      | `Numel_over_limit b -> Format.printf "Error %a@." Vec6.Numel_bound.pp b));
  [%expect
    {| Error axis C: 1073741824 elements so far times extent 4611686018427387903 reaches the maximum of 2147483648 |}]

let%expect_test "PT2 provenance: native ids map to qualified source origins" =
  let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2 in
  let graph =
    match
      Graph_builder.(
        build ~name:"ignored" ~outputs:(fun r -> [ r ])
        @@
        let* x = input ~shape () in
        let* weight = constant ~shape () in
        add x weight)
    with
    | Ok graph -> graph
    | Error _ -> assert false
  in
  let weight = List.nth graph.Graph_ir.Graph.inputs 1 in
  let node = List.hd graph.Graph_ir.Graph.nodes in
  let tensor_origins =
    Graph_ir.Tensor_id.Map.singleton weight
      (Pt2_native_graph.Source
         {
           Pt2_native_graph.Tensor_origin.graph_path = [ 4; 2 ];
           ssa_name = "p_layer_weight";
           meta = None;
         })
  in
  let node_origins =
    Graph_ir.Node_id.Map.singleton node.Graph_ir.Node.id
      [
        {
          Pt2_native_graph.Node_origin.graph_path =
            Pt2_native_graph.Graph_path.root;
          index = 7;
          target = "torch.ops.aten.add.Tensor";
          name = Some "add";
          metadata = Schema_runtime.String_map.empty;
        };
      ]
  in
  match
    Pt2_native_graph.make ~graph ~tensor_origins ~node_origins
      ~captured_targets:(Graph_ir.Tensor_id.Map.singleton weight "layer.weight")
  with
  | Error _ -> print_endline "unexpected error"
  | Ok provenance ->
      let origin =
        Graph_ir.Tensor_id.Map.find weight provenance.tensor_origins
      in
      (match origin with
      | Pt2_native_graph.Source { graph_path; ssa_name; _ } ->
          Format.printf "tensor=t%d path=%a name=%s target=%s@."
            (Graph_ir.Tensor_id.to_int weight)
            Pt2_native_graph.Graph_path.pp graph_path ssa_name
            (Graph_ir.Tensor_id.Map.find weight provenance.captured_targets)
      | Derived -> assert false);
      [%expect
        {| tensor=t1 path=root/4/2 name=p_layer_weight target=layer.weight |}]
