(* `index.Tensor` at the [Graph_ir]/[Eval_direct] level: a real graph node,
   JSON round trip, the out-of-range-index fixture proving [Eval_direct.run]
   RAISES [Err.Exn.E] (round 8) rather than returning [Error] or reading
   silently out of bounds, and the round-12 [Graph_ir]-level rejection of a
   non-unit leading index axis (reachable from a direct [Graph_builder] call,
   not only from importer validation). *)

open Graph_ir
open Graph_direct_fixtures

let build_graph ~index_values =
  Graph_builder.(
    build ~name:"index_tensor" ~outputs:(fun r -> [ r ])
    @@
    let* self = input ~shape:(s 1 1 1 2 3 2) ~name:"self" () in
    let* index =
      constant
        ~shape:(s 1 1 1 1 1 (List.length index_values))
        ~fmt:(Payload.Fmt Payload.I64) ~name:"index" ()
    in
    index_tensor ~name:"out"
      { Index_tensor.Index_tensor.axis = Axis.W }
      ~self ~index)

let self_tensor =
  Tensor.materialize (s 1 1 1 2 3 2) (fun c ->
      float_of_int
        ((100 * Dim.to_int (Vec6.get c Axis.H))
        + (10 * Dim.to_int (Vec6.get c Axis.W))
        + Dim.to_int (Vec6.get c Axis.C)))

let index_tensor_of values =
  Tensor.materialize_i64
    (s 1 1 1 1 1 (List.length values))
    (fun c -> List.nth values (Dim.to_int (Vec6.get c Axis.C)))

let self_and_index_ids (g : graph) =
  match g.Graph.inputs with
  | [ self_id; index_id ] -> (self_id, index_id)
  | _ -> assert false

let%expect_test
    "Direct graph: Index_tensor gathers self by index's runtime value" =
  let result =
    let open Err.Syntax in
    let* g = lift_build (build_graph ~index_values:[ 2L; -3L ]) in
    Format.printf "%a@." Graph_ir.pp g;
    let self_id, index_id = self_and_index_ids g in
    let index = index_tensor_of [ 2L; -3L ] in
    let* env =
      lift_eval
        (Eval_direct.run g
           ~inputs:[ (self_id, self_tensor) ]
           ~constants:[ (index_id, index) ])
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect
    {|
    graph
    inputs: [t0 f32 [H=2 W=3 C=2] ->[n0], t1 i64 [C=2] ->[n0] constant]
    nodes:
      n0: [t2 f32 [H=2 W=2 C=2]] = index_tensor self=t0 index=t1 params={axis=W}
    outputs: [t2 f32 [H=2 W=2 C=2] <-n0]
    out = tensor f32 [H=2 W=2 C=2] {20, 21, 0, 1, 120, 121, 100, 101}
    |}]

let%expect_test "Direct graph: Index_tensor JSON round-trips" =
  let g =
    match lift_build (build_graph ~index_values:[ 2L; -3L ]) with
    | Ok g -> g
    | Error _ -> assert false
  in
  let json =
    match Graph_json.encode_graph ~format:Jsont.Indent g with
    | Ok j -> j
    | Error _ -> assert false
  in
  let g2 =
    match Graph_json.decode_graph json with
    | Ok g2 -> g2
    | Error e -> Err.or_raise ~pp_error:Graph_json.pp_error (Error e)
  in
  let printed g = Format.asprintf "%a" Graph_ir.pp g in
  Format.printf "round-trips identically: %b@."
    (String.equal (printed g) (printed g2));
  [%expect {| round-trips identically: true |}]

(* Round 8: a data-dependent OOB gather value is discovered per-pixel, deep
   inside [Direct]'s evaluation -- not at graph-construction time, and not
   through the returned [Err.t] the way a shape mismatch is. *)
let%expect_test
    "Direct graph: an out-of-range gather value raises Err.Exn.E, not a silent \
     OOB read and not a returned Error" =
  let g =
    match lift_build (build_graph ~index_values:[ 5L ]) with
    | Ok g -> g
    | Error _ -> assert false
  in
  let self_id, index_id = self_and_index_ids g in
  let index = index_tensor_of [ 5L ] in
  (try
     ignore
       (Eval_direct.run g
          ~inputs:[ (self_id, self_tensor) ]
          ~constants:[ (index_id, index) ]);
     print_string "no exception"
   with Err.Exn.E e -> Format.printf "raised: %a@." Err.Exn.pp_kind e);
  [%expect {| raised: gather index 5 out of range [-3, 2] |}]

(* Round 12: [output_shape] itself rejects a non-unit leading index axis,
   reachable from a direct [Graph_builder] call (and, since a decoded JSON
   graph's shapes are recomputed by the very same [output_shape] rather than
   trusted from the wire, from a JSON-decoded graph too) -- not only from
   importer validation, which cannot produce this shape in the first place. *)
let%expect_test
    "Direct graph: a non-unit leading index axis is a typed rejection, not \
     importer-only" =
  let bad_index_shape =
    s 1 1 1 2 1 3
    (* H=2, C=3: H is non-unit *)
  in
  let result =
    Index_tensor.Index_tensor.output_shape ~self_shape:(s 1 1 1 2 3 2)
      ~index_shape:bad_index_shape
      { Index_tensor.Index_tensor.axis = Axis.W }
  in
  Format.printf "%a@."
    (Core.Pretty.err_result ~ok:Vec6.pp_shape ~error:Shape_error.pp)
    result;
  [%expect
    {| index.Tensor: index axis H must have extent 1 (only axis C carries real data), got 2 |}]
