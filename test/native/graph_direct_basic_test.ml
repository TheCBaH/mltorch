(* Split out of what was graph_test.ml see graph_direct_fixtures.ml. *)

open Graph_ir
open Graph_direct_fixtures

let%expect_test "Direct graph: sequence add -> relu (with intermediate)" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"seq" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 4) ~name:"a" () in
          let* b = input ~shape:(s1c 4) ~name:"b" () in
          let* t = add ~name:"sum" a b in
          relu ~name:"out" t)
    in
    let a =
      Tensor.materialize (s1c 4) (fun c -> [| 1.; -5.; 2.; -8. |].(chan c))
    in
    let b =
      Tensor.materialize (s1c 4) (fun c -> [| -3.; 1.; 2.; 3. |].(chan c))
    in
    let* env =
      lift_eval
        (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a; b ]))
    in
    let* sum = tensor_of_name g env "sum" in
    let* out = tensor_of_name g env "out" in
    Err.return (sum, out)
  in
  Format.printf "%a@." (pp_result (pp_named_tensor_pair "sum" "out")) result;
  [%expect
    {|
    sum = tensor f32 [C=4] {-2, -4, 4, -5}
    out = tensor f32 [C=4] {0, 0, 4, 0} |}]

let%expect_test "Direct graph: captured constant is bound by tensor id" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"captured" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s1c 3) ~name:"x" () in
          let* weight = constant ~shape:(s1c 3) ~name:"weight" () in
          add ~name:"out" x weight)
    in
    let x = Tensor.materialize (s1c 3) (fun c -> float_of_int (chan c + 1)) in
    let weight = Tensor.materialize (s1c 3) (fun _ -> 10.) in
    let* env =
      lift_eval
        (Eval_direct.run g
           ~inputs:[ (List.hd g.Graph.inputs, x) ]
           ~constants:[ (List.nth g.Graph.inputs 1, weight) ])
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result Tensor.pp) result;
  [%expect {| tensor f32 [C=3] {11, 12, 13} |}]

let pp_id_consistency ppf g =
  Tensor_id.Map.iter
    (fun tid sg ->
      Format.fprintf ppf "tid=%a sig_id=%a match=%b@." Tensor_id.pp tid
        Tensor_id.pp sg.Tensor_sig.id
        (Tensor_id.equal tid sg.Tensor_sig.id))
    g.Graph.tensors

let%expect_test "Builder: Tensor_sig.id equals Tensor_id for all edges" =
  let result =
    lift_build
      Graph_builder.(
        build ~name:"check" ~outputs:(fun r -> [ r ])
        @@
        let* a = input ~shape:(s1c 2) ~name:"a" () in
        let* b = input ~shape:(s1c 2) ~name:"b" () in
        add ~name:"out" a b)
  in
  Format.printf "%a@." (pp_result pp_id_consistency) result;
  [%expect
    {|
    tid=t0 sig_id=t0 match=true
    tid=t1 sig_id=t1 match=true
    tid=t2 sig_id=t2 match=true |}]

(* Building the same graph twice must produce identical tensor ids: the id
   generator is local to [build], not a global counter. *)
let pp_ids ids =
  ids
  |> List.map (fun (t, s) ->
      Format.asprintf "(%a,%a)" Tensor_id.pp t Tensor_id.pp s)
  |> String.concat " "

let pp_deterministic_ids ppf (ids1, ids2, matches) =
  Format.fprintf ppf "g1 ids: %s@.g2 ids: %s@.match: %b" (pp_ids ids1)
    (pp_ids ids2) matches

let%expect_test "Builder: ids are deterministic across multiple build calls" =
  let build_add () =
    Graph_builder.(
      build ~name:"add" ~outputs:(fun r -> [ r ])
      @@
      let* a = input ~shape:(s1c 3) ~name:"a" () in
      let* b = input ~shape:(s1c 3) ~name:"b" () in
      add ~name:"out" a b)
  in
  let result =
    let open Err.Syntax in
    let* g1 = lift_build (build_add ()) in
    let* g2 = lift_build (build_add ()) in
    let ids_of g =
      Tensor_id.Map.bindings g.Graph.tensors
      |> List.map (fun (tid, sg) -> (tid, sg.Tensor_sig.id))
    in
    Err.return (ids_of g1, ids_of g2, ids_of g1 = ids_of g2)
  in
  Format.printf "%a@." (pp_result pp_deterministic_ids) result;
  [%expect
    {|
    g1 ids: (t0,t0) (t1,t1) (t2,t2)
    g2 ids: (t0,t0) (t1,t1) (t2,t2)
    match: true |}]

(* A node whose output ARITY comes from its input signature rather than from the
   op: three outputs here only because W's extent is 3. Every one is evaluated,
   so the variable-arity path through [Graph_shape] -> builder -> [Eval_direct]'s
   per-ordinal loop is exercised end to end rather than at output 0. *)
