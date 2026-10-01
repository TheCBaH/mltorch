(* Coverage for the comparison-artifact serializer used by the canonical
   transform benchmark:
   an intentional change to an op, a constant literal value, and a map claim
   must each show up in the printed artifact. [Const_ssa.pp]/[Constant_store.pp]
   abbreviate a literal to the bare word "literal" (const_ssa.ml's [pp_leaf]),
   which is exactly the kind of change this guards against — the first two
   tests would fail to distinguish their variants if
   [Native_transform_bench_artifact] fell back to those printers instead of
   its own exact ones. The third confirms the map section (reused,
   unmodified [Graph_map.pp]) reflects a claim change too, so the whole
   artifact — not just the constant plan — has been audited. *)

let shape = Graph_fixtures.s 1 1 1 1 1 1
let f32 = Payload.Fmt Payload.F32
let tid = Graph_ir.Tensor_id.of_int
let sg id = Tensor_sig.create ~id:(tid id) ~name:"" ~shape ~fmt:f32 ()
let literal v = Tensor.materialize_fmt f32 shape (fun _ -> v)

let single_op g =
  match Graph_ir.nodes g with
  | [ (node : Graph_ir.node) ] -> node.Graph_ir.Node.op
  | _ -> failwith "expected exactly one node"

let%expect_test "constant literal value changes are not abbreviated away" =
  let plan_with v =
    let store =
      Constant_store.bind_literal Constant_store.empty ~tensor:(sg 0)
        (literal v)
      |> Result.get_ok
    in
    Fmt.str "%a" Native_transform_bench_artifact.pp_constant_store store
  in
  let a = plan_with 1.0 and b = plan_with 2.0 in
  Fmt.pr "differ: %b@." (not (String.equal a b));
  [%expect {| differ: true |}]

let%expect_test "an op change inside a Const-SSA Apply is not abbreviated away"
    =
  let one_node_graph op_of =
    Graph_builder.(
      build ~name:"g"
        ~outputs:(fun o -> [ o ])
        (let* x = input ~shape () in
         let* y = input ~shape () in
         op_of x y))
    |> Err.or_raise ~pp_error:Graph_builder.pp_error
  in
  let def op_of =
    let op = single_op (one_node_graph op_of) in
    let definition = Const_ssa.Apply { op; output = sg 2 } in
    Fmt.str "%a" Native_transform_bench_artifact.pp_definition
      (Const_ssa.Value_id.of_tensor_id (tid 2), definition)
  in
  let a = def Graph_builder.add and b = def Graph_builder.mul in
  Fmt.pr "differ: %b@." (not (String.equal a b));
  [%expect {| differ: true |}]

let%expect_test "a map claim change is visible in the composed-map dump" =
  let module A =
    (val Version_fixture.of_graph (Verify_fixtures.relu_of Graph_builder.add ()))
  in
  let module B =
    (val Version_fixture.of_graph (Verify_fixtures.relu_of Graph_builder.add ()))
  in
  let src id =
    Option.get (Snapshot.edge A.snapshot (Graph_ir.Tensor_id.of_int id))
  in
  let dst id =
    Option.get (Snapshot.edge B.snapshot (Graph_ir.Tensor_id.of_int id))
  in
  let map_with claim =
    (* t3 is the graph output (relu's result): claiming it has no downstream
       consumer to leave unclosed, unlike an intermediate edge. *)
    Verify_fixtures.hand_map ~src:A.snapshot ~dst:B.snapshot
      [ Correspondence.pair (src 3) (dst 3) claim ]
      []
    |> Fmt.to_to_string Graph_map.pp
  in
  let identical = map_with Correspondence.Identical in
  let approximate =
    map_with
      (Correspondence.Approximate
         (Correspondence.Precision.Set.of_list
            [
              {
                Correspondence.Precision.fmt = Payload.Fmt Payload.BF16;
                quant = None;
              };
            ]))
  in
  Fmt.pr "differ: %b@." (not (String.equal identical approximate));
  [%expect {| differ: true |}]
