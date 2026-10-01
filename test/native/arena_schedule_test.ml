(* [Arena_schedule]'s contract: the identity result, and the structural
   comparison that proves a reorder changed only [Graph.nodes]. *)

open Graph_ir

let fixture name = (List.assoc name Graph_fixtures.all) ()

let show r =
  match r with
  | Ok () -> Fmt.pr "ok@."
  | Error e -> Fmt.pr "%a@." Arena_schedule.pp_error (Err.Error.kind e)

let with_nodes (g : graph) nodes = { g with Graph.nodes }

let%expect_test "identity validates every fixture and keeps the graph" =
  List.iter
    (fun (name, build) ->
      let g = build () in
      match Arena_schedule.identity g with
      | Ok r ->
          assert (r.Arena_schedule.Result.graph == g);
          assert (Arena_schedule.same_structure g r.graph)
      | Error e ->
          Fmt.pr "%s: %a@." name Arena_schedule.pp_error (Err.Error.kind e))
    Graph_fixtures.all;
  [%expect {||}]

let%expect_test "permutation checks" =
  let g = fixture "diamond" in
  let relu, mul, add =
    match g.Graph.nodes with
    | [ a; b; c ] -> (a, b, c)
    | _ -> failwith "diamond shape"
  in
  let check nodes =
    show (Arena_schedule.check_permutation ~original:g (with_nodes g nodes))
  in
  check [ relu; mul; add ];
  check [ mul; relu; add ];
  check [ add; relu; mul ];
  check [ relu; mul ];
  check [ relu; mul; add; add ];
  [%expect
    {|
    ok
    ok
    node n2 reads an edge defined after it
    scheduling changed more than the order of the nodes
    scheduling changed more than the order of the nodes
    |}]

let%expect_test "a changed op or signature is not a reorder" =
  let g = fixture "diamond" in
  let g' =
    { g with Graph.outputs = List.rev g.Graph.outputs @ g.Graph.outputs }
  in
  show (Arena_schedule.check_permutation ~original:g g');
  [%expect {| scheduling changed more than the order of the nodes |}]

let%expect_test "limits" =
  let open Arena_schedule in
  let bytes = Core.Storage_units.Byte_size.zero in
  let try_ ~width ~expansions =
    match Limits.make ~width ~expansions ~state_bytes:bytes with
    | Ok l -> Fmt.pr "ok %b@." (Limits.equal l Limits.constructive_only)
    | Error e -> Fmt.pr "%a@." pp_error (Err.Error.kind e)
  in
  try_ ~width:1 ~expansions:0;
  try_ ~width:0 ~expansions:0;
  try_ ~width:4 ~expansions:(-1);
  try_ ~width:4 ~expansions:10;
  [%expect
    {|
    ok true
    invalid scheduler limit: width
    invalid scheduler limit: expansions
    ok false
    |}]
