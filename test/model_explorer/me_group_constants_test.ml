(* [Me_group_constants]: constant-namespace grouping, as a pure transform over
   an already-built [Model_explorer.Graph.t]. No session, no limits, no
   exporter — the fixtures below build the graph shapes the transform cares
   about directly. *)

module ME = Model_explorer

let node ?(namespace = "") ?(incoming = []) id =
  ME.GraphNode.create ~id ~label:id ~namespace
    ~incomingEdges:
      (List.map
         (fun src ->
           ME.IncomingEdge.create ~sourceNodeId:src ~sourceNodeOutputId:"0"
             ~targetNodeInputId:"0" ())
         incoming)
    ()

let graph nodes = ME.Graph.create ~id:"g" ~nodes ()

let show (g : ME.Graph.t) =
  List.iter
    (fun (n : ME.GraphNode.t) ->
      Printf.printf "%s: %S\n" n.ME.GraphNode.id n.ME.GraphNode.namespace)
    g.ME.Graph.nodes

let%expect_test "is_constant_id: only the const: prefix qualifies" =
  List.iter
    (fun id ->
      Printf.printf "%s -> %b\n" id (Me_group_constants.is_constant_id id))
    [ "const:x"; "const:"; "in:x"; "out:x"; "n3"; "constant" ];
  [%expect
    {|
    const:x -> true
    const: -> true
    in:x -> false
    out:x -> false
    n3 -> false
    constant -> false |}]

let%expect_test "Explicit is the identity" =
  let g =
    graph
      [
        node "const:w" ~namespace:"";
        node "n0" ~namespace:"features" ~incoming:[ "const:w" ];
      ]
  in
  show (Me_group_constants.apply Me_group_constants.Explicit g);
  [%expect {|
    const:w: ""
    n0: "features" |}]

let%expect_test "a constant with no consumer falls back to parameters" =
  (* An unconsumed constant (a declared buffer no traced op reads) has no
     owner to inherit from; left at namespace [""] it would sit at root as a
     scattered singleton exactly like the ungrouped baseline, which on
     MobileNet's 52 unused `num_batches_tracked` buffers alone was enough to
     defeat the whole transform. *)
  let g = graph [ node "const:w" ~namespace:"" ] in
  show (Me_group_constants.apply Me_group_constants.Grouped g);
  [%expect {| const:w: "parameters" |}]

let%expect_test "a single consumer: the constant takes its namespace" =
  let g =
    graph
      [
        node "const:w" ~namespace:"";
        node "n0" ~namespace:"features/conv_stem" ~incoming:[ "const:w" ];
      ]
  in
  show (Me_group_constants.apply Me_group_constants.Grouped g);
  [%expect {|
    const:w: "features/conv_stem"
    n0: "features/conv_stem" |}]

let%expect_test "two consumers sharing a prefix: the longest common namespace" =
  let g =
    graph
      [
        node "const:w" ~namespace:"";
        node "n0" ~namespace:"blocks/0/conv" ~incoming:[ "const:w" ];
        node "n1" ~namespace:"blocks/0/bn" ~incoming:[ "const:w" ];
      ]
  in
  show (Me_group_constants.apply Me_group_constants.Grouped g);
  [%expect
    {|
    const:w: "blocks/0"
    n0: "blocks/0/conv"
    n1: "blocks/0/bn" |}]

let%expect_test "two consumers with nothing in common: the parameters group" =
  let g =
    graph
      [
        node "const:w" ~namespace:"";
        node "n0" ~namespace:"features" ~incoming:[ "const:w" ];
        node "n1" ~namespace:"classifier" ~incoming:[ "const:w" ];
      ]
  in
  show (Me_group_constants.apply Me_group_constants.Grouped g);
  [%expect
    {|
    const:w: "parameters"
    n0: "features"
    n1: "classifier" |}]

let%expect_test "a lone root-level consumer is not treated as a conflict" =
  (* One consumer, at the root namespace: there is only one owner, so this is
     the trivial single-consumer case, not the "no shared ownership" case that
     falls back to [parameters]. *)
  let g =
    graph
      [
        node "const:w" ~namespace:"";
        node "n0" ~namespace:"" ~incoming:[ "const:w" ];
      ]
  in
  show (Me_group_constants.apply Me_group_constants.Grouped g);
  [%expect {|
    const:w: ""
    n0: "" |}]

let%expect_test "a repeated consumer namespace does not out-vote a third" =
  let g =
    graph
      [
        node "const:w" ~namespace:"";
        node "n0" ~namespace:"blocks/0" ~incoming:[ "const:w" ];
        node "n1" ~namespace:"blocks/0" ~incoming:[ "const:w" ];
        node "n2" ~namespace:"blocks/1" ~incoming:[ "const:w" ];
      ]
  in
  show (Me_group_constants.apply Me_group_constants.Grouped g);
  [%expect
    {|
    const:w: "blocks"
    n0: "blocks/0"
    n1: "blocks/0"
    n2: "blocks/1" |}]

let%expect_test "a non-constant id is never reassigned" =
  let g =
    graph
      [
        node "in:x" ~namespace:"";
        node "n0" ~namespace:"features" ~incoming:[ "in:x" ];
      ]
  in
  show (Me_group_constants.apply Me_group_constants.Grouped g);
  [%expect {|
    in:x: ""
    n0: "features" |}]
