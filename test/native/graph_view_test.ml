(* Exercises the framework's trust boundary: what [Graph_view.of_graph] rejects,
   what the index answers once a graph is trusted, and the boundary [Region]
   derives from a claimed node set. See .ai/native_transform_design.md §11. *)

open Graph_ir

type error = [ Graph_view.error | Region.error ]

let pp_error ppf : [< error ] -> unit = function
  | #Graph_view.error as e -> Graph_view.pp_error ppf e
  | #Region.error as e -> Region.pp_error ppf e

let pp_result pp_ok = Core.Pretty.core_result ~ok:pp_ok ~error:pp_error
let lift_view r = (r :> (Graph_view.t, error) Core.result)
let lift_region r = (r :> (Region.t, error) Core.result)
let view_of g = Graph_view.of_graph g |> Result.get_ok
let ids pp_id fmt l = Fmt.brackets (Fmt.list ~sep:Fmt.comma pp_id) fmt l
let n_ n = Node_id.of_int n
let t_ n = Tensor_id.of_int n

(* ---- what validation rejects --------------------------------------------- *)

(* Each case takes a well-formed fixture and breaks exactly one invariant, so
   the error names the invariant rather than a cascade. *)
let broken name f =
  let g = f (Graph_fixtures.diamond ()) in
  Format.printf "%s: %a@." name
    (pp_result (fun fmt _ -> Fmt.string fmt "accepted"))
    (lift_view (Graph_view.of_graph g))

let%expect_test "validation: a well-formed fixture is accepted" =
  List.iter
    (fun (name, build) ->
      let g = build () in
      Format.printf "%s: %a@." name
        (pp_result (fun fmt _ -> Fmt.string fmt "ok"))
        (lift_view (Graph_view.of_graph g)))
    Graph_fixtures.all;
  [%expect
    {|
    bypass_permute_fanout: ok
    bypass_permute_mixed_compatibility: ok
    bypass_permute_output: ok
    bypass_permute_pair: ok
    bypass_permute_shared: ok
    bypass_unlocks_sink: ok
    chain: ok
    const_arith: ok
    const_permute: ok
    const_pool: ok
    diamond: ok
    grouped: ok
    multi_output: ok
    permute_identity_chain: ok
    permute_noop: ok
    permute_pair: ok
    permute_partial_cancel: ok
    permute_sequence: ok
    permute_shared: ok
    reshape_flatten: ok
    reshape_relabel: ok
    residual: ok
    reuse_permute_backtrack_candidate: ok
    reuse_permute_basic: ok
    reuse_permute_competing_matches: ok
    reuse_permute_div_order: ok
    reuse_permute_missing_alternate: ok
    reuse_permute_self_inverse: ok
    reuse_permute_sub_order: ok
    reuse_permute_wide_fanout: ok
    reuse_permute_wrong_alternate: ok
    sink_permute_allowlist: ok
    sink_permute_binary: ok
    sink_permute_broadcast: ok
    sink_permute_fuse: ok
    sink_permute_mean_basic: ok
    sink_permute_mean_cycle: ok
    sink_permute_mean_not_keepdim: ok
    sink_permute_mean_shared: ok
    sink_permute_mismatch: ok
    sink_permute_output: ok
    sink_permute_shared: ok
    sink_permute_unary: ok |}]

let%expect_test
    "validation: an operand with no definition is a dangling operand" =
  (* The distinction the whole matcher rests on: [def = None] must mean "graph
     input", so an unknown operand has to be rejected here rather than silently
     read as one. *)
  broken "unknown operand" (fun g ->
      let bad =
        List.map
          (fun (n : node) ->
            match n.Node.op with
            | Relu _ -> { n with Node.op = Relu { x = t_ 99 } }
            | _ -> n)
          g.Graph.nodes
      in
      { g with Graph.nodes = bad });
  [%expect
    {| unknown operand: operand t99 has no definition and is not an input |}]

let%expect_test "validation: a node missing from the group tree is rejected" =
  broken "ungrouped node" (fun g ->
      let root = g.Graph.root in
      {
        g with
        Graph.root = { root with Group.items = List.tl root.Group.items };
      });
  [%expect {| ungrouped node: node n0 is in no group |}]

let%expect_test "validation: a node owned by two groups is rejected" =
  broken "duplicate group item" (fun g ->
      let root = g.Graph.root in
      {
        g with
        Graph.root =
          { root with Group.items = root.Group.items @ [ Group.Node (n_ 0) ] };
      });
  [%expect {| duplicate group item: node n0 is owned by more than one group |}]

let%expect_test "validation: out-of-order nodes are rejected" =
  broken "not topological" (fun g ->
      { g with Graph.nodes = List.rev g.Graph.nodes });
  [%expect {| not topological: node n2 reads an edge defined after it |}]

let%expect_test
    "validation: an input_kinds key that is not an input is rejected" =
  (* Totality is NOT required — the map is sparse by design — but a key that
     names a non-input is a real inconsistency. *)
  broken "stray input kind" (fun g ->
      {
        g with
        Graph.input_kinds =
          Tensor_id.Map.add (t_ 3) Input.Constant g.Graph.input_kinds;
      });
  [%expect
    {| stray input kind: input_kinds names t3, which is not a graph input |}]

let%expect_test "validation: a signature filed under the wrong key is rejected"
    =
  broken "sig key mismatch" (fun g ->
      let sg = Tensor_id.Map.find (t_ 0) g.Graph.tensors in
      { g with Graph.tensors = Tensor_id.Map.add (t_ 1) sg g.Graph.tensors });
  [%expect {| sig key mismatch: tensor map key t1 holds a signature for t0 |}]

let%expect_test "validation: a wrong output arity is rejected" =
  (* Shape inference is what knows an op's real arity, so a node claiming an
     extra edge is caught here rather than surfacing later as a rewrite that
     rewires an output nothing produces. (Dropping an edge instead would trip
     the operand check first, since its consumer would dangle.) *)
  broken "output arity" (fun g ->
      let bad =
        List.map
          (fun (n : node) ->
            match n.Node.op with
            | Relu _ -> { n with Node.outputs = n.Node.outputs @ [ t_ 50 ] }
            | _ -> n)
          g.Graph.nodes
      in
      { g with Graph.nodes = bad });
  [%expect {| output arity: node n0 declares 2 outputs but its op has 1 |}]

(* ---- the index ----------------------------------------------------------- *)

let%expect_test "index: def, uses, and the input/dangling distinction" =
  let g = Graph_fixtures.diamond () in
  let v = view_of g in
  let describe id =
    Format.printf "@[<h>%a: def=%a uses=%a output=%b@]@." Tensor_id.pp id
      (Fmt.option ~none:(Fmt.any "input") Node_id.pp)
      (Option.map (fun (n : node) -> n.Node.id) (Graph_view.def v id))
      (ids Node_id.pp)
      (List.map (fun (n : node) -> n.Node.id) (Graph_view.uses v id))
      (Graph_view.is_graph_output v id)
  in
  List.iter describe [ t_ 0; t_ 1; t_ 2; t_ 3 ];
  [%expect
    {|
    t0: def=input uses=[n0, n1] output=false
    t1: def=n0 uses=[n2] output=false
    t2: def=n1 uses=[n2] output=false
    t3: def=n2 uses=[] output=true |}]

let%expect_test "index: constants are read from the graph, not guessed" =
  let g = Graph_fixtures.const_permute () in
  let v = view_of g in
  List.iter
    (fun id ->
      Format.printf "%a constant=%b@." Tensor_id.pp id
        (Graph_view.is_constant v id))
    g.Graph.inputs;
  [%expect {|
    t0 constant=false
    t1 constant=true |}]

let%expect_test "index: group ownership and the nearest common ancestor" =
  (* The two permutes sit in sibling groups, so a rewrite spanning them belongs
     in the root; each alone belongs in its own group. *)
  let g = Graph_fixtures.grouped () in
  let v = view_of g in
  Format.printf "%a@." Graph_ir.pp g;
  List.iter
    (fun n ->
      Format.printf "%a in %a@." Node_id.pp n Group_id.pp
        (Graph_view.group_of v n))
    [ n_ 0; n_ 1; n_ 2 ];
  Format.printf "common(n0,n1) = %a@." Group_id.pp
    (Graph_view.common_group v [ n_ 0; n_ 1 ]);
  Format.printf "common(n0) = %a@." Group_id.pp
    (Graph_view.common_group v [ n_ 0 ]);
  [%expect
    {|
    graph
    inputs: [t0 f32 [H=2 W=3 C=4]]
    nodes:
      group g1 first:
        n0: [t1 f32 [H=3 W=4 C=2]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
      group g2 second:
        n1: [t2 f32 [H=2 W=3 C=4]] = permute x=t1 perm=[H<-C, W<-H, C<-W]
      n2: [t3 f32 [H=2 W=3 C=4]] = relu x=t2
    outputs: [t3 f32 [H=2 W=3 C=4]]
    n0 in g1
    n1 in g2
    n2 in g0
    common(n0,n1) = g0
    common(n0) = g1 |}]

let%expect_test "topo_sort is stable and detects a cycle" =
  let g = Graph_fixtures.residual () in
  let sorted = Graph_view.topo_sort (List.rev g.Graph.nodes) |> Result.get_ok in
  Format.printf "reversed then sorted: %a@." (ids Node_id.pp)
    (List.map (fun (n : node) -> n.Node.id) sorted);
  (* A node reading its own output is the smallest possible cycle. *)
  let self : node =
    { Node.id = n_ 9; op = Relu { x = t_ 9 }; outputs = [ t_ 9 ] }
  in
  (match Graph_view.topo_sort [ self ] with
  | Ok _ -> Format.printf "cycle: accepted@."
  | Error e -> (
      match e.Core.Error.kind with
      | `Cycle id -> Format.printf "cycle at %a@." Node_id.pp id));
  [%expect {|
    reversed then sorted: [n0, n1, n2]
    cycle at n9 |}]

(* ---- regions ------------------------------------------------------------- *)

let pp_region_and_match v fmt r =
  Format.fprintf fmt "%a@.@[<v>extracted:@,%a@]" Region.pp r Graph_ir.pp
    (Region.extract v r)

let region v nodes = lift_region (Region.of_nodes v (Node_id.Set.of_list nodes))

let%expect_test "region: the boundary is derived, not declared" =
  (* Claiming just the conv leaves its weight and bias as boundary inputs and
     its result as the single output, without the pattern saying so. *)
  let g = Graph_fixtures.chain () in
  let v = view_of g in
  Format.printf "%a@." (pp_result (pp_region_and_match v)) (region v [ n_ 0 ]);
  [%expect
    {|
    nodes: [n0]
    inputs: [t0, t1, t2]
    outputs: [t7]
    interior: []
    convex: true
    extracted:
    graph
    inputs:
      [t0 f32 [H=4 W=4 C=2], t1 f32 [N=3 T=1 D=1 H=2 W=2 C=2] constant,
       t2 f32 [C=3] constant]
    nodes:
      n0: [t7 f32 [H=3 W=3 C=3]] =
        conv2d
          x=t0
          weight=t1
          bias=t2
          params={h={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=2;
                 groups=1}
    outputs: [t7 f32 [H=3 W=3 C=3]]
    |}]

let%expect_test "region: an interior edge is one used only inside" =
  let g = Graph_fixtures.chain () in
  let v = view_of g in
  Format.printf "%a@." (pp_result Region.pp) (region v [ n_ 0; n_ 1 ]);
  [%expect
    {|
    nodes: [n0, n1]
    inputs: [t0, t1, t2, t3, t4, t5, t6]
    outputs: [t8]
    interior: [t7]
    convex: true |}]

let%expect_test "region: a graph output escapes even with no consumer" =
  let g = Graph_fixtures.diamond () in
  let v = view_of g in
  Format.printf "%a@." (pp_result Region.pp) (region v [ n_ 2 ]);
  [%expect
    {|
    nodes: [n2]
    inputs: [t1, t2]
    outputs: [t3]
    interior: []
    convex: true |}]

let%expect_test "region: convexity is reported, not enforced" =
  (* residual is x -> relu(n0) -> relu(n1) -> add(n2). Claiming {n0,n2} leaves a
     path n0 -> n1 -> n2 that exits the region and comes back: collapsing that to
     one node would be a cycle, so the fusing constructors refuse it while other
     rewrites may proceed. Claiming {n0,n1} instead is contiguous.

     Note that merely skipping a node is not enough to be non-convex — in the
     diamond, {relu, add} skips the mul but the mul reads the graph input rather
     than the relu, so no path leaves and returns. *)
  let g = Graph_fixtures.residual () in
  let v = view_of g in
  Format.printf "contiguous:@.%a@." (pp_result Region.pp)
    (region v [ n_ 0; n_ 1 ]);
  Format.printf "leaves and re-enters:@.%a@." (pp_result Region.pp)
    (region v [ n_ 0; n_ 2 ]);
  let d = Graph_fixtures.diamond () in
  let dv = view_of d in
  Format.printf "skips a node but stays convex:@.%a@." (pp_result Region.pp)
    (region dv [ n_ 0; n_ 2 ]);
  [%expect
    {|
    contiguous:
    nodes: [n0, n1]
    inputs: [t0]
    outputs: [t2]
    interior: [t1]
    convex: true
    leaves and re-enters:
    nodes: [n0, n2]
    inputs: [t0, t2]
    outputs: [t1, t3]
    interior: []
    convex: false
    skips a node but stays convex:
    nodes: [n0, n2]
    inputs: [t0, t2]
    outputs: [t3]
    interior: [t1]
    convex: true |}]

let%expect_test "region: unknown and empty claims are rejected" =
  let g = Graph_fixtures.diamond () in
  let v = view_of g in
  Format.printf "%a@." (pp_result Region.pp) (region v [ n_ 42 ]);
  Format.printf "%a@." (pp_result Region.pp) (region v []);
  [%expect {|
    region claims unknown node n42
    region claims no nodes |}]
