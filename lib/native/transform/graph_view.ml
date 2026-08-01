(* See graph_view.mli. Everything is indexed once at [of_graph]; the accessors
   are plain map lookups so a matcher can query freely.

   Parameterised over the dialect so a second one gets validation for free. The
   Native specialization at the bottom keeps every existing caller unchanged. *)

open Graph_common

(* Outside the functor: neither mentions the dialect, and a per-instantiation
   copy would make [Make (A).arity] and [Make (B).arity] incompatible for no
   reason. *)
type arity = { node : Node_id.t; expected : int; actual : int }
type sig_key = { key : Tensor_id.t; recorded : Tensor_id.t }

module Make (D : Dialect.S) = struct
  type node = D.op Node.t
  type graph = D.op Graph.t

  (* WRAPPED, not inherited: [D.shape_error] is abstract, and OCaml can inherit
     only a type it knows expands to a polymorphic-variant row. The tag is
     [`Graph_shape] rather than [`Shape] because a dialect's own shape row —
     Native4D's [Shape4.error] — is inherited directly elsewhere, and two
     same-named tags with different payloads cannot union. *)
  type error =
    [ `Graph_shape of D.shape_error
    | `Invalid_sig of Tensor_id.t * D.shape_error
    | `Duplicate_group_id of Group_id.t
    | `Duplicate_group_item of Node_id.t
    | `Duplicate_node_id of Node_id.t
    | `Duplicate_tensor_def of Tensor_id.t
    | `Input_defined_by_node of Tensor_id.t
    | `Node_not_grouped of Node_id.t
    | `Not_topological of Node_id.t
    | `Output_arity of arity
    | `Output_shape_mismatch of Tensor_id.t
    | `Sig_key_mismatch of sig_key
    | `Unknown_group_item of Node_id.t
    | `Unknown_input_kind of Tensor_id.t
    | `Unknown_operand of Tensor_id.t
    | `Unknown_output of Tensor_id.t ]

  let pp_error ppf : [< error ] -> unit = function
    | `Graph_shape e -> D.pp_shape_error ppf e
    | `Invalid_sig (id, e) ->
        Fmt.pf ppf "tensor %a is not a legal signature for this dialect: %a"
          Tensor_id.pp id D.pp_shape_error e
    | `Duplicate_group_id id ->
        Fmt.pf ppf "duplicate group id %a" Group_id.pp id
    | `Duplicate_group_item id ->
        Fmt.pf ppf "node %a is owned by more than one group" Node_id.pp id
    | `Duplicate_node_id id -> Fmt.pf ppf "duplicate node id %a" Node_id.pp id
    | `Duplicate_tensor_def id ->
        Fmt.pf ppf "tensor %a is defined more than once" Tensor_id.pp id
    | `Input_defined_by_node id ->
        Fmt.pf ppf "input %a is also defined by a node" Tensor_id.pp id
    | `Node_not_grouped id -> Fmt.pf ppf "node %a is in no group" Node_id.pp id
    | `Not_topological id ->
        Fmt.pf ppf "node %a reads an edge defined after it" Node_id.pp id
    | `Output_arity { node; expected; actual } ->
        Fmt.pf ppf "node %a declares %d outputs but its op has %d" Node_id.pp
          node actual expected
    | `Output_shape_mismatch id ->
        Fmt.pf ppf "%a's signature is not the shape its op produces"
          Tensor_id.pp id
    | `Sig_key_mismatch { key; recorded } ->
        Fmt.pf ppf "tensor map key %a holds a signature for %a" Tensor_id.pp key
          Tensor_id.pp recorded
    | `Unknown_group_item id ->
        Fmt.pf ppf "group names unknown node %a" Node_id.pp id
    | `Unknown_input_kind id ->
        Fmt.pf ppf "input_kinds names %a, which is not a graph input"
          Tensor_id.pp id
    | `Unknown_operand id ->
        Fmt.pf ppf "operand %a has no definition and is not an input"
          Tensor_id.pp id
    | `Unknown_output id ->
        Fmt.pf ppf "graph output %a has no definition and is not an input"
          Tensor_id.pp id

  type t = {
    graph : graph;
    nodes : node Node_id.Map.t;
    defs : node Tensor_id.Map.t;
        (* absent = graph input, validation guarantees *)
    uses : node list Tensor_id.Map.t; (* in [Graph.nodes] order *)
    outputs : Tensor_id.Set.t;
    order : int Node_id.Map.t;
    groups : Group_id.t Node_id.Map.t;
    parents : Group_id.t Group_id.Map.t;
  }

  let graph t = t.graph
  let node t id = Node_id.Map.find_opt id t.nodes
  let def t id = Tensor_id.Map.find_opt id t.defs
  let uses t id = Option.value (Tensor_id.Map.find_opt id t.uses) ~default:[]
  let sig_of t id = Tensor_id.Map.find_opt id t.graph.Graph.tensors
  let is_graph_output t id = Tensor_id.Set.mem id t.outputs
  let is_constant t id = Graph_common.input_kind t.graph id = Input.Constant
  let topo_index t id = Node_id.Map.find_opt id t.order

  let group_of t id =
    Option.value (Node_id.Map.find_opt id t.groups) ~default:(Group_id.of_int 0)

  (* Root-first path to [g], so a common prefix of two paths is a common ancestor
   chain and its last element is the nearest one. *)
  let ancestors t g =
    let rec up acc g =
      match Group_id.Map.find_opt g t.parents with
      | None -> g :: acc
      | Some parent -> up (g :: acc) parent
    in
    up [] g

  let common_group t ids =
    match ids with
    | [] -> Group_id.of_int 0
    | first :: rest ->
        let common_prefix a b =
          let rec go acc = function
            | x :: xs, y :: ys when Group_id.equal x y -> go (x :: acc) (xs, ys)
            | _ -> List.rev acc
          in
          go [] (a, b)
        in
        let path =
          List.fold_left
            (fun acc id -> common_prefix acc (ancestors t (group_of t id)))
            (ancestors t (group_of t first))
            rest
        in
        (* The prefix always contains the root, so [last] is total. *)
        List.fold_left (fun _ g -> g) (Group_id.of_int 0) path

  (* ---- validation ---------------------------------------------------------- *)

  let fold_result f init l =
    List.fold_left
      (fun acc x -> Core.Syntax.( let* ) acc (fun acc -> f acc x))
      (Core.return init) l

  (* Group ids, ownership of each node, and the parent chain, in one walk. *)
  let index_groups (g : graph) =
    let open Core.Syntax in
    let rec walk (owners, parents, seen) parent (grp : Group.t) =
      let* () =
        if Group_id.Set.mem grp.Group.id seen then
          Core.fail (`Duplicate_group_id grp.Group.id)
        else Core.return ()
      in
      let seen = Group_id.Set.add grp.Group.id seen in
      let parents =
        match parent with
        | None -> parents
        | Some p -> Group_id.Map.add grp.Group.id p parents
      in
      fold_result
        (fun (owners, parents, seen) item ->
          match item with
          | Group.Node id ->
              if Node_id.Map.mem id owners then
                Core.fail (`Duplicate_group_item id)
              else
                Core.return
                  (Node_id.Map.add id grp.Group.id owners, parents, seen)
          | Group.Group child ->
              walk (owners, parents, seen) (Some grp.Group.id) child)
        (owners, parents, seen) grp.Group.items
    in
    walk
      (Node_id.Map.empty, Group_id.Map.empty, Group_id.Set.empty)
      None g.Graph.root

  let of_graph (g : graph) =
    let open Core.Syntax in
    (* node ids unique *)
    let* nodes =
      fold_result
        (fun acc (n : node) ->
          if Node_id.Map.mem n.Node.id acc then
            Core.fail (`Duplicate_node_id n.Node.id)
          else Core.return (Node_id.Map.add n.Node.id n acc))
        Node_id.Map.empty g.Graph.nodes
    in
    (* the group tree is a partition of the node list *)
    let* groups, parents, _ = index_groups g in
    let* () =
      fold_result
        (fun () (n : node) ->
          if Node_id.Map.mem n.Node.id groups then Core.return ()
          else Core.fail (`Node_not_grouped n.Node.id))
        () g.Graph.nodes
    in
    let* () =
      fold_result
        (fun () (id, _) ->
          if Node_id.Map.mem id nodes then Core.return ()
          else Core.fail (`Unknown_group_item id))
        ()
        (Node_id.Map.bindings groups)
    in
    (* every signature is filed under its own id *)
    let* () =
      fold_result
        (fun () (key, (sg : Tensor_sig.t)) ->
          if Tensor_id.equal key sg.id then Core.return ()
          else Core.fail (`Sig_key_mismatch { key; recorded = sg.id }))
        ()
        (Tensor_id.Map.bindings g.Graph.tensors)
    in
    (* single assignment, and inputs are not node-defined *)
    let* defs =
      fold_result
        (fun acc (n : node) ->
          fold_result
            (fun acc id ->
              if Tensor_id.Map.mem id acc then
                Core.fail (`Duplicate_tensor_def id)
              else Core.return (Tensor_id.Map.add id n acc))
            acc n.Node.outputs)
        Tensor_id.Map.empty g.Graph.nodes
    in
    let inputs = Tensor_id.Set.of_list g.Graph.inputs in
    let* () =
      fold_result
        (fun () id ->
          if Tensor_id.Map.mem id defs then
            Core.fail (`Input_defined_by_node id)
          else Core.return ())
        () g.Graph.inputs
    in
    let known id = Tensor_id.Map.mem id defs || Tensor_id.Set.mem id inputs in
    (* operands resolve, outputs resolve *)
    let* () =
      fold_result
        (fun () (n : node) ->
          fold_result
            (fun () id ->
              if known id then Core.return ()
              else Core.fail (`Unknown_operand id))
            () (D.operands n.Node.op))
        () g.Graph.nodes
    in
    let* () =
      fold_result
        (fun () id ->
          if known id then Core.return () else Core.fail (`Unknown_output id))
        () g.Graph.outputs
    in
    (* input_kinds is sparse by design: keys must be inputs, but need not cover
     them — [Graph_common.input_kind] defaults the rest. *)
    let* () =
      fold_result
        (fun () (id, _) ->
          if Tensor_id.Set.mem id inputs then Core.return ()
          else Core.fail (`Unknown_input_kind id))
        ()
        (Tensor_id.Map.bindings g.Graph.input_kinds)
    in
    (* topological order: an operand must be defined by an earlier node *)
    let order =
      List.fold_left
        (fun (i, acc) (n : node) -> (i + 1, Node_id.Map.add n.Node.id i acc))
        (0, Node_id.Map.empty) g.Graph.nodes
      |> snd
    in
    let position id =
      Option.value (Node_id.Map.find_opt id order) ~default:(-1)
    in
    let* () =
      fold_result
        (fun () (n : node) ->
          fold_result
            (fun () operand ->
              match Tensor_id.Map.find_opt operand defs with
              | Some producer
                when position producer.Node.id >= position n.Node.id ->
                  Core.fail (`Not_topological n.Node.id)
              | _ -> Core.return ())
            () (D.operands n.Node.op))
        () g.Graph.nodes
    in
    (* declared output arity matches what the op actually produces *)
    (* NOT [Core.of_option]: its payload argument is eager, and [D.missing_sig]
       is an abstract functor callback. Both dialects implement it as a cheap
       constructor application, but [Dialect.S] does not require that, so
       folding here would run an unbounded callback on every successful
       lookup. *)
    let sig_of id =
      match Tensor_id.Map.find_opt id g.Graph.tensors with
      | Some sg -> Core.return sg
      | None -> Core.fail (D.missing_sig id)
    in
    let* () =
      fold_result
        (fun () (n : node) ->
          let* shapes =
            Core.map_error
              (fun e -> `Graph_shape e)
              (D.output_shape n.Node.op ~sig_of)
          in
          let expected = List.length shapes
          and actual = List.length n.Node.outputs in
          let* () =
            if expected = actual then Core.return ()
            else
              Core.fail (`Output_arity { node = n.Node.id; expected; actual })
          in
          (* The recorded signature must be the shape the op actually produces.
           Arity alone would let a rewrite install an output whose declared
           shape contradicts its operands, which then only surfaces at eval. *)
          fold_result
            (fun () (id, computed) ->
              match Tensor_id.Map.find_opt id g.Graph.tensors with
              | Some sg
                when List.for_all
                       (fun ax ->
                         Dim.equal
                           (Vec6.get sg.Tensor_sig.shape ax)
                           (Vec6.get computed ax))
                       Axis.all ->
                  Core.return ()
              | Some _ -> Core.fail (`Output_shape_mismatch id)
              | None -> Core.fail (`Graph_shape (D.missing_sig id)))
            ()
            (List.combine n.Node.outputs shapes))
        () g.Graph.nodes
    in
    (* Every LIVE id has a signature, and that signature satisfies the dialect.

     Iterating [Graph.tensors] would not be the same check: an id with no entry
     at all is invisible to it. And the lookup that would otherwise catch one —
     [sig_of] inside per-node shape inference — never runs for a graph input
     used DIRECTLY as a graph output, because no node consumes it. So the live
     set is enumerated explicitly.

     [D.validate_sig] is then the only place a dialect invariant that no
     operation implies can be enforced: shape inference constrains outputs, and
     a graph input or a captured constant is produced by no op. Native answers
     [Ok] unconditionally, so nothing about Native's behaviour changes. *)
    let* () =
      let live =
        List.fold_left
          (fun acc (n : node) ->
            List.fold_left
              (fun acc id -> Tensor_id.Set.add id acc)
              acc n.Node.outputs)
          (Tensor_id.Set.of_list (g.Graph.inputs @ g.Graph.outputs))
          g.Graph.nodes
      in
      fold_result
        (fun () id ->
          match Tensor_id.Map.find_opt id g.Graph.tensors with
          | None -> Core.fail (`Graph_shape (D.missing_sig id))
          (* [map_error], not a rebuilt [Core.fail]: the latter captures a fresh
             callstack here and discards the one the dialect check took at the
             point it actually detected the violation. *)
          | Some sg ->
              D.validate_sig sg
              |> Core.map_error (fun e -> `Invalid_sig (id, e)))
        ()
        (Tensor_id.Set.elements live)
    in
    (* Consumers, deduplicated: a node reading the same edge twice (`mul x x`) is
     one consumer, which is what "exactly one use" has to mean for fusion to be
     safe. [Graph.nodes] is topo-ordered and folded in order, so prepending and
     reversing keeps them in execution order. *)
    let uses =
      List.fold_left
        (fun acc (n : node) ->
          List.fold_left
            (fun acc operand ->
              let current =
                Option.value (Tensor_id.Map.find_opt operand acc) ~default:[]
              in
              match current with
              | (previous : node) :: _
                when Node_id.equal previous.Node.id n.Node.id ->
                  acc
              | _ -> Tensor_id.Map.add operand (n :: current) acc)
            acc (D.operands n.Node.op))
        Tensor_id.Map.empty g.Graph.nodes
      |> Tensor_id.Map.map List.rev
    in
    Core.return
      {
        graph = g;
        nodes;
        defs;
        uses;
        outputs = Tensor_id.Set.of_list g.Graph.outputs;
        order;
        groups;
        parents;
      }

  (* ---- topological sort ---------------------------------------------------- *)

  (* Kahn over the given node list only: edges are the operand links between those
   nodes, so a caller can re-sort a spliced list without knowing the rest of the
   graph. Ready nodes are taken in input order, which makes the sort stable.

   Producers come from the SUBJECT list, not from the view's index. The caller
   that matters is [Rewrite.apply], re-sorting a list that contains freshly
   inserted nodes — their outputs are by definition absent from the view, which
   is of the graph before the rewrite, so consulting the view would make
   dependencies between new nodes invisible and happily emit a wrong order. *)
  let topo_sort (subject : node list) =
    let producers =
      List.fold_left
        (fun acc (n : node) ->
          List.fold_left
            (fun acc out -> Tensor_id.Map.add out n.Node.id acc)
            acc n.Node.outputs)
        Tensor_id.Map.empty subject
    in
    let producer_within id = Tensor_id.Map.find_opt id producers in
    (* Within the subject list a node's operand may be produced by another subject
     node; everything else is free, whether a graph input or an edge produced
     outside the list. *)
    let deps =
      List.fold_left
        (fun acc (n : node) ->
          let d =
            List.filter_map producer_within (D.operands n.Node.op)
            |> Node_id.Set.of_list
          in
          Node_id.Map.add n.Node.id d acc)
        Node_id.Map.empty subject
    in
    let rec emit pending deps acc =
      match pending with
      | [] -> Core.return (List.rev acc)
      | _ -> (
          let ready, blocked =
            List.partition
              (fun (n : node) ->
                Node_id.Set.is_empty
                  (Option.value
                     (Node_id.Map.find_opt n.Node.id deps)
                     ~default:Node_id.Set.empty))
              pending
          in
          match ready with
          | [] ->
              let stuck = List.hd blocked in
              Core.fail (`Cycle stuck.Node.id)
          | ready ->
              let done_ids =
                List.fold_left
                  (fun acc (n : node) -> Node_id.Set.add n.Node.id acc)
                  Node_id.Set.empty ready
              in
              let deps =
                Node_id.Map.map (fun d -> Node_id.Set.diff d done_ids) deps
              in
              emit blocked deps (List.rev_append ready acc))
    in
    emit subject deps []
end

(* The Native specialization. Every existing caller — [Rewrite], [Pattern],
   [Pass], the passes, the tests — sees exactly the interface it saw before,
   apart from the shape error now arriving wrapped in [`Graph_shape]. *)
include Make (Native_dialect)
