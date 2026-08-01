(* See region.mli. *)

open Graph_ir

module Make (D : Dialect.S) = struct
  module View = Graph_view.Make (D)

  type node = D.op Graph_common.Node.t
  type graph = D.op Graph_common.Graph.t
  type error = [ `Empty_region | `Unknown_node of Node_id.t ]

  let pp_error ppf : [< error ] -> unit = function
    | `Empty_region -> Fmt.string ppf "region claims no nodes"
    | `Unknown_node id ->
        Fmt.pf ppf "region claims unknown node %a" Node_id.pp id

  type t = {
    nodes : Node_id.Set.t;
    inputs : Tensor_id.t list;
    outputs : Tensor_id.t list;
    interior : Tensor_id.t list;
    convex : bool;
  }

  (* Claimed nodes in execution order, which is what makes every derived list
     below deterministic. *)
  let ordered view nodes =
    Node_id.Set.elements nodes
    |> List.filter_map (fun id -> View.node view id)
    |> List.sort (fun (a : node) (b : node) ->
        Int.compare
          (Option.value (View.topo_index view a.Node.id) ~default:0)
          (Option.value (View.topo_index view b.Node.id) ~default:0))

  let dedup l =
    List.fold_left
      (fun (seen, acc) id ->
        if Tensor_id.Set.mem id seen then (seen, acc)
        else (Tensor_id.Set.add id seen, id :: acc))
      (Tensor_id.Set.empty, []) l
    |> snd |> List.rev

  (* Does anything downstream of the region re-enter it? Walk forward from the
     external consumers of the region's outputs; if the walk reaches a claimed
     node, some path left and came back. *)
  let is_convex view nodes outputs =
    let claimed id = Node_id.Set.mem id nodes in
    let rec walk visited = function
      | [] -> true
      | (n : node) :: rest ->
          if claimed n.Node.id then false
          else if Node_id.Set.mem n.Node.id visited then walk visited rest
          else
            let visited = Node_id.Set.add n.Node.id visited in
            let next =
              List.concat_map (fun id -> View.uses view id) n.Node.outputs
            in
            walk visited (next @ rest)
    in
    let external_consumers =
      List.concat_map
        (fun id ->
          List.filter
            (fun (n : node) -> not (claimed n.Node.id))
            (View.uses view id))
        outputs
    in
    walk Node_id.Set.empty external_consumers

  let of_nodes view nodes =
    let open Core.Syntax in
    let* () =
      if Node_id.Set.is_empty nodes then Core.fail `Empty_region
      else Core.return ()
    in
    let* () =
      Node_id.Set.fold
        (fun id acc ->
          let* () = acc in
          match View.node view id with
          | Some _ -> Core.return ()
          | None -> Core.fail (`Unknown_node id))
        nodes (Core.return ())
    in
    let claimed id = Node_id.Set.mem id nodes in
    let members = ordered view nodes in
    let produced_inside id =
      match View.def view id with
      | Some (p : node) -> claimed p.Node.id
      | None -> false
    in
    let inputs =
      List.concat_map (fun (n : node) -> D.operands n.Node.op) members
      |> List.filter (fun id -> not (produced_inside id))
      |> dedup
    in
    let escapes id =
      View.is_graph_output view id
      || List.exists
           (fun (n : node) -> not (claimed n.Node.id))
           (View.uses view id)
    in
    let defined = List.concat_map (fun (n : node) -> n.Node.outputs) members in
    let outputs = List.filter escapes defined |> dedup in
    let interior = List.filter (fun id -> not (escapes id)) defined |> dedup in
    Core.return
      {
        nodes;
        inputs;
        outputs;
        interior;
        convex = is_convex view nodes outputs;
      }

  let extract view t =
    let members = ordered view t.nodes in
    let tensors =
      List.fold_left
        (fun acc id ->
          match View.sig_of view id with
          | Some sg -> Tensor_id.Map.add id sg acc
          | None -> acc)
        Tensor_id.Map.empty
        (t.inputs @ List.concat_map (fun (n : node) -> n.Node.outputs) members)
    in
    (* Kept sparse the way the IR keeps it: only constants get an entry. *)
    let input_kinds =
      List.fold_left
        (fun acc id ->
          if View.is_constant view id then
            Tensor_id.Map.add id Input.Constant acc
          else acc)
        Tensor_id.Map.empty t.inputs
    in
    {
      Graph.nodes = members;
      root =
        {
          Group.id = Group_id.of_int 0;
          label = Some "match";
          items = List.map (fun (n : node) -> Group.Node n.Node.id) members;
        };
      tensors;
      inputs = t.inputs;
      input_kinds;
      outputs = t.outputs;
    }

  let pp fmt t =
    let ids pp_id fmt l = Fmt.brackets (Fmt.list ~sep:Fmt.comma pp_id) fmt l in
    Fmt.pf fmt
      "@[<v>@[<h>nodes: %a@]@,\
       @[<h>inputs: %a@]@,\
       @[<h>outputs: %a@]@,\
       @[<h>interior: %a@]@,\
       @[<h>convex: %b@]@]"
      (ids Node_id.pp)
      (Node_id.Set.elements t.nodes)
      (ids Tensor_id.pp) t.inputs (ids Tensor_id.pp) t.outputs
      (ids Tensor_id.pp) t.interior t.convex
end

include Make (Native_dialect)
