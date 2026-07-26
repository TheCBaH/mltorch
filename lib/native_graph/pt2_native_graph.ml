open Graph_ir

module Graph_path = struct
  type t = int list

  let root = []
  let child path node_index = path @ [ node_index ]

  let pp fmt path =
    match path with
    | [] -> Format.pp_print_string fmt "root"
    | _ ->
        Fmt.pf fmt "root/%a"
          (Fmt.list ~sep:(Fmt.any "/") Format.pp_print_int)
          path
end

module Tensor_origin = struct
  type t = {
    graph_path : Graph_path.t;
    ssa_name : string;
    meta : Pytorch_types.TensorMeta.t option;
  }
end

module Node_origin = struct
  type t = {
    graph_path : Graph_path.t;
    index : int;
    target : string;
    name : string option;
    metadata : string Schema_runtime.String_map.t;
  }
end

type tensor_origin = Source of Tensor_origin.t | Derived

type t = {
  graph : graph;
  tensor_origins : tensor_origin Tensor_id.Map.t;
  node_origins : Node_origin.t list Node_id.Map.t;
  captured_targets : string Tensor_id.Map.t;
}

type error =
  [ `Unknown_tensor_id of Tensor_id.t
  | `Unknown_node_id of Node_id.t
  | `Captured_target_for_non_constant of Tensor_id.t ]

let pp_error ppf : [< error ] -> unit = function
  | `Unknown_tensor_id id ->
      Format.fprintf ppf "PT2 provenance refers to unknown tensor %a"
        Tensor_id.pp id
  | `Unknown_node_id id ->
      Format.fprintf ppf "PT2 provenance refers to unknown node %a" Node_id.pp
        id
  | `Captured_target_for_non_constant id ->
      Format.fprintf ppf "captured target attached to non-constant tensor %a"
        Tensor_id.pp id

let graph_ids (g : graph) =
  let tensors =
    Tensor_id.Map.fold
      (fun id _ ids -> Tensor_id.Map.add id () ids)
      g.Graph.tensors Tensor_id.Map.empty
  in
  let nodes =
    List.fold_left
      (fun ids (node : node) -> Node_id.Map.add node.Node.id () ids)
      Node_id.Map.empty g.Graph.nodes
  in
  (tensors, nodes)

let make ~graph ~tensor_origins ~node_origins ~captured_targets =
  let tensors, nodes = graph_ids graph in
  let open Core.Syntax in
  let* () =
    Core.List.iter
      (fun (id, _) ->
        if Tensor_id.Map.mem id tensors then Core.return ()
        else Core.fail (`Unknown_tensor_id id))
      (Tensor_id.Map.bindings tensor_origins)
  in
  let* () =
    Core.List.iter
      (fun (id, _) ->
        if Node_id.Map.mem id nodes then Core.return ()
        else Core.fail (`Unknown_node_id id))
      (Node_id.Map.bindings node_origins)
  in
  let* () =
    Core.List.iter
      (fun (id, _) ->
        if Graph_ir.input_kind graph id = Input.Constant then Core.return ()
        else Core.fail (`Captured_target_for_non_constant id))
      (Tensor_id.Map.bindings captured_targets)
  in
  Core.return { graph; tensor_origins; node_origins; captured_targets }

(* ---- the transformation lens ---------------------------------------------- *)

type lens_error =
  [ error
  | Graph_map.error
  | `Sidecar_graph_mismatch
  | `Unknown_destination_node of Node_id.t
  | `Unknown_destination_tensor of Tensor_id.t ]

let pp_lens_error ppf : [< lens_error ] -> unit = function
  | #error as e -> pp_error ppf e
  | #Graph_map.error as e -> Graph_map.pp_error ppf e
  | `Sidecar_graph_mismatch ->
      Format.fprintf ppf "the sidecar does not describe the map's source graph"
  | `Unknown_destination_node id ->
      Format.fprintf ppf "%a is not a node of the destination graph" Node_id.pp
        id
  | `Unknown_destination_tensor id ->
      Format.fprintf ppf "%a is not an edge of the destination graph"
        Tensor_id.pp id

type 'dst lens =
  | Lens : {
      map : ('src, 'dst) Graph_map.t;
      nodes : Node_id.Set.t;
      sidecar : t;
      tensors : Tensor_id.Set.t;
    }
      -> 'dst lens

(* Canonical bytes, not structural equality: [graph] holds a [Tensor_id.Map]
   whose tree shape depends on insertion order, so [=] can call two identical
   maps unequal. A graph that will not encode compares equal to nothing, which is
   the conservative answer. *)
let canonical g = Result.to_option (Graph_json.encode_graph g)

let lens sidecar ~src map ~dst =
  let open Core.Syntax in
  let src_g = Rewrite.graph src and dst_g = Rewrite.graph dst in
  let* () =
    match (canonical sidecar.graph, canonical src_g) with
    | Some a, Some b when String.equal a b -> Core.return ()
    | _ -> Core.fail `Sidecar_graph_mismatch
  in
  let* () =
    (Graph_map.validate map ~src:src_g ~dst:dst_g
      :> (unit, lens_error) Core.result)
  in
  let tensors =
    Tensor_id.Map.fold
      (fun id _ acc -> Tensor_id.Set.add id acc)
      dst_g.Graph.tensors Tensor_id.Set.empty
  in
  let nodes =
    List.fold_left
      (fun acc (n : node) -> Node_id.Set.add n.Node.id acc)
      Node_id.Set.empty dst_g.Graph.nodes
  in
  Core.return (Lens { map; nodes; sidecar; tensors })

(* The source ids a destination edge corresponds to, with the claim over them.
   An id in no cluster is implicitly [Identical] to itself (§3); [Set.elements]
   is ascending, so every list below is deterministic. *)
let sources_of (Lens l) id =
  match
    List.find_opt
      (fun (c : Correspondence.Cluster.t) -> Tensor_id.Set.mem id c.dst)
      (Correspondence.clusters l.map.Graph_map.values)
  with
  | Some c -> (Tensor_id.Set.elements c.src, c.label)
  | None -> ([ id ], Correspondence.Identical)

let known_tensor (Lens l) id =
  if Tensor_id.Set.mem id l.tensors then Core.return ()
  else Core.fail (`Unknown_destination_tensor id)

let dedup_by key l =
  List.fold_left
    (fun (seen, acc) x ->
      if List.mem (key x) seen then (seen, acc) else (key x :: seen, x :: acc))
    ([], []) l
  |> snd |> List.rev

let tensor_origins (Lens l as lens) id =
  let open Core.Syntax in
  let+ () = known_tensor lens id in
  let sources, _ = sources_of lens id in
  List.filter_map
    (fun s ->
      match Tensor_id.Map.find_opt s l.sidecar.tensor_origins with
      | Some (Source o) -> Some o
      | Some Derived | None -> None)
    sources
  |> dedup_by (fun (o : Tensor_origin.t) -> (o.graph_path, o.ssa_name))

let node_origins (Lens l) id =
  let open Core.Syntax in
  let+ () =
    if Node_id.Set.mem id l.nodes then Core.return ()
    else Core.fail (`Unknown_destination_node id)
  in
  let sources =
    match
      List.find_opt
        (fun (c : Node_map.Cluster.t) -> Node_id.Set.mem id c.dst)
        (Node_map.clusters l.map.Graph_map.nodes)
    with
    | Some c -> Node_id.Set.elements c.src
    | None -> [ id ]
  in
  let key (o : Node_origin.t) = (o.graph_path, o.index) in
  List.concat_map
    (fun s ->
      Option.value (Node_id.Map.find_opt s l.sidecar.node_origins) ~default:[])
    sources
  |> List.stable_sort (fun a b -> compare (key a) (key b))
  |> dedup_by key

let captured_target (Lens l as lens) id =
  let open Core.Syntax in
  let+ () = known_tensor lens id in
  match sources_of lens id with
  | sources, Correspondence.Identical ->
      List.find_map
        (fun s -> Tensor_id.Map.find_opt s l.sidecar.captured_targets)
        sources
  (* Anything weaker than [Identical] means the bytes differ by construction. *)
  | _, _ -> None

let provenance_sources (Lens l) id =
  Tensor_id.Set.elements (Provenance.sources_of l.map.Graph_map.provenance id)
