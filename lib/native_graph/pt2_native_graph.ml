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
  let open Err.Syntax in
  let* () =
    Err.List.iter
      (fun (id, _) ->
        if Tensor_id.Map.mem id tensors then Err.return ()
        else Err.fail (`Unknown_tensor_id id))
      (Tensor_id.Map.bindings tensor_origins)
  in
  let* () =
    Err.List.iter
      (fun (id, _) ->
        if Node_id.Map.mem id nodes then Err.return ()
        else Err.fail (`Unknown_node_id id))
      (Node_id.Map.bindings node_origins)
  in
  let* () =
    Err.List.iter
      (fun (id, _) ->
        if Graph_ir.input_kind graph id = Input.Constant then Err.return ()
        else Err.fail (`Captured_target_for_non_constant id))
      (Tensor_id.Map.bindings captured_targets)
  in
  Err.return { graph; tensor_origins; node_origins; captured_targets }

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
      dst : 'dst Snapshot.t;
      map : ('src, 'dst) Graph_map.t;
      sidecar : t;
      src : 'src Snapshot.t;
    }
      -> 'dst lens

(* Canonical bytes, not structural equality: [graph] holds a [Tensor_id.Map]
   whose tree shape depends on insertion order, so [=] can call two identical
   maps unequal. A graph that will not encode compares equal to nothing, which is
   the conservative answer. *)
let canonical g = Result.to_option (Graph_json.encode_graph g)

let lens sidecar ~src map ~dst =
  let open Err.Syntax in
  let src = Rewrite.snapshot src and dst = Rewrite.snapshot dst in
  let* () =
    match (canonical sidecar.graph, canonical (Snapshot.graph src)) with
    | Some a, Some b when String.equal a b -> Err.return ()
    | _ -> Err.fail `Sidecar_graph_mismatch
  in
  (* Endpoints were checked when the map was built; closure was not, because the
     map handed to a lens is composed and [Graph_map.compose] cannot re-establish
     it. An unclosed map is precisely what would make [captured_target] below
     return source bytes for an edge whose value differs. *)
  let* () =
    (Graph_map.check_claim_closure map ~src ~dst :> (unit, lens_error) Err.t)
  in
  Err.return (Lens { dst; map; sidecar; src })

let dst_edge (Lens l) id =
  Snapshot.edge l.dst id |> Err.of_option (`Unknown_destination_tensor id)

let dst_node (Lens l) id =
  Snapshot.node l.dst id |> Err.of_option (`Unknown_destination_node id)

(* The source ids a destination edge corresponds to, with the claim over them.
   An id in no cluster is implicitly [Identical] to itself (§3) — meaningful only
   where the SOURCE graph has that id, which is why the fallback is a lookup in
   [l.src] rather than a bare retag of the number. [Set.elements] is ascending,
   so every list below is deterministic. *)
let sources_of (Lens l) d =
  match
    List.find_opt
      (fun (c : (_, _) Correspondence.Cluster.t) ->
        Correspondence.Set.mem d c.dst)
      (Correspondence.clusters (Graph_map.values l.map))
  with
  | Some c -> (Correspondence.raws c.src |> Tensor_id.Set.elements, c.label)
  | None ->
      let id = Correspondence.raw d in
      ( (match Snapshot.edge l.src id with Some _ -> [ id ] | None -> []),
        Correspondence.Identical )

let dedup_by key l =
  List.fold_left
    (fun (seen, acc) x ->
      if List.mem (key x) seen then (seen, acc) else (key x :: seen, x :: acc))
    ([], []) l
  |> snd |> List.rev

let tensor_origins (Lens l as lens) id =
  let open Err.Syntax in
  let+ d = dst_edge lens id in
  let sources, _ = sources_of lens d in
  List.filter_map
    (fun s ->
      match Tensor_id.Map.find_opt s l.sidecar.tensor_origins with
      | Some (Source o) -> Some o
      | Some Derived | None -> None)
    sources
  |> dedup_by (fun (o : Tensor_origin.t) -> (o.graph_path, o.ssa_name))

let node_origins (Lens l as lens) id =
  let open Err.Syntax in
  let+ n = dst_node lens id in
  let sources =
    match
      List.find_opt
        (fun (c : (_, _) Node_map.Cluster.t) -> Node_map.Set.mem n c.dst)
        (Node_map.clusters (Graph_map.nodes l.map))
    with
    | Some c -> Node_map.raws c.src |> Node_id.Set.elements
    | None -> (
        match Snapshot.node l.src id with Some _ -> [ id ] | None -> [])
  in
  let key (o : Node_origin.t) = (o.graph_path, o.index) in
  List.concat_map
    (fun s ->
      Option.value (Node_id.Map.find_opt s l.sidecar.node_origins) ~default:[])
    sources
  |> List.stable_sort (fun a b -> compare (key a) (key b))
  |> dedup_by key

let captured_target (Lens l as lens) id =
  let open Err.Syntax in
  let+ d = dst_edge lens id in
  match sources_of lens d with
  | sources, Correspondence.Identical ->
      List.find_map
        (fun s -> Tensor_id.Map.find_opt s l.sidecar.captured_targets)
        sources
  (* Anything weaker than [Identical] means the bytes differ by construction. *)
  | _, _ -> None

let provenance_sources (Lens l) id =
  Snapshot.edge l.dst id
  |> Option.fold ~none:[] ~some:(fun d ->
      Provenance.sources_of (Graph_map.provenance l.map) d
      |> Correspondence.raws |> Tensor_id.Set.elements)
