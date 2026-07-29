(* See snapshot.mli. *)

open Graph_ir

type 'v t = {
  edges : 'v Correspondence.Universe.t;
  graph : graph;
  nodes : 'v Node_map.Universe.t;
  view : Graph_view.t;
}

type packed = Pack : 'v t -> packed
type 'v edge = 'v Correspondence.id
type 'v node = 'v Node_map.id

let tensor_ids (g : graph) =
  Tensor_id.Map.fold
    (fun id _ acc -> Tensor_id.Set.add id acc)
    g.Graph.tensors Tensor_id.Set.empty

let node_ids (g : graph) =
  List.fold_left
    (fun acc (n : Node.t) -> Node_id.Set.add n.Node.id acc)
    Node_id.Set.empty g.Graph.nodes

(* One brand for both id spaces, so tensors and nodes share the version. *)
let create g =
  let open Core.Syntax in
  let+ view = Graph_view.of_graph g in
  let (Brand.Pack brand) = Brand.fresh () in
  Pack
    {
      edges = Correspondence.Universe.create brand (tensor_ids g);
      graph = g;
      nodes = Node_map.Universe.create brand (node_ids g);
      view;
    }

let edge t id = Correspondence.Universe.find t.edges id
let node t id = Node_map.Universe.find t.nodes id
let edges t = t.edges
let nodes t = t.nodes
let graph t = t.graph
let view t = t.view
