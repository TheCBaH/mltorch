(* See snapshot.mli. Parameterised over the dialect, with the Native
   specialization at the bottom so no existing caller changes. *)

open Graph_common

module Make (D : Dialect.S) = struct
  module View = Graph_view.Make (D)

  type 'v t = {
    edges : 'v Correspondence.Universe.t;
    graph : D.op Graph.t;
    nodes : 'v Node_map.Universe.t;
    view : View.t;
  }

  type packed = Pack : 'v t -> packed
  type 'v edge = 'v Correspondence.id
  type 'v node = 'v Node_map.id

  let tensor_ids (g : D.op Graph.t) =
    Tensor_id.Map.fold
      (fun id _ acc -> Tensor_id.Set.add id acc)
      g.Graph.tensors Tensor_id.Set.empty

  let node_ids (g : D.op Graph.t) =
    List.fold_left
      (fun acc (n : D.op Node.t) -> Node_id.Set.add n.Node.id acc)
      Node_id.Set.empty g.Graph.nodes

  (* One brand for both id spaces, so tensors and nodes share the version. *)
  let create g =
    let open Core.Syntax in
    let+ view = View.of_graph g in
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
end

include Make (Native_dialect)
