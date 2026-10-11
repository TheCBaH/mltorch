open Graph_ir

module Node_cluster = struct
  type t = { src : Node_id.t list; dst : Node_id.t list }
end

module Value_cluster = struct
  type t = {
    src : Tensor_id.t list;
    dst : Tensor_id.t list;
    claim : Correspondence.relation;
  }
end

type t = { nodes : Node_cluster.t list; values : Value_cluster.t list }

module Make (Src : Side.S) (Dst : Side.S) = struct
  module Pair = Graph_map.Make_pair (Src) (Dst)

  let ascending compare l = List.sort_uniq compare l

  let export map ~src ~dst =
    let values =
      List.map
        (fun (c : (_, _) Correspondence.Cluster.t) ->
          {
            Value_cluster.src =
              ascending Tensor_id.compare
                (Tensor_id.Set.elements (Correspondence.raws c.src));
            dst =
              ascending Tensor_id.compare
                (Tensor_id.Set.elements (Correspondence.raws c.dst));
            claim = c.label;
          })
        (Pair.clusters_over map ~src ~dst)
    in
    let explicit = Node_map.clusters (Graph_map.nodes map) in
    let raws s = Node_id.Set.elements (Node_map.raws s) in
    let mentioned =
      List.fold_left
        (fun acc (c : (_, _) Node_map.Cluster.t) ->
          Node_id.Set.union acc (Node_map.raws c.src))
        Node_id.Set.empty explicit
    in
    let implicit =
      List.filter_map
        (fun s ->
          let id = Node_map.raw s in
          if Node_id.Set.mem id mentioned then None
          else
            Option.map
              (fun _ -> { Node_cluster.src = [ id ]; dst = [ id ] })
              (Dst.Snapshot.node dst id))
        (Node_map.Universe.elements (Src.Snapshot.nodes src))
    in
    let nodes =
      List.map
        (fun (c : (_, _) Node_map.Cluster.t) ->
          {
            Node_cluster.src = ascending Node_id.compare (raws c.src);
            dst = ascending Node_id.compare (raws c.dst);
          })
        explicit
      @ implicit
    in
    { nodes; values }
end
