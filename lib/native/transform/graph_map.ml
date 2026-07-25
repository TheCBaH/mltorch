(* See graph_map.mli. *)

open Graph_ir

type ('src, 'dst) t = {
  values : ('src, 'dst) Correspondence.t;
  nodes : ('src, 'dst) Node_map.t;
  provenance : ('src, 'dst) Provenance.t;
}

type error =
  [ `Node_endpoint of Node_id.t Cluster_relation.issue
  | `Provenance_endpoint of Tensor_id.t Cluster_relation.issue
  | `Value_endpoint of Tensor_id.t Cluster_relation.issue ]

let pp_error ppf : [< error ] -> unit = function
  | `Node_endpoint issue ->
      Fmt.pf ppf "@[<h>node map: %a@]"
        (Cluster_relation.pp_issue Node_id.pp)
        issue
  | `Provenance_endpoint issue ->
      Fmt.pf ppf "@[<h>provenance: %a@]"
        (Cluster_relation.pp_issue Tensor_id.pp)
        issue
  | `Value_endpoint issue ->
      Fmt.pf ppf "@[<h>value map: %a@]"
        (Cluster_relation.pp_issue Tensor_id.pp)
        issue

let identity =
  {
    values = Correspondence.identity;
    nodes = Node_map.identity;
    provenance = Provenance.empty;
  }

let compose a b =
  {
    values = Correspondence.compose a.values b.values;
    nodes = Node_map.compose a.nodes b.nodes;
    provenance =
      Provenance.compose a.provenance b.provenance ~values:(a.values, b.values);
  }

let clusters t = Correspondence.clusters t.values

let tensor_ids (g : graph) =
  Tensor_id.Map.fold
    (fun id _ acc -> Tensor_id.Set.add id acc)
    g.Graph.tensors Tensor_id.Set.empty

let node_ids (g : graph) =
  List.fold_left
    (fun acc (n : node) -> Node_id.Set.add n.Node.id acc)
    Node_id.Set.empty g.Graph.nodes

(* The implicit identities the relation deliberately does not store: every id
   present in both graphs and mentioned by no cluster. *)
let clusters_over t ~src ~dst =
  let explicit = Correspondence.clusters t.values in
  let mentioned =
    List.fold_left
      (fun acc (c : Correspondence.Cluster.t) ->
        Tensor_id.Set.union acc (Tensor_id.Set.union c.src c.dst))
      Tensor_id.Set.empty explicit
  in
  let common = Tensor_id.Set.inter (tensor_ids src) (tensor_ids dst) in
  let implicit =
    Tensor_id.Set.fold
      (fun id acc ->
        if Tensor_id.Set.mem id mentioned then acc
        else
          {
            Correspondence.Cluster.src = Tensor_id.Set.singleton id;
            dst = Tensor_id.Set.singleton id;
            label = Correspondence.Identical;
          }
          :: acc)
      common []
  in
  explicit @ List.rev implicit

let validate t ~src ~dst =
  let open Core.Syntax in
  let lift wrap = function
    | Ok () -> Core.return ()
    | Error issue -> Core.fail (wrap issue)
  in
  let src_tensors = tensor_ids src and dst_tensors = tensor_ids dst in
  let* () =
    lift
      (fun i -> `Value_endpoint i)
      (Correspondence.validate t.values ~src:src_tensors ~dst:dst_tensors)
  in
  let* () =
    lift
      (fun i -> `Node_endpoint i)
      (Node_map.validate t.nodes ~src:(node_ids src) ~dst:(node_ids dst))
  in
  lift
    (fun i -> `Provenance_endpoint i)
    (Provenance.validate t.provenance ~src:src_tensors ~dst:dst_tensors)

let pp fmt t =
  Fmt.pf fmt
    "@[<v>@[<v 2>values:@,%a@]@,@[<v 2>nodes:@,%a@]@,@[<v 2>provenance:@,%a@]@]"
    Correspondence.pp t.values Node_map.pp t.nodes Provenance.pp t.provenance
