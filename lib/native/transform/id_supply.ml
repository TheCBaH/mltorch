(* See id_supply.mli. Three independent counters plus the frozen origin
   watermarks; everything is a plain int triple, so advancing is allocation-free
   and comparing watermarks is cheap. *)

open Graph_ir

type marks = { tensor : int; node : int; group : int }
type t = { next : marks; origin : marks }

(* One past the highest id in each space. Every group in the tree is visited, so
   a fresh group id cannot collide with a nested one. *)
(* Op-polymorphic: watermarks are about ids, and ids are dialect-free. *)
let marks_of_graph : 'op. 'op Graph_common.Graph.t -> marks =
 fun g ->
  let tensor =
    Tensor_id.Map.fold
      (fun id _ acc -> max acc (Tensor_id.to_int id + 1))
      g.Graph.tensors 0
  in
  (* Inputs are always in [tensors], but scan them anyway: a graph that failed
     validation should still yield a supply that cannot collide. *)
  let tensor =
    List.fold_left
      (fun acc id -> max acc (Tensor_id.to_int id + 1))
      tensor
      (g.Graph.inputs @ g.Graph.outputs)
  in
  let tensor =
    List.fold_left
      (fun acc (n : _ Graph_common.Node.t) ->
        List.fold_left
          (fun acc id -> max acc (Tensor_id.to_int id + 1))
          acc n.Node.outputs)
      tensor g.Graph.nodes
  in
  let node =
    List.fold_left
      (fun acc (n : _ Graph_common.Node.t) ->
        max acc (Node_id.to_int n.Node.id + 1))
      0 g.Graph.nodes
  in
  let rec group_marks acc (grp : Group.t) =
    let acc = max acc (Group_id.to_int grp.Group.id + 1) in
    List.fold_left
      (fun acc -> function
        | Group.Node _ -> acc | Group.Group child -> group_marks acc child)
      acc grp.Group.items
  in
  { tensor; node; group = group_marks 0 g.Graph.root }

let of_graph : 'op. 'op Graph_common.Graph.t -> t =
 fun g ->
  let marks = marks_of_graph g in
  { next = marks; origin = marks }

let origin t = { next = t.origin; origin = t.origin }

let tensor t =
  ( Tensor_id.of_int t.next.tensor,
    { t with next = { t.next with tensor = t.next.tensor + 1 } } )

let node t =
  ( Node_id.of_int t.next.node,
    { t with next = { t.next with node = t.next.node + 1 } } )

let group t =
  ( Group_id.of_int t.next.group,
    { t with next = { t.next with group = t.next.group + 1 } } )

let tensors t n =
  let rec take t acc = function
    | 0 -> (List.rev acc, t)
    | n ->
        let id, t = tensor t in
        take t (id :: acc) (n - 1)
  in
  if n <= 0 then ([], t) else take t [] n

let origin_marks t = (t.origin.tensor, t.origin.node, t.origin.group)
let repack t ~tensor ~node ~group = { t with next = { tensor; node; group } }
let is_post t id = Tensor_id.to_int id >= t.origin.tensor
let is_post_node t id = Node_id.to_int id >= t.origin.node
let is_post_group t id = Group_id.to_int id >= t.origin.group
let equal a b = a.next = b.next && a.origin = b.origin

let pp fmt t =
  Fmt.pf fmt "@[<h>ids next=(t%d n%d g%d) origin=(t%d n%d g%d)@]" t.next.tensor
    t.next.node t.next.group t.origin.tensor t.origin.node t.origin.group
