(* See id_supply.mli. Three independent counters plus the frozen origin
   watermarks, so advancing is allocation-light and comparing watermarks is
   cheap. *)

open Graph_ir

module Marks = struct
  type t = {
    tensor : Tensor_id.Next.t;
    node : Node_id.Next.t;
    group : Group_id.Next.t;
  }
end

type t = { next : Marks.t; origin : Marks.t }

(* One past the highest id in each space. Every group in the tree is visited, so
   a fresh group id cannot collide with a nested one. *)
(* Op-polymorphic: watermarks are about ids, and ids are dialect-free. *)
let marks_of_graph : 'op. 'op Graph_common.Graph.t -> Marks.t =
 fun g ->
  let tensor =
    Tensor_id.Map.fold
      (fun id _ acc -> Tensor_id.Next.after id acc)
      g.Graph.tensors Tensor_id.Next.first
  in
  (* Inputs are always in [tensors], but scan them anyway: a graph that failed
     validation should still yield a supply that cannot collide. *)
  let tensor =
    List.fold_left
      (fun acc id -> Tensor_id.Next.after id acc)
      tensor
      (g.Graph.inputs @ g.Graph.outputs)
  in
  let tensor =
    List.fold_left
      (fun acc (n : _ Graph_common.Node.t) ->
        List.fold_left
          (fun acc id -> Tensor_id.Next.after id acc)
          acc n.Node.outputs)
      tensor g.Graph.nodes
  in
  let node =
    List.fold_left
      (fun acc (n : _ Graph_common.Node.t) -> Node_id.Next.after n.Node.id acc)
      Node_id.Next.first g.Graph.nodes
  in
  let rec group_marks acc (grp : Group.t) =
    let acc = Group_id.Next.after grp.Group.id acc in
    List.fold_left
      (fun acc -> function
        | Group.Group child -> group_marks acc child | Group.Node _ -> acc)
      acc grp.Group.items
  in
  { Marks.tensor; node; group = group_marks Group_id.Next.first g.Graph.root }

let of_graph : 'op. 'op Graph_common.Graph.t -> t =
 fun g ->
  let marks = marks_of_graph g in
  { next = marks; origin = marks }

let origin t = { next = t.origin; origin = t.origin }
let next_tensor t = t.next.Marks.tensor
let next_node t = t.next.Marks.node

let tensor t =
  let id, tensor = Tensor_id.Next.alloc t.next.Marks.tensor in
  (id, { t with next = { t.next with Marks.tensor } })

let node t =
  let id, node = Node_id.Next.alloc t.next.Marks.node in
  (id, { t with next = { t.next with Marks.node } })

let group t =
  let id, group = Group_id.Next.alloc t.next.Marks.group in
  (id, { t with next = { t.next with Marks.group } })

let tensors t n =
  if n <= 0 then ([], t)
  else
    let ids, tensor = Tensor_id.Next.alloc_n t.next.Marks.tensor n in
    (ids, { t with next = { t.next with Marks.tensor } })

let origin_marks t = t.origin
let repack t next = { t with next }
let is_post t id = Tensor_id.Next.reaches t.origin.Marks.tensor id
let is_post_node t id = Node_id.Next.reaches t.origin.Marks.node id
let is_post_group t id = Group_id.Next.reaches t.origin.Marks.group id
let equal a b = a.next = b.next && a.origin = b.origin

let pp fmt t =
  Fmt.pf fmt "@[<h>ids next=(%a %a %a) origin=(%a %a %a)@]" Tensor_id.Next.pp
    t.next.Marks.tensor Node_id.Next.pp t.next.Marks.node Group_id.Next.pp
    t.next.Marks.group Tensor_id.Next.pp t.origin.Marks.tensor Node_id.Next.pp
    t.origin.Marks.node Group_id.Next.pp t.origin.Marks.group
