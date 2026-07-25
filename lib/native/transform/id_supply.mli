(* Monotone id allocation for the transformation framework. Seeded ONCE from the
   origin graph and only ever advanced: it is never re-derived from the current
   graph, because re-deriving would re-issue the id of a deleted edge and an id
   would stop denoting one tensor for the life of a pipeline. Terminal packing
   ([Rewrite.pack]) is the single bounded exception, which is why [origin] is
   kept alongside [next]. See .ai/native_transform_design.md §4, §9. *)

open Graph_ir

type t

(* Watermarks: the first free id in each space. [of_graph] takes them one past
   the highest id the graph mentions anywhere — tensor signature keys, node ids,
   and every group id in the tree — so a fresh id differs from every id the graph
   used, including ids of edges a later step deletes. *)
val of_graph : graph -> t

(* The origin's watermark, frozen at [of_graph] and carried unchanged through
   every advance. Packing compacts post-origin ids upward from here, which is what
   keeps them disjoint from every origin id. *)
val origin : t -> t
val tensor : t -> Tensor_id.t * t
val node : t -> Node_id.t * t
val group : t -> Group_id.t * t

(* Allocate [n] tensor ids at once, in ascending order. *)
val tensors : t -> int -> Tensor_id.t list * t

(* True when [id] is at or above the given watermark, i.e. it was introduced
   after that point rather than carried over. *)
val is_post : t -> Tensor_id.t -> bool
val is_post_node : t -> Node_id.t -> bool
val is_post_group : t -> Group_id.t -> bool

(* Structural equality of the watermarks, for the [Rewrite] staleness and
   contiguity checks (a recipe's end watermark must equal the next one's start). *)
val equal : t -> t -> bool
val pp : Format.formatter -> t -> unit
