(* The allocation problems a dry run's [Alloc_script] poses: one validated
   interval-allocation script per element kind, over the eligible edges only.
   Shared by [Arena_plan], which solves them for a run, and by any evaluation
   that measures allocators on exactly the same problems.

   Only eligible edges take part; their frees keep their place among the
   allocs, and the [Node] markers are dropped: an interval allocator needs
   order, not node boundaries. An operand freed after its consumer's output was
   allocated therefore still conflicts with that output, and an eligible edge
   never freed stays live to the end of the script.

   A block is placed at its exact payload size, its start at its alignment.
   Padding is only for a minimum: the [padded] script poses every block padded
   to its alignment, so a padded bound is provable where an exact one, which
   leaves the alignment gaps out, often is not. Any placement of the padded
   script is also one of the exact script, so its minimum bounds the exact
   optimum from above, and the exact script's live bound from below. *)

module Kind = Alloc_script.Kind

(** One kind's problem, in bytes, each block placed at its script alignment:
    [script] at its exact size, [padded] padded to that alignment. *)
module Kind_problem : sig
  type t = {
    kind : Kind.t;
    script : Tensor_id.t Interval_alloc.Script.t;
    padded : Tensor_id.t Interval_alloc.Script.t;
  }
end

type t

val of_script : Alloc_script.t -> (t, [> `Arena_script of Tensor_id.t ]) Err.t
(** [`Arena_script id]: the script allocates or frees [id] inconsistently. *)

val script : t -> Alloc_script.t
(** The whole script the problems were projected from. *)

val kinds : t -> Kind_problem.t list
(** In [Kind.all] order; a kind with no eligible edge has no problem. *)

val eligible : t -> Tensor_id.t -> Alloc_script.Alloc.t option

val first_id : t -> Tensor_id.t
(** The smallest eligible edge, or edge 0 when there is none: the edge an
    aggregate's overflow is reported against. *)

val combined_bound_bytes :
  t ->
  ( Core.Storage_units.Byte_size.t,
    [> `Peak_bytes_overflow of Tensor_id.t ] )
  Err.t
(** The live bound of one pool shared by every kind, in padded bytes: a
    minimum's lower bound, not an exact placement's. *)

val out_of_arena_bytes :
  t ->
  ( Core.Storage_units.Byte_size.t,
    [> `Peak_bytes_overflow of Tensor_id.t ] )
  Err.t
(** The script's ineligible allocations, summed. *)
