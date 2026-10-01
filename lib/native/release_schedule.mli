(* When each edge's tensor may leave a direct evaluator's env: right after the
   last node that reads it. Shared by both dialects' evaluators, so the one
   definition of "reader" also decides which index outputs are allocated at
   all. Generic over the op type rather than a functor: it has two callers.
   See .ai/ (tensor release). *)

open Graph_common

(* Which edges a run's result keeps. [g.outputs] are always kept. *)
module Retain : sig
  type t =
    | All  (** every edge: the evaluators' default *)
    | Only of Tensor_id.Set.t  (** these and [g.outputs]; the rest released *)
end

module Schedule : sig
  type t
end

val schedule :
  operands:('op -> Tensor_id.t list) ->
  is_sink:('op -> bool) ->
  retain:Retain.t ->
  'op Graph.t ->
  Schedule.t
(** One backward pass over [g.nodes]. A sink ([Discard]) is not a reader and
    produces nothing. Under [Retain.All] nothing is ever released, but
    [has_reader] still answers. *)

val initial : Schedule.t -> Tensor_id.t list
(** Graph inputs no non-sink node reads: released before the first node. *)

val after : Schedule.t -> Node_id.t -> Tensor_id.t list
(** Edges released once the node has run: each operand it is the last reader of,
    and each of its outputs nothing reads. *)

val has_reader : Schedule.t -> Tensor_id.t -> bool
(** A graph output, or an operand of a non-sink node. *)

val peak_bytes :
  is_index_output:('op -> Output_ordinal.t -> bool) ->
  'op Graph.t ->
  Schedule.t ->
  ( int64,
    [> `Missing_tensor_sig of Tensor_id.t
    | `Numel_over_limit of Vec6.Numel_bound.t
    | `Peak_bytes_overflow of Tensor_id.t ] )
  Err.t
(** The most payload bytes resident at once when the schedule is followed:
    [g.inputs] throughout, each node's outputs from the node on, and an edge
    until its release. Measured after a node's outputs exist and before its
    releases, since both are live while it runs. An index output nothing reads
    is skipped, as the evaluators never allocate it. *)
