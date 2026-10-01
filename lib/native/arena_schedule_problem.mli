(* The facts a schedule is searched over, extracted once from a baseline dry
   run: which tensors the evaluator allocates and how big they are, which node
   produces and which nodes read each, and which tensors a run ever releases.
   Every fact here is independent of the node order; only the release time of a
   tensor depends on it. Intermediate lifetimes only (see
   [Arena_schedule.Mode]). *)

open Graph_ir
open Core.Storage_units

module Position = Graph_view.Position
(** A node's place in the original order. *)

module Block : sig
  type t = {
    id : Tensor_id.t;
    kind : Alloc_script.Kind.t;
    bytes : Byte_size.t;
    eligible : bool;  (** [Alloc_script.Alloc.eligible]: an arena may back it *)
    releasable : bool;
        (** The baseline run frees it: false for graph outputs and retained
            edges, whatever their kind. *)
  }
end

type error =
  [ Arena_schedule.error
  | `Arena_script of Tensor_id.t
  | `Dry_run of Eval_direct.error
  | `Order_not_valid of Node_id.t
  | `Peak_bytes_overflow of Tensor_id.t
  | `Roles_unsupported
  | `Too_many_nodes ]

val pp_error : Format.formatter -> [< error ] -> unit

val add :
  id:Tensor_id.t ->
  Byte_size.t ->
  Byte_size.t ->
  (Byte_size.t, [> `Peak_bytes_overflow of Tensor_id.t ]) Err.t
(** Checked byte arithmetic, an overflow reported against [id]. *)

val sub :
  id:Tensor_id.t ->
  Byte_size.t ->
  Byte_size.t ->
  (Byte_size.t, [> `Peak_bytes_overflow of Tensor_id.t ]) Err.t

type t

val of_graph : Arena_schedule.Config.t -> graph -> (t, [> error ]) Err.t
(** Validates the graph ([Graph_view.of_graph]) and dry-runs it once under the
    config's retention and alignment. Only [Intermediate] mode is supported
    here. *)

val graph : t -> graph

val node_count : t -> int
(** The number of nodes: a tally, bounded by [of_graph]. *)

val node : t -> Position.t -> node

val blocks : t -> Position.t -> Block.t list
(** What the node allocates, in output order; an unread index output is absent.
*)

val reads : t -> Position.t -> Block.t list
(** The allocated tensors the node reads as a non-sink, each once. *)

val preds : t -> Position.t -> Position.t list
(** Distinct producer nodes of every operand, sinks' included. *)

val readers : t -> Tensor_id.t -> int
(** Distinct non-sink readers: zero for an unread tensor. *)

val is_valid_order : t -> Position.t array -> (unit, [> error ]) Err.t
(** [Ok ()] iff the array is a permutation of all positions in which every node
    follows its predecessors. *)

val reorder : t -> Position.t array -> (graph, [> error ]) Err.t
(** The graph with its nodes in the given (validated) order. *)

val metrics :
  t -> Position.t array -> (Arena_schedule.Metrics.t, [> error ]) Err.t
(** The exact payload metrics of the order, by replaying releases against the
    facts: no dry run. *)

val lower_bound : t -> (Arena_schedule.Metrics.t, [> error ]) Err.t
(** Order-independent bounds, in {!Arena_schedule.Metrics.t}'s fields: the most
    any node must hold while it runs (its eligible outputs and eligible
    operands), per pool and combined. [outside_peak] and [all_peak] are likewise
    the per-node outside and total need. *)

val fresh_metrics :
  Arena_schedule.Config.t ->
  graph ->
  (Arena_schedule.Metrics.t, [> error ]) Err.t
(** The same figures from a fresh dry run of the graph, computed independently
    of {!metrics}: an exact event fold for the combined and outside peaks,
    [Alloc_script.peak_bytes] for the total, and [Interval_alloc.lower_bound]
    over each [Arena_problem.Kind_problem.script] for the pools. Never the
    padded [combined_bound_bytes]. *)
