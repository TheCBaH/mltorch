(* Memory-aware topological scheduling: the contract. A schedule is a
   dependency-valid permutation of [Graph.nodes] chosen to lower the live
   tensor payload; nothing else of the graph changes. The original order is
   always the incumbent. See .ai/ on the arena scheduling design.

   This module owns the vocabulary (mode, configuration, metrics, stop reasons,
   results) and the identity schedule; the replay engine, the constructive
   policies and the beam search build on it. *)

open Graph_ir
open Core.Storage_units

(** What the schedule's lifetimes are measured against. [Intermediate]: the
    evaluator's own allocations ([Eval_direct.dry_run]), pooled by element kind.
    [Roles]: every block of a storage-role run ([Eval_direct.storage_script]
    under the given layout and ownership), pooled by arena and kind. *)
module Mode : sig
  type t = Intermediate | Roles of Storage_script.Config.t

  val equal : t -> t -> bool
  val pp : Format.formatter -> t -> unit
end

(** A pool whose peak is reported: [arena] is [None] in [Intermediate] mode. *)
module Pool : sig
  type t = {
    arena : Storage_script.Arena_id.t option;
    kind : Alloc_script.Kind.t;
  }

  val equal : t -> t -> bool
  val compare : t -> t -> int
  val pp : Format.formatter -> t -> unit
end

(** Search bounds. [width] and [expansions] bound the beam; [expansions = 0]
    runs the constructive schedules only. [state_bytes] caps the estimated size
    of the retained search states. *)
module Limits : sig
  type t = private { width : int; expansions : int; state_bytes : Byte_size.t }
  type error = [ `Invalid_limit of [ `Expansions | `State_bytes | `Width ] ]

  val make :
    width:int ->
    expansions:int ->
    state_bytes:Byte_size.t ->
    (t, [> error ]) Err.t
  (** [width >= 1] and [expansions >= 0]. *)

  val constructive_only : t
  (** No beam: width 1, no expansions. *)

  val default_beam : t
  (** An explicit beam request: width 8, 100000 expansions, 256 MiB of search
      state. Provisional, pending corpus cost evidence. *)

  val equal : t -> t -> bool
end

module Config : sig
  type t = {
    mode : Mode.t;
    retain : Release_schedule.Retain.t;
        (** The effective retention of the run the schedule will serve: the
            scheduler never defaults it. *)
    alignment : Alignment_policy.t;
    limits : Limits.t;
  }
end

(** The three payload figures that do not imply one another, plus the outside
    and all-output ones. Exact bytes throughout; alignment padding never enters.
*)
module Metrics : sig
  type t = {
    target_peak : Byte_size.t;
        (** The most eligible payload live at once, over every pool. *)
    pool_peaks : (Pool.t * Byte_size.t) list;
        (** Each pool's own peak, in {!Pool.compare} order. *)
    pool_peak_sum : Byte_size.t;  (** [pool_peaks] summed. *)
    outside_peak : Byte_size.t;
        (** The most payload live at once that no pool backs. *)
    all_peak : Byte_size.t;  (** The most of every block live at once. *)
  }

  val equal : t -> t -> bool
  val pp : Format.formatter -> t -> unit
end

(** Independent work counters: never one number standing for several costs. *)
module Stats : sig
  type t = {
    score_evaluations : int;
    expansions : int;
    depth : int;
    max_ready_width : int;
    retained_states : int;
    state_bytes : Byte_size.t;
  }

  val zero : t
end

module Stop : sig
  type t = Budget_exhausted | Completed | Lower_bound_reached | State_limit
end

module Strategy : sig
  type t = Beam | Identity | Live_first | Peak_first

  val pp : Format.formatter -> t -> unit
end

type error =
  [ `Duplicate_node of Node_id.t
  | `Graph of Graph_view.error
  | `Invalid_limit of [ `Expansions | `State_bytes | `Width ]
  | `Not_a_permutation
  | `Structure_changed ]

val pp_error : Format.formatter -> [< error ] -> unit

val same_structure : graph -> graph -> bool
(** [true] iff the graphs differ at most in the order of [Graph.nodes]: the same
    nodes (id, op, outputs) as a set, and identical root group, tensors, inputs,
    input kinds and outputs. An order digest could not establish this. *)

val check_permutation : original:graph -> graph -> (unit, [> error ]) Err.t
(** [Ok ()] iff [graph] is a validated, dependency-valid reordering of
    [original] ({!same_structure} and {!Graph_view.of_graph}). *)

module Result : sig
  type t = {
    graph : graph;  (** The selected graph: execution must use this one. *)
    strategy : Strategy.t;  (** Which candidate produced [graph]. *)
    stop : Stop.t;
    stats : Stats.t;
  }
end

val identity : graph -> (Result.t, [> error ]) Err.t
(** The original order, validated, as the incumbent every search starts from. *)
