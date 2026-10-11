(** The deterministic execution order of matched units and the lifetime of every
    canonical value.

    The dependency graph is the union of both endpoints' inter-unit data and
    control edges and of the canonical writer-to-reader edges. Each edge keeps
    the side, tensor, node or canonical value that produced it. Either graph can
    be acyclic while their union is not: that is reported with a witness cycle
    listing every edge's reason. *)

open Inference_compare_schema

module Edge : sig
  type t = Plan_error.Cycle_edge.t = {
    from_unit : Id.Unit.t;
    to_unit : Id.Unit.t;
    reason : Plan_error.Edge_reason.t;
  }
end

module Lifetime : sig
  type read = {
    unit : Id.Unit.t;
    bindings : int;  (** Input-binding recipes reading the value. *)
    checks : int;  (** Reconstruction checks of the value (both sides). *)
  }

  type t = {
    canonical : Id.Canonical_value.t;
    writer : Id.Unit.t option;  (** [None] for an initial value. *)
    reads : read list;  (** In plan order. *)
    final_uses : int;  (** Final output occurrences returning the value. *)
    total_uses : int;  (** Every read, check and final use. *)
  }
end

type t

val compute :
  limits:Resource_limits.t ->
  left:Graph_index.t ->
  right:Graph_index.t ->
  partition:Partition_check.t ->
  correspondence:Correspondence_check.t ->
  (t, Plan_error.t) Err.t

val order : t -> Id.Unit.t list

val edges : t -> Edge.t list
(** Sorted by source unit, target unit and reason, one entry per distinct
    reason. *)

val predecessors : t -> Id.Unit.t -> Id.Unit.t list

val lifetimes : t -> Lifetime.t list
(** Sorted by canonical value. *)

val consumers : t -> Id.Canonical_value.t -> Id.Unit.t list
(** The units reading a canonical value, in plan order. *)
