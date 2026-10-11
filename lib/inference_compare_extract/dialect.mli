(** Dialect tooling: building an engine's own ordinary graph from a planned
    extraction. Distinct from the runner that prepares and executes it, so two
    engines on one dialect can share graph bodies. *)

open Inference_compare_schema

module type S = sig
  type graph
  (** The dialect's full graph, with every attribute and constant. *)

  type body
  (** An extracted ordinary graph in the same dialect. *)

  val dialect : Id.Dialect.t
  val describe : graph -> Graph_desc.t

  val extract : graph -> Extraction.t -> (body, Extract_error.t) Err.t
  (** Realizes the planned extraction natively: same operations and attributes,
      external operands as inputs, declared results as outputs. *)

  val describe_body : body -> Graph_desc.t

  val validate : body -> (unit, Extract_error.t) Err.t
  (** The dialect's own graph validator. *)
end

val realize :
  (module S with type graph = 'g and type body = 'b) ->
  'g ->
  Extraction.t ->
  ('b, Extract_error.t) Err.t
(** [extract], then the dialect's [validate], then {!Reconcile.check} of the
    body's description against the plan. No other path yields a body. *)
