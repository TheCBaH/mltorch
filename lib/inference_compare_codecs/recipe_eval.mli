(** Static checking and evaluation of conversion recipes.

    {!check} resolves a recipe against the specs it may read, infers every
    node's output spec, and decides whether the recipe is lossless from the cast
    laws and the operation registry, not from the recipe's own flag. {!evaluate}
    runs a checked recipe over actual values, releasing each intermediate at its
    last use, and reports the lineage of every node. *)

open Inference_compare_schema

type env = {
  codecs : Element_codec.Registry.t;
  operations : Conversion_registry.t;
  canonical : Id.Canonical_value.t -> Tensor_spec.t option;
  boundary : Tensor_ref.t -> Tensor_spec.t option;
}

module Checked : sig
  type t

  val recipe : t -> Recipe.t

  val order : t -> Recipe.node list
  (** Reachable nodes, dependencies first: a depth-first post-order from the
      result that visits arguments in their declared order, so equal recipes
      give equal orders. Nodes the result does not reach are omitted. *)

  val node_spec : t -> Id.Conversion_node.t -> Tensor_spec.t option
  val result_spec : t -> Tensor_spec.t

  val lossless : t -> bool
  (** Computed, not declared: every step preserves every value. *)

  val canonical_reads : t -> Id.Canonical_value.t list

  val boundary_reads : t -> Tensor_ref.t list
  (** Distinct arguments the reachable nodes (and result) read, sorted. *)
end

val check : env -> Recipe.t -> (Checked.t, Codec_error.t) Err.t

module Lineage : sig
  type source =
    | Canonical_input of Id.Canonical_value.t * Digest.Logical_value.t
    | Boundary_input of Tensor_ref.t * Digest.Logical_value.t
    | Node_result of Id.Conversion_node.t

  type entry = {
    node : Id.Conversion_node.t;
    operation : Conversion_step.t;
    sources : source list;
    result : Digest.Logical_value.t;
  }

  type t = entry list
end

type inputs = {
  canonical : Id.Canonical_value.t -> Logical_value.t option;
  boundary : Tensor_ref.t -> Logical_value.t option;
}

val evaluate :
  Logical_value.Context.t ->
  env ->
  Checked.t ->
  inputs ->
  (Logical_value.t * Lineage.t, Codec_error.t) Err.t
(** The result is a fresh value the caller owns, even when the recipe is a bare
    passthrough. Intermediates are dropped on every path, success or failure; a
    ledger limit in the context bounds their live bytes. *)
