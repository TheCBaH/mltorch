(** Plan, extract, resolve policy, seal and run one pair of in-memory endpoints
    through the coordinator: the diagnostic counterpart of {!Session} and
    {!Run}, for a pair assembled by a driver rather than read from an experiment
    file. Nothing here prints or exits; the caller maps the result to the
    command's exit code. *)

open Inference_compare_schema

val limits : Resource_limits.t
(** Generous limits for a whole model, which an experiment file would declare.
*)

val site_policy : tolerance:float -> Comparison_site.t -> Comparison_policy.t
(** The policy a comparison site takes from its canonical element type: exact
    for booleans and integers (any difference disagrees, and an integer is
    compared as its bytes), floating at [tolerance] for floats, with the
    arithmetic of the format. *)

val selection : tolerance:float -> Comparison_site.t list -> Policy_selection.t
(** One {!site_policy} per site. *)

type outcome = {
  result : Inference_compare_core.Coordinator.result;
  ctx : Inference_compare_codecs.Logical_value.Context.t;
      (** The run's context, which owns [result.finals]. *)
  extracted : int;  (** The graphs extracted, one per unit and side. *)
}

val run :
  on_event:(Event.t -> unit) ->
  donor:Side.t ->
  tolerance:float ->
  mapping:Mapping_artifact.t ->
  left_graph:Graph_desc.t ->
  right_graph:Graph_desc.t ->
  initial:(Id.Canonical_value.t * Inference_compare_codecs.Logical_value.t) list ->
  node_attributes:(Side.t -> Node_ref.t -> string) ->
  left_endpoint:Inference_compare_core.Side_exec.endpoint ->
  right_endpoint:Inference_compare_core.Side_exec.endpoint ->
  (outcome, Session.error) result
(** An [Error] is found before any engine runs: an invalid mapping, extraction,
    policy or seal. *)

val error_exit_code : int
(** 2: invalid input, map or policy. *)

val exit_code : Inference_compare_core.Coordinator.result -> int
(** As {!Run.exit_code}. *)
