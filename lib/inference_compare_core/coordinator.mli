(** The shared-frontier loop.

    For each unit, in the plan's deterministic order, both engines receive the
    same canonical operands. Before either graph runs, the imported handles are
    read back and reconstructed; any difference from the common value stops the
    unit. Both graphs then run on their own ordinary extracted graph, both
    computed output bundles are snapshotted and compared under the sealed
    per-site policies, and only after the comparison does the donor's candidate
    for each canonical output become the common value for both engines'
    downstream inputs. The unit's frontier is staged, recorded in the event
    stream and then published all at once: an execution, transfer, resource or
    evidence failure at any step publishes nothing and never substitutes the
    other side's answer. A numerical mismatch does not stop the run: the
    labelled diagnostic trajectory continues so propagation can be measured, and
    the summary keeps agreement, coverage and execution separate. *)

open Inference_compare_schema
module Codecs = Inference_compare_codecs

(** What the host may retain about one unit. All values are borrowed: a retainer
    copies or serializes them before returning. *)
type stage =
  | Before_commit
      (** Everything of the unit exists and has been compared; nothing is
          published yet. A refusal here fails the unit. *)
  | After_failure
      (** The unit stopped; only the parts that existed are present. The result
          is ignored. *)

type snapshot = {
  unit : Id.Unit.t;
  stage : stage;
  common_inputs :
    (Id.Canonical_value.t * Id.Value_instance.t * Codecs.Logical_value.t) list;
      (** The canonical values the unit read, as both engines saw them. *)
  raw :
    (Side.t * Tensor_ref.t * Id.Value_instance.t * Codecs.Logical_value.t) list;
      (** Each side's computed output snapshots, original tensor ids. *)
  candidates :
    (Id.Canonical_value.t
    * Side.t
    * Id.Value_instance.t
    * Codecs.Logical_value.t)
    list;
  mismatching : bool;  (** Some comparison of the unit was not a pass. *)
}

type mode =
  | Full  (** Every unit, from the pinned initial payloads. *)
  | Single_unit of {
      unit : Id.Unit.t;
      inputs : (Id.Canonical_value.t * Codecs.Logical_value.t) list;
    }  (** Replay one unit from retained common inputs; no final outputs. *)

type config = {
  ctx : Codecs.Logical_value.Context.t;
      (** Its ledger bounds every live payload of the run. *)
  codecs : Codecs.Element_codec.Registry.t;
  operations : Codecs.Conversion_registry.t;
  metrics : Inference_compare_metrics.Tensor_metrics.env;
  hashers : Codecs.Payload_hasher.Registry.t;
  run_id : Id.Run.t;
  case_id : Id.Case.t;
  experiment : Experiment.t;
  initial : (Id.Canonical_value.t * Codecs.Logical_value.t) list;
      (** The pinned initial payloads; read, never consumed. *)
  expected_identities : Engine_identity.t option * Engine_identity.t option;
  emit : Event.t -> (unit, string) result;
      (** The event sink. A refused event is a failure of the unit. *)
  retain : snapshot -> (unit, string) result;
  mode : mode;
}

type result = {
  completion : Completion.t;
  failure : Failure.t option;
  comparisons : Comparison_result.t list;  (** In emission order. *)
  commits : (Id.Unit.t * Committed_value.t list) list;
  finals : (Id.Output_occurrence.t * Codecs.Logical_value.t) list;
      (** Owned by the caller; present only for a complete run. *)
  executed : Id.Unit.t list;  (** Units whose frontier was published. *)
  lineage : (Id.Value_instance.t * Value_origin.t) list;
      (** Every value instance of the run, in creation order: initial payloads,
          raw computed outputs, converted candidates (with the raw instances
          their recipes read) and committed frontier values (with the selected
          candidate as parent). *)
  events : Event.t list;  (** Oldest first. *)
}

val run :
  config ->
  Sealed_plan.t ->
  left:Side_exec.endpoint ->
  right:Side_exec.endpoint ->
  result
(** Runs only a sealed plan: its policy bindings are the ones every donor pass
    shares. *)
