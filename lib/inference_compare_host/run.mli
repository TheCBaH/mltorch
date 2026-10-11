(** Running a prepared experiment and writing its evidence. *)

open Inference_compare_schema

type outcome = {
  result : Inference_compare_core.Coordinator.result;
  directory : string;
  finalized : (unit, Evidence.error) result;
      (** Whether the completion record was written. When it is an error the
          directory is incomplete and must not be read as a result. *)
}

val default_run_id : Session.prepared -> Id.Run.t
(** The experiment id and the first digits of the plan digest: the same
    experiment run twice collides, deliberately. *)

val config :
  Session.prepared ->
  emit:(Event.t -> (unit, string) result) ->
  retain:(Inference_compare_core.Coordinator.snapshot -> (unit, string) result) ->
  mode:Inference_compare_core.Coordinator.mode ->
  ledger_limit:int64 ->
  Inference_compare_core.Coordinator.config
(** The coordinator configuration for a prepared experiment. *)

val run :
  root:string ->
  ?run_id:Id.Run.t ->
  Session.prepared ->
  (outcome, Evidence.error) result

val exit_code : Inference_compare_core.Coordinator.result -> int
(** 0 complete agreement; 1 completed numerical disagreement; 2 an invalid map,
    policy or input found before execution; 3 a refused or failed preparation,
    transfer, execution or resource limit; 4 incomplete coverage or an agreement
    that could not be established. *)
