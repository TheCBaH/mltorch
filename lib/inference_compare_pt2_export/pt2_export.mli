(** Writing an experiment directory for one exported model and one sample. *)

open Inference_compare_schema

type right =
  | Native  (** The Native direct evaluator. *)
  | Native4d
      (** The four-axis evaluator, after the canonical passes; a model outside
          its domain is an error, never the Native route. *)

type certify = {
  tested : Side.t;  (** The reference is the other side. *)
  fidelity : Partition_fidelity.t;
}
(** Also write a computation contract and a validation workflow for the two-pass
    certification: ATen on one side, the [right] engine on the other. *)

val write_experiment :
  ?right:right ->
  ?certify:certify ->
  dir:string ->
  model:string ->
  inputs:string ->
  index:int ->
  tolerance:float ->
  unit ->
  (unit, string) result
(** [dir/experiment.json] and the files it names, for the ATen interpreter on
    the left and the [right] evaluator (default [Native]) on the right, both
    pinned to [model] by SHA-256. [inputs] is a functional-release input map and
    [index] picks its sample, to which a batch axis is added. The model is
    recorded by absolute path. Every floating site is compared in binary32
    within [tolerance]. *)
