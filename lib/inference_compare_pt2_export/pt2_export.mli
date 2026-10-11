(** Writing an experiment directory for one exported model and one sample. *)

val write_experiment :
  dir:string ->
  model:string ->
  inputs:string ->
  index:int ->
  tolerance:float ->
  (unit, string) result
(** [dir/experiment.json] and the files it names, for the ATen interpreter on
    the left and the Native evaluator on the right, both pinned to [model] by
    SHA-256. [inputs] is a functional-release input map and [index] picks its
    sample, to which a batch axis is added. The model is recorded by absolute
    path. Every floating site is compared in binary32 within [tolerance]. *)
