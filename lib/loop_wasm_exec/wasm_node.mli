(** The node side of the Wasm hosts: one runner script, its command line and the
    classification of what it prints. Shared by the host for the direct backend
    ({!Wasm_host}) and the one for C compiled to Wasm, which both run a module
    over the same payload files. *)

open Loop_ir

val node : string list ref
(** The command that runs a script: [["node"]]. *)

val exit_inference : int
(** The runner's exit status for a failure record: [5]. *)

type placement = {
  weights : int;
  inputs : int;
  workspace : int;
  outputs : int;
  workspace_bytes : int64;
  outputs_bytes : int64;
}
(** Where the four regions go in the module's memory. *)

val execute :
  dir:string ->
  module_file:string ->
  weights:string ->
  inputs:string ->
  template:string ->
  outputs:string ->
  placement ->
  poison:bool ->
  repeat:int ->
  (Loop_c_exec.Proc.status * string, string) result
(** Instantiates the module (entry [model_run], or [run] with an [error_ptr]
    export, as a C-compiled module has), places the payload files, optionally
    poisons the workspace and outputs, runs the schedule, repeats it [repeat]
    more times on the same instance requiring identical outputs, and writes the
    outputs region to [outputs]. The log carries the failure record or the
    timing line. [Error] only if [node] could not be started. *)

val parse_failure :
  Loop_bundle.t ->
  string ->
  ( 'a,
    [> `Bad_failure_record of string
    | `Inference_failed of int * Loop_interp.error ] )
  result
(** The failing invocation and the interpreter's row for its failure, from the
    runner's [model_error:] line. *)

val timing_of : string -> (string * float) list
(** The phases the runner reported, in milliseconds. *)
