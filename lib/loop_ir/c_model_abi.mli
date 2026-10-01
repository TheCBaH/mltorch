(** The ABI shared by the two generated C files, and the process contract of the
    standalone application. *)

val declarations : string
(** The [extern] requirement constants and the [model_run] prototype, emitted
    into both files after {!Loop_c_runtime.prelude}. *)

val exit_ok : int
val exit_usage : int
val exit_payload : int
val exit_allocation : int
val exit_inference : int

val error_line_prefix : string
(** The last stderr line of an inference failure begins with this, then the
    invocation, the kind and the failure record's words, decimal. *)
