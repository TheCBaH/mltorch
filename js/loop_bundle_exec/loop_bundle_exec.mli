(** Runs a [Loop_bundle.t] as one JavaScript entry call over prepared pools
    (plan W4). [prepare] compiles the bundle once, allocates the constants pool
    and [max_outstanding] execution arena sets, and copies the constants in;
    [run_leased] copies the inputs into a free set, makes the single entry call
    and leaves the outputs in that set under a lease.

    Scope: [Copied] constants and inputs, and no graph output that is a graph
    input or constant. Anything else is refused at [prepare], never during a
    run. js_of_ocaml only. *)

open Graph_ir
open Loop_ir

type prepared
type lease

type error =
  [ Loop_bundle_js.error
  | Loop_js_exec.error
  | `Invocation_failed of Node_id.t * Loop_js_exec.error
  | `No_free_arena
  | `Released
  | `Unknown_output of Tensor_id.t
  | `Unsupported_forwarded_output of Tensor_id.t ]

val pp_error : Format.formatter -> [< error ] -> unit

val prepare :
  ?max_outstanding:int ->
  ?kernel:(Loop_bundle.invocation -> (Js_ast.Program.t, string) result) ->
  Loop_bundle.t ->
  constants:(Tensor_id.t -> Tensor.packed option) ->
  (prepared, error) Err.t
(** [max_outstanding] (default 1, at least 1) execution arena sets exist, each
    holding every pool but the constants' (which all share one copy). A run pins
    one set until its lease is released. *)

val run_leased :
  prepared -> bind:(Tensor_id.t -> Tensor.packed option) -> (lease, error) Err.t
(** Inputs are validated against the buffer signature a consuming kernel
    declares before anything executes. [`No_free_arena] when every set is
    running or leased. A failed run frees its set. The result stays in the
    leased set, unchanged by later runs, until [release]. *)

val output : lease -> Tensor_id.t -> (Tensor.packed, error) Err.t
(** A fresh copy of one output; [`Released] after [release]. *)

val release : lease -> unit
(** Idempotent; frees the set for the next run. *)

val run :
  prepared ->
  bind:(Tensor_id.t -> Tensor.packed option) ->
  (Tensor.packed Tensor_id.Map.t, error) Err.t
(** [run_leased], copy every output out, release. *)

type stats = {
  invocations : int;
  distinct_kernels : int;
  source_bytes : int;  (** the printed wrapper plus every kernel *)
  pools : (Storage_script.Arena_id.t * Alloc_script.Kind.t * int) list;
      (** every pool's element count; the constants' is resident once, every
          other is allocated once per execution set *)
  execution_sets : int;
}

val stats : prepared -> stats
