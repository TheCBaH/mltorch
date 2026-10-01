(** The baseline route: the C backend's whole-model unit ({!Loop_bundle_c})
    compiled to WebAssembly with Clang against a wasm32 C library, then run
    under node over the same payload files as the direct backend. It exists to
    compare the direct emitter with: both run the same schedule and storage plan
    in the same engine, so a difference is the code generator's. It needs an
    external toolchain, unlike the direct route.

    The toolchain is the installed Clang and [wasm-ld] plus a wasm32 libc and
    compiler runtime ([scripts/wasi-sysroot-userland.py]); a missing piece is an
    error naming it, never a skip. The compile is scalar by construction: loop
    and SLP vectorization, SIMD and FP contraction are disabled, and the module
    is scanned for vector instructions rather than trusted to be free of them.
*)

open Graph_ir
open Loop_ir

type toolchain = {
  clang : string list;
      (** the compiler and the flags every compilation and the link share,
          including [--target] and [--sysroot] *)
  builtins : string;  (** the wasm32 [libclang_rt.builtins] archive *)
}

type prepared

type error =
  [ `Generate_c of Loop_bundle_c.error
  | `Toolchain_missing of string
  | `Compile_failed of Loop_c_exec.Proc.status * string
  | `Missing_constant of Graph_ir.Tensor_id.t
  | `Missing_input of Graph_ir.Tensor_id.t
  | `Binding_mismatch of Kernel_eval.Binding_mismatch.t
  | `Io of string
  | `Node_unavailable of string
  | `Run_failed of Loop_c_exec.Proc.status * string
  | `Inference_failed of int * Loop_interp.error
  | `Bad_failure_record of string
  | `Bad_output of string
  | `Memory_over_policy of int64 ]

val pp_error : Format.formatter -> [< error ] -> unit

val toolchain_from_env : unit -> (toolchain, error) result
(** From [MLTORCH_WASI_SYSROOT] (the [usr] directory
    [scripts/wasi-sysroot-userland.py] prints) and [MLTORCH_WASM_CLANG] (default
    [clang]). *)

val scalar_flags : string list
(** [-O2 -ffp-contract=off -fno-strict-aliasing -fno-vectorize
     -fno-slp-vectorize -mno-simd128]: the strict scalar build. *)

val prepare :
  ?flags:string list ->
  toolchain:toolchain ->
  dir:string ->
  Loop_bundle.t ->
  constants:(Tensor_id.t -> Tensor.packed option) ->
  (prepared, error) Err.t
(** Generates [model_infer.c] and a small export wrapper, compiles and links
    them with [flags] (default {!scalar_flags}), sizes the memory, and writes
    [weights.bin]. [dir] is kept. *)

val run :
  ?poison:bool ->
  ?repeat:int ->
  prepared ->
  bind:(Tensor_id.t -> Tensor.packed option) ->
  (Tensor.packed list, error) Err.t
(** As {!Wasm_host.run}: graph outputs in order, workspace and outputs poisoned
    on request, [repeat] more runs on the same instance requiring identical
    outputs. *)

val timings : prepared -> (string * float) list

type sizes = {
  source_bytes : int;
  module_bytes : int;
  memory_bytes : int;  (** the linear memory the placement needs *)
  heap_base : int;
  vector_instructions : int;
      (** [v128] instructions found in the module: [0] for a scalar build *)
}

val sizes : prepared -> sizes
val bundle_c : prepared -> Loop_bundle_c.t
val directory : prepared -> string
val module_path : prepared -> string
val weights_path : prepared -> string
