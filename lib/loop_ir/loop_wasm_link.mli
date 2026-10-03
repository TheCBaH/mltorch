(** Fixes the function index space of a module whose bodies call symbolic
    {!Loop_wasm_runtime.Callee}s: the reached [Math] imports first, then the
    reached helpers, then the given functions. *)

val reached : Loop_wasm_runtime.Callee.t list -> Loop_wasm_runtime.Callee.t list
(** The callees closed under the helpers' own calls, in [Callee.all] order. *)

val kernel_call : int -> Wasm.Instr.t
(** A call to the [k]th function given to {!link}, symbolic like a callee, so
    functions that call each other are renumbered together. *)

val link :
  callees:Loop_wasm_runtime.Callee.t list ->
  Wasm.Func.t list ->
  Wasm.Import.t list * Wasm.Func.t list * int
(** [link ~callees fs] is the imports, every function in index order (helpers,
    then [fs] renumbered), and the index of the first of [fs]. [callees] must
    cover everything [fs] call. *)

val manifest :
  ?numerics:Loop_numerics.t ->
  ?precisions:Loop_numerics.Precision.t list ->
  callees:Loop_wasm_runtime.Callee.t list ->
  Wasm.Module.t ->
  string
(** The text of the module's [manifest] custom section: ABI, required features,
    imports, helpers and numeric policy. [numerics] (reference by default) and
    the working [precisions] of the module's kernels are part of it: a binary64
    module under the reference policy prints the text it always has, any other
    names the policy and the precisions it uses, so two numerical plans never
    share a manifest, and so never an identity. *)
