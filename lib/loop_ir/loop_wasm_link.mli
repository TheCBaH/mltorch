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
  callees:Loop_wasm_runtime.Callee.t list -> Wasm.Module.t -> string
(** The text of the module's [manifest] custom section: ABI, required features,
    imports, helpers and numeric policy. *)
