(** The deterministic binary encoder. [module_] validates with {!Wasm_check}
    first, then writes the spec's binary format: sections in the spec's order,
    function types interned in order of first use (imports, then functions),
    empty sections omitted, locals run-length compressed, and an [if] without an
    else branch when the else list is empty. The same module value always yields
    the same bytes, on every backend: immediates go through [int32] and [int64],
    never a 63-bit [int]. *)

val module_ : Wasm.Module.t -> (string, Wasm_check.error) Err.t

(** {1 Primitives, exposed for tests} *)

val uleb128 : Buffer.t -> int64 -> unit
(** The value is read as unsigned. *)

val sleb128 : Buffer.t -> int64 -> unit
