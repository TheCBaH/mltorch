(** The failure record layout shared with the C backend's [struct model_error]:
    [i32 kind] at 0, [i32 invocation] at 4, then [error_words] [i64] slots from
    byte 8, little-endian. *)

val error_words : int
val kind_offset : int
val invocation_offset : int

module Slot : Core.Tagged_int.S
(** The ordinal of an [i64] slot of the record. *)

val slot_arg : Slot.t -> Wasm.Mem_arg.t
(** The [i64] store or load of a slot, at its byte offset in the record. *)

val record_bytes : int

val kind_index : Loop_js_failure.Kind.t -> int
(** The position of the kind in [Loop_js_failure.Kind.all]. *)
