(** The failure record layout shared with the C backend's [struct model_error]:
    [i32 kind] at 0, [i32 invocation] at 4, then [error_words] [i64] slots from
    byte 8, little-endian. *)

val error_words : int
val kind_offset : int
val invocation_offset : int

val slot_offset : int -> int
(** The byte offset of slot [k]. *)

val record_bytes : int

val kind_index : Loop_js_failure.Kind.t -> int
(** The position of the kind in [Loop_js_failure.Kind.all]. *)
