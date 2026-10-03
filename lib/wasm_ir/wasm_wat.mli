(** A readable rendering of a module for debugging and goldens: the text
    format's mnemonics in a flat, indented layout. It is a dump, not a
    round-trippable WAT: constants print their exact bits next to the value. *)

val pp : Format.formatter -> Wasm.Module.t -> unit
val to_string : Wasm.Module.t -> string
