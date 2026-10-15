(** A census of the binary64 instructions of a module, by role: how much of a
    binary32 module still computes in binary64, and what it imports.

    Wasm is typed, so an instruction is counted when its mnemonic or its operand
    type is [f64] ([f64x2] included). Counted before encoding, over the module
    an emitter builds. [Select] and [Drop] are untyped and not counted; the
    producing instruction is. *)

module Role : sig
  type t =
    | Arithmetic  (** [f64.add], [f64x2.mul], comparisons, [f64.sqrt], ... *)
    | Conversion
        (** [f64.convert_*], [f64.promote_f32], [f32.demote_f64], [trunc_sat] *)
    | Helper_call  (** a call to a function with an [f64] parameter or result *)
    | Memory_move
        (** [f64.const], loads, stores, locals and globals of [f64], lane
            access, reinterprets *)

  val all : t list
  val name : t -> string
end

type t

val of_module : Wasm.Module.t -> t

val role_of_op : Wasm_op.t -> Role.t option
(** The role of an operation that touches [f64]; [None] for any other. Total: a
    new operation does not compile until it is classified. *)

val count : t -> Role.t -> int
val total : t -> int

val helpers : t -> (string * int) list
(** The callees counted under [Helper_call]: ["module.name"] for an import,
    ["funcK"] for the module's [K]th own function. *)

val imports : t -> string list
(** Every import of the module, ["module.name"], in order. *)

val pp : Format.formatter -> t -> unit
