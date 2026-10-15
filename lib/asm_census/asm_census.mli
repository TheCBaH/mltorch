(** A census of the binary64 instructions of assembly text, by role.

    An instruction is counted when it names a binary64 register or arrangement
    (AArch64: a [d0]..[d31] operand, or a [.2d]/[.1d]/[.d] arrangement) or is a
    double-precision mnemonic (x86-64: the [sd] and [pd] families, the
    conversions to and from them, and [movq] with an [%xmm] operand).
    Directives, labels, comments and integer instructions are skipped. A
    floating-point mnemonic that is in no table is an {!error}, never skipped.
*)

module Isa : sig
  type t = Aarch64 | X86_64

  val name : t -> string
end

module Role : sig
  type t =
    | Arithmetic  (** add, mul, sqrt, compare, rounding, logic on doubles *)
    | Conversion  (** between precisions, and to or from integers *)
    | Memory_move  (** register moves, loads, stores, constants, selects *)

  val all : t list
  val name : t -> string
end

type t
type error = [ `Unknown_fp_mnemonic of Isa.t * string ]

val pp_error : Format.formatter -> [< error ] -> unit
val of_text : Isa.t -> string -> (t, [> error ]) Err.t

val count : t -> Role.t -> int
(** How many instructions of a role touch binary64. *)

val total : t -> int

val mnemonics : t -> (string * int) list
(** The counted mnemonics with their counts, ascending: what a failing gate
    prints. *)

val pp : Format.formatter -> t -> unit
