(** A census of the binary64 instructions of a verified generic program, by
    role: how much of a binary32 artifact still computes in binary64.

    An instruction is counted when an operand or its result is [f64] (or a
    vector of [f64]). Block parameters, edge arguments and function results are
    not instructions and are not counted. *)

module Role : sig
  type t =
    | Arithmetic  (** [Fbinary], [Ffma], [Funary], [Fcmp] *)
    | Conversion  (** [Fconvert], [Fto_sint] *)
    | Helper_call  (** a call whose argument or result is [f64] *)
    | Memory_move
        (** loads, stores, bitcasts, copies, selects, constants, vector lane
            moves *)

  val all : t list
  val name : t -> string
end

type t

val of_program : Mir_verify.Generic.t -> t

val count : t -> Role.t -> int
(** How many instructions of a role touch [f64]. *)

val helpers : t -> (string * int) list
(** The callees counted under [Helper_call], by name, ascending. *)

val total : t -> int
val pp : Format.formatter -> t -> unit
