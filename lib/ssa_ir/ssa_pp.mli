(** The deterministic text of a program: definitions are numbered by their
    position in the structure, so the order in which a builder happened to
    allocate ids, and the ids themselves, never appear. A diagnostic format
    only: nothing parses it. *)

(** The printing a second form of the program shares: the same value naming,
    operation text and buffer lines. *)
module Parts : sig
  type state

  val create : unit -> state
  val line : state -> int -> ('a, Format.formatter, unit, unit) format4 -> 'a
  val buffer : state -> Ssa_buffer.t -> unit
  val defs : state -> Ssa_value.t list -> string
  val uses : state -> Ssa_value.t list -> string
  val name : state -> Ssa_value.t -> string
  val instr : state -> int -> Ssa_instr.t -> unit
  val contents : state -> string
end

val to_string : Ssa_program.t -> string
val pp : Format.formatter -> Ssa_program.t -> unit
