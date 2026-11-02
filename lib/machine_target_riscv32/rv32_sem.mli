(* The semantics of each admitted RISC-V form, from the unprivileged ISA: the
   interpreter's [exec] and [test]. The per-form helpers stay private. *)

val exec :
  Machine_interp.Mir_sel_env.t -> Rv32_op.t -> Machine_interp.Mir_datum.t list
(** The values the form writes, in result order. *)

val test : Machine_interp.Mir_sel_env.t -> Rv32_op.test -> bool
(** Whether a branch condition holds. *)

val fres : float -> int64
(** A binary32 result as its bit pattern: rounded to nearest even, a NaN
    canonical. Exposed so a conformance mutation can build a faulty reference
    from the same rounding. *)
