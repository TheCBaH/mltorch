(* The semantics of each admitted AArch64 form, from the Arm ARM pseudocode: the
   interpreter's [exec] and [test]. The per-form helpers stay private. *)

val exec :
  Machine_interp.Mir_sel_env.t -> A64_op.t -> Machine_interp.Mir_datum.t list
(** The values the form writes, in result order. *)

val test : Machine_interp.Mir_sel_env.t -> A64_op.test -> bool
(** Whether a condition holds on the flags. *)
