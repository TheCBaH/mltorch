(** What a kernel asks of the registers and what its allocation pays, as counts:
    the feedback the structured SSA blocking reads. *)

open Machine_ir

type t = {
  peak : (Mir_target.Bank.t * int) list;
      (** per register bank, the most values live at one position *)
  hot_stores : (Mir_target.Bank.t * int) list;
      (** per bank, register to frame in a block on a cycle *)
  hot_loads : (Mir_target.Bank.t * int) list;
      (** per bank, frame to register in a block on a cycle *)
  by_depth : (int * int) list;
      (** per loop depth from 1, ascending: the hot stores and loads of every
          bank in blocks that many natural loops contain *)
  innermost : int;
      (** the hot stores and loads in blocks of a loop holding no other *)
  stats : Mir_alloc_stats.t;
  frame : int64 option;  (** the largest realized frame, when realized *)
  helper_calls : int;
}

val hot_spills : t -> int
(** Hot stores and loads together. *)

val pp_depths : Format.formatter -> (int * int) list * int -> unit
val pp : Format.formatter -> t -> unit

module Make (T : Mir_sel.TARGET) : sig
  val report :
    Mir_sel.Make(T).Verified.t -> (T.op, T.test) Mir_phys.Program.t -> t
end
