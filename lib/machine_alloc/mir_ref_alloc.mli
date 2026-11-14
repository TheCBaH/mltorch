(** The reference allocator: every value in a frame slot, operands reloaded
    around each instruction. *)

open Machine_ir

module type REGISTERS = sig
  val use_scratch :
    Mir_target.Bank.t -> bits:Mir_target.Bits.t -> int -> Mir_target.View.t
  (** the register for an instruction's [k]-th distinct operand *)

  val result_scratch :
    Mir_target.Bank.t -> bits:Mir_target.Bits.t -> Mir_target.View.t

  val copy_scratch :
    Mir_target.Bank.t -> bits:Mir_target.Bits.t -> Mir_target.View.t

  val cycle_scratch :
    Mir_target.Bank.t -> bits:Mir_target.Bits.t -> Mir_target.View.t

  val args : Mir_type.t list -> Mir_target.View.t list
  (** where a function's parameters arrive *)

  val results : Mir_type.t list -> Mir_target.View.t list
end

(* Fault injection for the evidence suite: one deliberate allocation defect
   each, which the checker or interpreter must catch. No consumer passes one. *)
module Mutation : sig
  type t =
    | Copy_placement  (** a branch's edge moves placed on the other edge *)
    | Cycle_scratch  (** a transfer cycle broken through the copy scratch *)
    | Reload_order
        (** an operand reloaded before the previous result's spill *)
    | Slot_reuse  (** a value spilled into the previous value's slot *)
    | Spill_width  (** a 64-bit value spilled at half width *)
    | Tie  (** a tied result written to another register *)
end

module Make (T : Mir_sel.TARGET) (_ : REGISTERS) : sig
  val allocate :
    ?mutation:Mutation.t ->
    Mir_sel.Make(T).Verified.t ->
    (T.op, T.test) Mir_phys.Program.t
end
