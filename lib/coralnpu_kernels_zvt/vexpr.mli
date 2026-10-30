(** Binary32 expressions over vector registers: {!F32_math.Make} instantiated to
    build a graph of operations, then emitted as kernel items with registers
    allocated by last use. One graph is one instance of [exp] (or any other
    sequence of {!F32_math.OPS}) over a vector of [vl] lanes, so the vector unit
    computes the very operations the oracle's {!F32_math.Value} does, each
    rounded to binary32 and none fused. *)

type t
(** A node of the graph. A value used twice is one node, evaluated once. *)

val input : int -> t
(** The [k]th input of an emission: a register the caller loaded. *)

module Ops : F32_math.OPS with type t = t

(** What an emission produced. *)
module Plan : sig
  type t = {
    items : Machine_rivet_riscv32.Rv32_kernel_item.t list;
    result : Machine_rivet_riscv32.Rv32_kernel_reg.Vreg.t;
        (** the register holding the value, no longer on the free list *)
    peak : int;  (** the most register groups live at once *)
  }
end

val emit :
  lmul:int ->
  scratch:Machine_rivet_riscv32.Rv32_kernel_reg.Gpr.t ->
  free:Machine_rivet_riscv32.Rv32_kernel_reg.Vreg.t list ->
  inputs:
    (Machine_rivet_riscv32.Rv32_kernel_reg.Vreg.t * [ `Consumed | `Kept ]) array ->
  t ->
  (Plan.t, string) result
(** Items computing the graph under the vector configuration in force ([lmul] is
    its register grouping: 1, 2, 4 or 8). [inputs.(k)] holds {!input} [k]; a
    [`Consumed] input's register joins the free list after its last use, a
    [`Kept] one is never written. [free] are the registers (each a multiple of
    [lmul], never [v0], which the compares write) the emission may use.
    [scratch] holds a constant's bits. A constant is built at each use, in a
    register of its own. An error says what ran out. *)
