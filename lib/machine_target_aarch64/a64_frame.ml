(* What frame realization uses on AArch64: x16 (IP0) for a large offset's
   address, x17 (IP1) for FPCR in transit — both reserved, so they never hold
   an allocated value — ADD (immediate) for the address, and MRS/MSR FPCR with
   a zero to establish round-to-nearest-even, no flush-to-zero, no default NaN
   and no traps. *)

open Machine_ir
open A64_op

type op = A64_op.t

let scratch = A64_reg.x 16
let control_scratch = A64_reg.x 17
let add_imm base k = Add_imm (Sz.X, base, k)

let fp_control =
  Some
    ( A64_reg.fpcr,
      (fun v -> Mrs_fpcr v),
      (fun v -> Msr_fpcr v),
      Movz (Mir_type.i64, 0, 0) )
