(* What frame realization uses on RV32: t6 for a large offset's address,
   computed by the wide add (a 4 KiB multiple needs [lui], which [addi] cannot
   reach). There is no FP control state to establish: a Linux process starts
   with round-to-nearest-even in frm, and no form writes it. *)

open Rv32_op

type op = Rv32_op.t

let scratch = Rv32_reg.scratch
let control_scratch = Rv32_reg.scratch
let add_imm base k = Add_wide (base, k)
let fp_control = None
