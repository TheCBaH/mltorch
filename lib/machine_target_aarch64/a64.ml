(* The AArch64 target as the selected stage, its verifier and interpreter
   consume it. *)
include A64_op

type op = t

let abi = A64_reg.abi
let flags_view = A64_reg.nzcv
let stack_pointer = A64_reg.sp
let call_push = 0L
let link = Some (A64_reg.x 30)
let address_step = function Add_imm (Sz.X, _, k) -> Some k | _ -> None

(* LDR/STR (unsigned offset): a multiple of the size, at most 4095 of them. *)
let frame_offset_ok ~bytes offset =
  Int64.compare offset 0L >= 0
  && Int64.compare bytes 0L > 0
  && Int64.equal (Int64.rem offset bytes) 0L
  && Int64.compare (Int64.div offset bytes) 4095L <= 0

(* ADD/SUB SP, SP, #imm12 or #imm12, LSL 12, keeping SP 16-byte aligned: an
   access based on a misaligned SP faults (SCTLR_EL1.SA). *)
let stack_step_ok delta =
  imm12 (Int64.abs delta) && Int64.equal (Int64.rem delta 16L) 0L

let exec = A64_sem.exec
let test = A64_sem.test
