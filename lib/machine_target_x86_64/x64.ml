(* The x86-64 target as the selected stage, its verifier and interpreter
   consume it. *)
include X64_op

type op = t

let abi = X64_reg.abi
let flags_view = X64_reg.rflags
let stack_pointer = X64_reg.q X64_reg.rsp
let call_push = 8L
let link = None
let address_step = function Lea (_, k) -> Some k | _ -> None

(* a signed 32-bit displacement *)
let frame_offset_ok ~bytes offset = Int64.compare bytes 0L > 0 && disp32 offset

(* SUB/ADD RSP, imm32, in whole 8-byte words *)
let stack_step_ok delta = disp32 delta && Int64.equal (Int64.rem delta 8L) 0L
let exec = X64_sem.exec
let test = X64_sem.test
