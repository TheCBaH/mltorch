(* The RV32 target as the selected stage, its verifier and interpreter
   consume it. *)
include Rv32_op

type op = t

let abi = Rv32_reg.abi
let flags_view = Rv32_reg.flags
let stack_pointer = Rv32_reg.sp
let call_push = 0L
let link = Some Rv32_reg.ra

let address_step = function
  | Addi (_, k) | Add_wide (_, k) -> Some k
  | _ -> None

(* A register holds a 32-bit word: a pointer, an integer up to 32 bits or a
   predicate in a general register, a binary32 in a floating-point one; a
   vector register four binary32 or int32 lanes, or a mask of four. An i64
   lives in two general registers, which selection names separately, and
   there is no binary64. *)
let shape (ty : Machine_ir.Mir_type.t) =
  match ty with
  | Machine_ir.Mir_type.Ptr | Machine_ir.Mir_type.Pred
  | Machine_ir.Mir_type.Int Machine_ir.Mir_width.(W8 | W16 | W32) ->
      Some (Machine_ir.Mir_target.Bank.Gpr, 32)
  | Machine_ir.Mir_type.F32 -> Some (Machine_ir.Mir_target.Bank.Fpr, 32)
  | Machine_ir.Mir_type.Flags ->
      Some (Machine_ir.Mir_target.Bank.Flags, Rv32_reg.flags.bits)
  | Machine_ir.Mir_type.Vec
      ( ( Machine_ir.Mir_type.Elem.F32
        | Machine_ir.Mir_type.Elem.Int Machine_ir.Mir_width.W32 ),
        n )
  | Machine_ir.Mir_type.Mask n
    when Machine_ir.Mir_type.Lanes.to_int n = 4 ->
      Some (Machine_ir.Mir_target.Bank.Vector, 128)
  | Machine_ir.Mir_type.Vec
      ( ( Machine_ir.Mir_type.Elem.F32
        | Machine_ir.Mir_type.Elem.Int Machine_ir.Mir_width.W32 ),
        n )
  | Machine_ir.Mir_type.Mask n
    when Machine_ir.Mir_type.Lanes.to_int n = 16 ->
      (* a group of four registers: [Rv32_regs.Grouped] allocates them *)
      Some (Machine_ir.Mir_target.Bank.Vector, 512)
  | Machine_ir.Mir_type.Vec
      (Machine_ir.Mir_type.Elem.Int Machine_ir.Mir_width.W8, n)
    when Machine_ir.Mir_type.Lanes.to_int n = 16 ->
      Some (Machine_ir.Mir_target.Bank.Vector, 128)
  | Machine_ir.Mir_type.Vec
      (Machine_ir.Mir_type.Elem.Int Machine_ir.Mir_width.W16, n)
    when Machine_ir.Mir_type.Lanes.to_int n = 8 ->
      Some (Machine_ir.Mir_target.Bank.Vector, 128)
  | _ -> None

(* A frame access: offsets up to 2047 encode; up to 4094 the emitter adds 2047
   to the base first, in the reserved scratch. *)
let frame_offset_ok ~bytes offset =
  Int64.compare offset 0L >= 0
  && Int64.compare bytes 0L > 0
  && Int64.equal (Int64.rem offset (Int64.min bytes 4L)) 0L
  && Int64.compare (Int64.add offset bytes) 4094L <= 0

(* [addi sp, sp, k]: a 12-bit step keeping sp 16-byte aligned. *)
let stack_step_ok delta =
  imm12 delta && imm12 (Int64.neg delta) && Int64.equal (Int64.rem delta 16L) 0L

let exec = Rv32_sem.exec
let test = Rv32_sem.test
