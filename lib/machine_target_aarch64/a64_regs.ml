(* The registers allocation draws on. The reference strategy's scratch comes
   from the caller-saved temporaries (x9-x15, v16-v23), never a reserved
   register: operands x9-x11 and v16-v18, the result x12 and v19, slot copies
   x13 and v20, transfer cycles x14 and v21; linear scan keeps only the copy
   and cycle scratch out of its pool. Arguments and results follow AAPCS64's sequence —
   integers and pointers in x0-x7 (a 32-bit value in its W view), floats in
   v0-v7 (S or D) — and a function with several results returns them in that
   same sequence, an extension of the base convention's single result. *)

open Machine_ir
module Num = A64_reg.Num
module Num_seq = A64_reg.Num_seq

let view_at bank ~(bits : Mir_target.Bits.t) k =
  let width = (bits :> int) in
  match (bank, width) with
  | Mir_target.Bank.Gpr, 64 -> A64_reg.x k
  | Mir_target.Bank.Gpr, 32 -> A64_reg.w k
  | Mir_target.Bank.Fpr, 128 -> A64_reg.q k
  | Mir_target.Bank.Fpr, 64 -> A64_reg.d k
  | Mir_target.Bank.Fpr, 32 -> A64_reg.s k
  | Mir_target.Bank.Flags, _ -> A64_reg.nzcv
  | _ -> invalid_arg "A64_regs: no view of that shape"

let view = view_at

let use_scratch bank ~bits k =
  view bank ~bits
    (Num_seq.at
       (Num.of_int (match bank with Mir_target.Bank.Fpr -> 16 | _ -> 9))
       k)

let result_scratch bank ~bits =
  view_at bank ~bits
    (match bank with
    | Mir_target.Bank.Fpr -> Num.of_int 19
    | _ -> Num.of_int 12)

let copy_scratch bank ~bits =
  view_at bank ~bits
    (match bank with
    | Mir_target.Bank.Fpr -> Num.of_int 20
    | _ -> Num.of_int 13)

let cycle_scratch bank ~bits =
  view_at bank ~bits
    (match bank with
    | Mir_target.Bank.Fpr -> Num.of_int 21
    | _ -> Num.of_int 14)

let shape (t : Mir_type.t) =
  match t with
  | Mir_type.Int Mir_width.W64 | Mir_type.Ptr ->
      (Mir_target.Bank.Gpr, Mir_target.Bits.of_int 64)
  | Mir_type.Int _ | Mir_type.Pred ->
      (Mir_target.Bank.Gpr, Mir_target.Bits.of_int 32)
  | Mir_type.F64 -> (Mir_target.Bank.Fpr, Mir_target.Bits.of_int 64)
  | Mir_type.F32 -> (Mir_target.Bank.Fpr, Mir_target.Bits.of_int 32)
  | _ -> invalid_arg "A64_regs: a value no register holds"

(* Integers and floats each take the next register of their own sequence. *)
let sequence tys =
  let gi = ref Num_seq.zero and fi = ref Num_seq.zero in
  List.map
    (fun t ->
      let bank, bits = shape t in
      let k = match bank with Mir_target.Bank.Fpr -> fi | _ -> gi in
      let r = view_at bank ~bits !k in
      k := Num.succ !k;
      if Num.compare !k (Num.of_int 8) > 0 then
        invalid_arg "A64_regs: more than eight of one class";
      r)
    tys

let args = sequence
let results = sequence

(* What production allocation draws on, in preference order: caller-saved
   registers first (a value not live across a call pays nothing to keep), then
   callee-saved ones; never a reserved register (x16-x18, x29, x30) nor the
   parallel copy's scratch (x13, x14, v20, v21). The reference strategy's
   operand and result scratch is allocatable here: linear scan gives every
   operand and result a register of its own. *)
let pool = function
  | Mir_target.Bank.Gpr ->
      Num_seq.range 0 12 @ Num_seq.of_ints [ 15 ] @ Num_seq.range 19 28
  | Mir_target.Bank.Fpr ->
      Num_seq.range 0 7 @ Num_seq.range 16 19 @ Num_seq.range 22 31
      @ Num_seq.range 8 15
  | Mir_target.Bank.Control | Mir_target.Bank.Flags | Mir_target.Bank.Vector ->
      []

let allocatable = pool
