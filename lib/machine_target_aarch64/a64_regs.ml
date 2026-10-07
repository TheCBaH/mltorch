(* The registers allocation draws on. Scratch comes from the caller-saved
   temporaries (x9-x15, v16-v23), never a reserved register: operands x9-x11
   and v16-v18, the result x12 and v19, slot copies x13 and v20, transfer
   cycles x14 and v21. Arguments and results follow AAPCS64's sequence —
   integers and pointers in x0-x7 (a 32-bit value in its W view), floats in
   v0-v7 (S or D) — and a function with several results returns them in that
   same sequence, an extension of the base convention's single result. *)

open Machine_ir

let view bank ~bits k =
  match (bank, bits) with
  | Mir_target.Bank.Gpr, 64 -> A64_reg.x k
  | Mir_target.Bank.Gpr, 32 -> A64_reg.w k
  | Mir_target.Bank.Fpr, 128 -> A64_reg.q k
  | Mir_target.Bank.Fpr, 64 -> A64_reg.d k
  | Mir_target.Bank.Fpr, 32 -> A64_reg.s k
  | Mir_target.Bank.Flags, _ -> A64_reg.nzcv
  | _ -> invalid_arg "A64_regs: no view of that shape"

let use_scratch bank ~bits k =
  view bank ~bits (match bank with Mir_target.Bank.Fpr -> 16 + k | _ -> 9 + k)

let result_scratch bank ~bits =
  view bank ~bits (match bank with Mir_target.Bank.Fpr -> 19 | _ -> 12)

let copy_scratch bank ~bits =
  view bank ~bits (match bank with Mir_target.Bank.Fpr -> 20 | _ -> 13)

let cycle_scratch bank ~bits =
  view bank ~bits (match bank with Mir_target.Bank.Fpr -> 21 | _ -> 14)

let shape (t : Mir_type.t) =
  match t with
  | Mir_type.Int Mir_width.W64 | Mir_type.Ptr -> (Mir_target.Bank.Gpr, 64)
  | Mir_type.Int _ | Mir_type.Pred -> (Mir_target.Bank.Gpr, 32)
  | Mir_type.F64 -> (Mir_target.Bank.Fpr, 64)
  | Mir_type.F32 -> (Mir_target.Bank.Fpr, 32)
  | _ -> invalid_arg "A64_regs: a value no register holds"

(* Integers and floats each take the next register of their own sequence. *)
let sequence tys =
  let gi = ref 0 and fi = ref 0 in
  List.map
    (fun t ->
      let bank, bits = shape t in
      let k = match bank with Mir_target.Bank.Fpr -> fi | _ -> gi in
      let r = view bank ~bits !k in
      incr k;
      if !k > 8 then invalid_arg "A64_regs: more than eight of one class";
      r)
    tys

let args = sequence
let results = sequence

(* What production allocation draws on, in preference order: caller-saved
   registers first (a value not live across a call pays nothing to keep), then
   callee-saved ones; never a reserved register (x16-x18, x29, x30) nor the
   allocator's own scratch (x9-x14, v16-v21). *)
let allocatable = function
  | Mir_target.Bank.Gpr -> A64_reg.range 0 8 @ [ 15 ] @ A64_reg.range 19 28
  | Mir_target.Bank.Fpr ->
      A64_reg.range 0 7 @ A64_reg.range 22 31 @ A64_reg.range 8 15
  | Mir_target.Bank.Control | Mir_target.Bank.Flags -> []
