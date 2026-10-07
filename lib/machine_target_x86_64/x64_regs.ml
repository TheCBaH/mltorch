(* The registers allocation draws on: operands r8, r9, rsi and xmm8-xmm10,
   the result rdi and xmm11, slot copies rcx and xmm12, transfer cycles rdx and
   xmm13 — all caller-saved, none reserved. Arguments follow System V: integers
   and pointers in rdi, rsi, rdx, rcx, r8, r9 (a 32-bit value in its 32-bit
   view), floats in xmm0-xmm7; results in rax then rdx, xmm0 then xmm1. *)

open Machine_ir

let view bank ~bits k =
  match (bank, bits) with
  | Mir_target.Bank.Gpr, 64 -> X64_reg.q k
  | Mir_target.Bank.Gpr, 32 -> X64_reg.d k
  | Mir_target.Bank.Fpr, 64 -> X64_reg.sd k
  | Mir_target.Bank.Fpr, 32 -> X64_reg.ss k
  | Mir_target.Bank.Flags, _ -> X64_reg.rflags
  | _ -> invalid_arg "X64_regs: no view of that shape"

let gpr_uses = [| 8; 9; X64_reg.rsi |]

let use_scratch bank ~bits k =
  view bank ~bits
    (match bank with Mir_target.Bank.Fpr -> 8 + k | _ -> gpr_uses.(k))

let result_scratch bank ~bits =
  view bank ~bits
    (match bank with Mir_target.Bank.Fpr -> 11 | _ -> X64_reg.rdi)

let copy_scratch bank ~bits =
  view bank ~bits
    (match bank with Mir_target.Bank.Fpr -> 12 | _ -> X64_reg.rcx)

let cycle_scratch bank ~bits =
  view bank ~bits
    (match bank with Mir_target.Bank.Fpr -> 13 | _ -> X64_reg.rdx)

let shape (t : Mir_type.t) =
  match t with
  | Mir_type.Int Mir_width.W64 | Mir_type.Ptr -> (Mir_target.Bank.Gpr, 64)
  | Mir_type.Int _ | Mir_type.Pred -> (Mir_target.Bank.Gpr, 32)
  | Mir_type.F64 -> (Mir_target.Bank.Fpr, 64)
  | Mir_type.F32 -> (Mir_target.Bank.Fpr, 32)
  | _ -> invalid_arg "X64_regs: a value no register holds"

let sequence ~ints ~floats tys =
  let gi = ref 0 and fi = ref 0 in
  List.map
    (fun t ->
      let bank, bits = shape t in
      let pool, k =
        match bank with Mir_target.Bank.Fpr -> (floats, fi) | _ -> (ints, gi)
      in
      if !k >= Array.length pool then
        invalid_arg "X64_regs: more operands of one class than registers";
      let r = view bank ~bits pool.(!k) in
      incr k;
      r)
    tys

let args =
  sequence
    ~ints:[| X64_reg.rdi; X64_reg.rsi; X64_reg.rdx; X64_reg.rcx; 8; 9 |]
    ~floats:(Array.init 8 Fun.id)

let results = sequence ~ints:[| X64_reg.rax; X64_reg.rdx |] ~floats:[| 0; 1 |]
