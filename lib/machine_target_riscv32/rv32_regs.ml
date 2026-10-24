(* The registers allocation draws on. The reference strategy's scratch comes
   from the caller-saved temporaries, never a reserved register: operands t0-t2
   and ft0-ft2, the result t3 and ft3, slot copies t4 and ft4, transfer cycles
   t5 and ft5; linear scan keeps only the copy and cycle scratch out of its
   pool. Arguments and results follow ILP32F's sequence — integers and pointers
   in a0-a7, binary32 in fa0-fa7 — and a function with several results returns
   them in that same sequence, an extension of the base convention's two. *)

open Machine_ir

let view bank ~bits k =
  match (bank, bits) with
  | Mir_target.Bank.Gpr, 32 -> Rv32_reg.x k
  | Mir_target.Bank.Fpr, 32 -> Rv32_reg.f k
  | Mir_target.Bank.Vector, 128 -> Rv32_reg.v k
  | Mir_target.Bank.Vector, 512 ->
      Err.or_raise ~pp_error:Rv32_reg.pp_group_error
        (Rv32_reg.group Mir_target.Span.Four k)
  | Mir_target.Bank.Flags, _ -> Rv32_reg.flags
  | _ -> invalid_arg "Rv32_regs: no view of that shape"

(* t0-t2 are x5-x7, t3-t5 x28-x30; ft0-ft5 are f0-f5. *)
let use_scratch bank ~bits k =
  view bank ~bits
    (match bank with
    | Mir_target.Bank.Fpr -> k
    | Mir_target.Bank.Vector -> 1 + k
    | _ -> 5 + k)

let result_scratch bank ~bits =
  view bank ~bits
    (match bank with
    | Mir_target.Bank.Fpr -> 3
    | Mir_target.Bank.Vector -> 4
    | _ -> 28)

let copy_scratch bank ~bits =
  view bank ~bits
    (match bank with
    | Mir_target.Bank.Fpr -> 4
    | Mir_target.Bank.Vector -> 5
    | _ -> 29)

let cycle_scratch bank ~bits =
  view bank ~bits
    (match bank with
    | Mir_target.Bank.Fpr -> 5
    | Mir_target.Bank.Vector -> 6
    | _ -> 30)

let args tys = Rv32_op.sequence tys
let results tys = Rv32_op.sequence tys

(* What production allocation draws on, in preference order: caller-saved
   registers first (a value not live across a call pays nothing to keep), then
   callee-saved ones; never a reserved register nor the parallel copy's
   scratch (t4, t5, ft4, ft5). *)
let allocatable = function
  | Mir_target.Bank.Gpr ->
      Rv32_reg.range 10 17 @ [ 5; 6; 7; 28 ] @ [ 8; 9 ] @ Rv32_reg.range 18 26
  | Mir_target.Bank.Fpr ->
      Rv32_reg.range 10 17 @ [ 0; 1; 2; 3; 6; 7 ] @ Rv32_reg.range 28 31
      @ [ 8; 9 ] @ Rv32_reg.range 18 27
  | Mir_target.Bank.Vector ->
      (* v0 first, like any register: the forms that need it in passing admit
         no operand there, so the allocator moves around them *)
      [ 0; 1; 2; 3; 4 ] @ Rv32_reg.range 7 31
  | Mir_target.Bank.Control | Mir_target.Bank.Flags -> []

(* The pool of the grouped (LMUL 4) vector choice: sixteen-lane values in
   aligned groups of four as well as four-lane ones in single registers. A
   group's copy and cycle scratch are groups too, v24-v27 and v28-v31, so
   nothing is allocated there; the single-register scratch stays v5 and v6.
   A group may start at v0, v8, v12, v16 or v20, the bases whose members are
   all allocatable. *)
module Grouped = struct
  let view = view
  let use_scratch = use_scratch
  let result_scratch = result_scratch

  let group k =
    Err.or_raise ~pp_error:Rv32_reg.pp_group_error
      (Rv32_reg.group Mir_target.Span.Four k)

  let copy_scratch bank ~bits =
    match (bank, bits) with
    | Mir_target.Bank.Vector, 512 -> group 24
    | _ -> copy_scratch bank ~bits

  let cycle_scratch bank ~bits =
    match (bank, bits) with
    | Mir_target.Bank.Vector, 512 -> group 28
    | _ -> cycle_scratch bank ~bits

  let args = args
  let results = results

  let allocatable = function
    | Mir_target.Bank.Vector -> [ 0; 1; 2; 3; 4 ] @ Rv32_reg.range 7 23
    | b -> allocatable b
end
