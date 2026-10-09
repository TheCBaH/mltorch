(* Call-frame information for a realized physical function, from the same
   stack-pointer and save/restore operations the instructions are made from.
   [.cfi_startproc] sets the CFA from the architecture's convention; each
   [Sp delta] adjusts it, each save of a register to the stack records where the
   caller's value is, each restore forgets it. The path to a [ret] is bracketed by
   [remember_state]/[restore_state] so the blocks after it still see the frame
   the body has. It describes the code; it changes no byte of it. *)

open Machine_ir
module Loc = Mir_phys.Loc

type directive = { name : string; argument : string }
type t = { mutable cfa : int64 }

(* [entry_cfa]: the CFA's offset from the stack pointer on entry (8 on x86-64,
   whose call pushed the return address; 0 on AArch64). *)
let create ~entry_cfa = { cfa = entry_cfa }
let cfa t = t.cfa
let set_cfa t v = t.cfa <- v
let d name argument = { name; argument }

(* What follows one operation. [reg] names a register the way GNU as's
   [.cfi_offset] reads it. *)
let after t ~reg (i : _ Mir_phys.Instr.t) =
  match i with
  | Mir_phys.Instr.Sp delta ->
      t.cfa <- Int64.sub t.cfa delta;
      [ d ".cfi_adjust_cfa_offset" (Int64.to_string (Int64.neg delta)) ]
  | Mir_phys.Instr.Save { dst = Loc.Mem { offset; _ }; src = Loc.Reg r } ->
      [ d ".cfi_offset" (Fmt.str "%s, %Ld" (reg r) (Int64.sub offset t.cfa)) ]
  | Mir_phys.Instr.Save { dst = Loc.Reg r; src = Loc.Mem _ } ->
      [ d ".cfi_restore" (reg r) ]
  | _ -> []

(* The index from which the rest of a block's operations tear the frame down: a
   trailing run of restores and positive stack steps. *)
let epilogue_start body =
  let n = List.length body in
  let arr = Array.of_list body in
  let rec go k =
    if k = 0 then 0
    else
      match arr.(k - 1) with
      | Mir_phys.Instr.Sp delta when Int64.compare delta 0L > 0 -> go (k - 1)
      | Mir_phys.Instr.Save { dst = Loc.Reg _; src = Loc.Mem _ } -> go (k - 1)
      | _ -> k
  in
  go n
