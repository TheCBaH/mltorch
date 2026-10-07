(* An SSA mark as the Machine IR event it is observed as. *)
let of_mark = function
  | Ssa_ir.Ssa_mark.Emitter -> Machine_ir.Mir_event.Emitter
  | Ssa_ir.Ssa_mark.Key -> Machine_ir.Mir_event.Key
  | Ssa_ir.Ssa_mark.Local -> Machine_ir.Mir_event.Local
  | Ssa_ir.Ssa_mark.Reduction -> Machine_ir.Mir_event.Reduction
  | Ssa_ir.Ssa_mark.Scan -> Machine_ir.Mir_event.Scan
  | Ssa_ir.Ssa_mark.Scan_update -> Machine_ir.Mir_event.Scan_update
