(* The selected stage instantiated for AArch64: verifier, printer and
   interpreter. *)
module Interp = Machine_interp.Mir_sel_interp.Make (A64)
module Sel = Interp.S
