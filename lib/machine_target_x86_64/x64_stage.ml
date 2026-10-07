(* The selected stage instantiated for x86-64: verifier, printer and
   interpreter. *)
module Interp = Machine_interp.Mir_sel_interp.Make (X64)
module Sel = Interp.S
