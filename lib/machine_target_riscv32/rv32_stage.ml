(* The selected stage instantiated for RV32: verifier, printer and
   interpreter. *)
module Interp = Machine_interp.Mir_sel_interp.Make (Rv32)
module Sel = Interp.S
