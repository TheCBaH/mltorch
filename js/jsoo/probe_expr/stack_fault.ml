(* jsoo route's recognizer for probe_expr.ml's [--deep] deep-index negative
   control (tail-call plan, Stage 7; see .ai/). Measured directly (a
   scratch probe, reverted, recorded in .ai/expr_tailcall_design.md): a
   200,000-deep [Index.add] chain overflows [eval_index]'s ordinary
   recursion as OCaml's own, in-process-catchable [Stack_overflow] under
   node, identically to native -- js_of_ocaml does not turn this into an
   uncaught JS [RangeError]. *)

let run_in_process = true
let is_exhausted = function Stack_overflow -> true | _ -> false
