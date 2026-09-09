(* Backend-specific recognizer for the tail-call plan's Stage 7 deep-index
   negative control (see .ai/ and probe_expr.ml's [--deep] mode): the
   [eval_index] path stays ordinary recursion by design (out of scope for
   this conversion), so a deep enough [Index.t] must still exhaust the
   stack, and this file names exactly what that exhaustion looks like on
   this route. Hand-written per route, not copied -- the jsoo and Melange
   copies (js/jsoo/probe_expr/, js/melange/probe_expr/) recognize their own
   backend's fault shape instead. Native's is the reference: an in-process,
   OCaml-catchable [Stack_overflow]. *)

let run_in_process = true
let is_exhausted = function Stack_overflow -> true | _ -> false
