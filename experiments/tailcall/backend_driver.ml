(* This file is intentionally passed through cppo with a definition supplied
   by each backend's dune stanza.  It demonstrates the selection mechanism,
   not a claim that the current Expr library can distinguish an executable's
   [(modes exe)] and [(modes js)] while compiling its shared bytecode once. *)

#if defined JS_BACKEND
let name = "tag-state-loop"
let run n = Tailcall_cases.mutual_tag n 0
#else
let name = "mutual-functions"
let run n = Tailcall_cases.mutual_tail_a n 0
#endif
