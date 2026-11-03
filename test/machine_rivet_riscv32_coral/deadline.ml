(* Wall-clock seconds a simulator run may take. Ten minutes is the default, so a
   slower run fails rather than stalls: 180 s was enough on an arm core but the
   64x64 attention case missed it on the CI runner. CORALNPU_RUN_DEADLINE
   raises it. *)
let seconds =
  match Sys.getenv_opt "CORALNPU_RUN_DEADLINE" with
  | Some s -> float_of_string s
  | None -> 600.
