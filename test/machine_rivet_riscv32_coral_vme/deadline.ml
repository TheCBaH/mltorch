(* Wall-clock seconds a simulator run may take. A test is meant to take under
   ten minutes (the arm64 runner needs several times the amd64 time), so that
   is the default and a slower run fails rather than stalls;
   `make coralnpu.vme.long.runtest` raises it with CORALNPU_RUN_DEADLINE. *)
let seconds =
  match Sys.getenv_opt "CORALNPU_RUN_DEADLINE" with
  | Some s -> float_of_string s
  | None -> 600.
