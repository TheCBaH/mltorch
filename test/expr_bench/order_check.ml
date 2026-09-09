(* Operand-order verification for [Eval_candidates]'s stack-safe candidates
   (tail-call plan, Stage 5; see .ai/). [Binary] and [Value_lt] are two of
   the seven order-sensitive call sites [order_probe] measures (see its
   committed goldens under js/probe/): jsoo evaluates the right operand
   first, Melange the left -- a genuine per-backend divergence in the real
   evaluator, not an artifact of this harness. Rewriting these two sites
   into explicit machine transitions (Stage 5 M1) originally hardcoded a
   single left-then-right order, silently matching Melange but diverging
   from jsoo's own reference evaluator; fixed by backend-gating both the
   initial dispatch and the final recombination in every candidate that
   turns [Binary]/[Value_lt] into a frame or a CPS continuation (see .ai/).

   This file catches a regression of that class: for each site, run a
   both-load and a both-fail case through the reference evaluator FIRST to
   get THIS backend's own actual order (rather than hardcoding
   [order_probe]'s golden strings again here, which this route has no cppo
   wiring to select between per backend -- see .ai/), then check every
   candidate's own recorded [Env.load] trace against it. Checking the
   TRACE, not just the final value or error, matters because a wrong
   evaluation order can still produce the right VALUE on a both-load case
   (addition is commutative) -- only the trace, or the winning error on a
   both-fail case, exposes it. *)

open Expr_internal

let trace : string list ref = ref []
let reset_trace () = trace := []
let record tag = trace := tag :: !trace
let trace_str () = String.concat "," (List.rev !trace)
let src_a = Source.create 60
let src_b = Source.create 61
let zero_coord = Coord.of_fn (fun _ -> Index.zero)
let origin = Coord.of_fn (fun _ -> 0)

let tag_of s =
  if Source.equal s src_a then "a"
  else if Source.equal s src_b then "b"
  else assert false

let ok_env =
  {
    Eval_common.Env.load =
      (fun s _ ->
        let tag = tag_of s in
        record tag;
        Err.return (if tag = "a" then 1. else 2.));
    load_index = (fun _ _ -> assert false);
  }

let fail_env =
  {
    Eval_common.Env.load =
      (fun s _ ->
        let tag = tag_of s in
        record tag;
        Err.fail (`Unknown_source s));
    load_index = (fun _ _ -> assert false);
  }

let load_a = Value.load src_a zero_coord
let load_b = Value.load src_b zero_coord
let binary_expr = Value.add load_a load_b

let value_lt_expr =
  Value.select (Bool.value_lt load_a load_b) (Value.const 1.) (Value.const 0.)

let traced_run ~env ~expr =
  reset_trace ();
  (try ignore (Eval.value env ~output:origin expr) with _ -> ());
  trace_str ()

let check () =
  let failures = ref 0 in
  let fail fmt =
    Printf.ksprintf
      (fun s ->
        incr failures;
        Printf.printf "FAIL order_check: %s\n" s)
      fmt
  in
  let run_case ~site ~expr ~env ~expected (cname, (ev : Corpus.evaluator)) =
    reset_trace ();
    try
      let (_ : (float, Eval_common.error) Err.t) = ev env ~output:origin expr in
      let got = trace_str () in
      if not (String.equal got expected) then
        fail "%s/%s: expected trace %s (reference's own order), got %s" site
          cname expected got
    with exn -> fail "%s/%s: raised %s" site cname (Printexc.to_string exn)
  in
  let check_site ~site ~expr =
    (* A non-vacuous baseline: the reference's own ok/fail traces must
       actually differ in content (not just be equal-but-reordered) so this
       check has power -- both a genuinely-swapped candidate order AND a
       candidate that fails to short-circuit on the first error would be
       caught. *)
    let expected_ok = traced_run ~env:ok_env ~expr in
    let expected_fail = traced_run ~env:fail_env ~expr in
    if String.equal expected_ok "a,b" = String.equal expected_ok "b,a" then
      fail "%s: reference ok-trace %s is not one of a,b / b,a" site expected_ok;
    if not (String.length expected_fail = 1) then
      fail "%s: reference fail-trace %s did not short-circuit" site
        expected_fail;
    List.iter
      (fun (cname, ev) ->
        run_case ~site:(site ^ "/ok") ~expr ~env:ok_env ~expected:expected_ok
          (cname, ev);
        run_case ~site:(site ^ "/fail") ~expr ~env:fail_env
          ~expected:expected_fail (cname, ev))
      Corpus.candidate_evaluators
  in
  check_site ~site:"Binary" ~expr:binary_expr;
  check_site ~site:"Value_lt" ~expr:value_lt_expr;
  !failures
