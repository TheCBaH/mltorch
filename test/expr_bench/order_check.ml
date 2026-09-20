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

(* [load_index] feeds the int64 sites. [ok_env] gives a = 7 and b = 2, so
   [I64_div] succeeds with 3; [div_zero_env] gives a = 1 and b = 0. *)
let env_with ~load ~load_index = { Eval_common.Env.load; load_index }

let ok_env =
  env_with
    ~load:(fun s _ ->
      let tag = tag_of s in
      record tag;
      Err.return (if tag = "a" then 1. else 2.))
    ~load_index:(fun s _ ->
      let tag = tag_of s in
      record tag;
      Err.return (if tag = "a" then 7L else 2L))

let div_zero_env =
  env_with ~load:ok_env.Eval_common.Env.load ~load_index:(fun s _ ->
      let tag = tag_of s in
      record tag;
      Err.return (if tag = "a" then 1L else 0L))

let fail_env =
  env_with
    ~load:(fun s _ ->
      record (tag_of s);
      Err.fail (`Unknown_source s))
    ~load_index:(fun s _ ->
      record (tag_of s);
      Err.fail (`Unknown_source s))

let load_a = Value.load src_a zero_coord
let load_b = Value.load src_b zero_coord
let binary_expr = Value.add load_a load_b

let value_eq_expr =
  Value.select (Bool.value_eq load_a load_b) (Value.const 1.) (Value.const 0.)

let value_lt_expr =
  Value.select (Bool.value_lt load_a load_b) (Value.const 1.) (Value.const 0.)

let i64_a = Value.i64_load src_a zero_coord
let i64_b = Value.i64_load src_b zero_coord
let i64_binary_expr = Value.i64_to_float (Value.i64_add i64_a i64_b)
let i64_div_expr = Value.i64_to_float (Value.i64_div i64_a i64_b)

let i64_eq_expr =
  Value.select (Bool.i64_eq i64_a i64_b) (Value.const 1.) (Value.const 0.)

let i64_lt_expr =
  Value.select (Bool.i64_lt i64_a i64_b) (Value.const 1.) (Value.const 0.)

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
    (* The shipped evaluator hands off to its machine below its depth cutoff,
       so the same site nested [deep] levels down must keep the order it has
       when evaluated directly. Right operand of each [add] is the constant,
       so the wrapper adds no load. *)
    let deep =
      let e = ref expr in
      for _ = 1 to 200 do
        e := Value.add !e (Value.const 0.)
      done;
      !e
    in
    if not (String.equal (traced_run ~env:ok_env ~expr:deep) expected_ok) then
      fail "%s: shipped evaluator's ok-trace changes below the cutoff" site;
    if not (String.equal (traced_run ~env:fail_env ~expr:deep) expected_fail)
    then fail "%s: shipped evaluator's fail-trace changes below the cutoff" site;
    List.iter
      (fun (cname, ev) ->
        run_case ~site:(site ^ "/ok") ~expr ~env:ok_env ~expected:expected_ok
          (cname, ev);
        run_case ~site:(site ^ "/fail") ~expr ~env:fail_env
          ~expected:expected_fail (cname, ev))
      Corpus.candidate_evaluators
  in
  (* [I64_div]'s error fires in the combine step, after both operands: the
     trace must still be the reference's a/b order, and the error itself the
     reference's, not an operand's. *)
  let show r =
    match r with
    | Ok v -> Printf.sprintf "ok %g" v
    | Error e ->
        Format.asprintf "error %a" Eval_common.pp_error (Err.Error.kind e)
  in
  let check_div_zero () =
    let site = "I64_div/zero" in
    reset_trace ();
    let reference = Eval.value div_zero_env ~output:origin i64_div_expr in
    let expected_trace = trace_str () and expected = show reference in
    if not (String.equal expected "error I64 division by zero") then
      fail "%s: reference gave %s, not the division error" site expected;
    List.iter
      (fun (cname, (ev : Corpus.evaluator)) ->
        reset_trace ();
        match show (ev div_zero_env ~output:origin i64_div_expr) with
        | exception exn ->
            fail "%s/%s: raised %s" site cname (Printexc.to_string exn)
        | got ->
            let got_trace = trace_str () in
            if not (String.equal got expected) then
              fail "%s/%s: expected %s, got %s" site cname expected got;
            if not (String.equal got_trace expected_trace) then
              fail "%s/%s: expected trace %s, got %s" site cname expected_trace
                got_trace)
      Corpus.candidate_evaluators
  in
  check_site ~site:"Binary" ~expr:binary_expr;
  check_site ~site:"I64_binary" ~expr:i64_binary_expr;
  check_site ~site:"I64_div" ~expr:i64_div_expr;
  check_site ~site:"I64_eq" ~expr:i64_eq_expr;
  check_site ~site:"I64_lt" ~expr:i64_lt_expr;
  check_div_zero ();
  check_site ~site:"Value_eq" ~expr:value_eq_expr;
  check_site ~site:"Value_lt" ~expr:value_lt_expr;
  !failures
