(* Standalone Expr probe for the tail-call conversion (see
   expr-tailcall-implementation-plan.md under .ai/), separate from
   order_probe.ml: that probe pins down per-backend evaluation ORDER, this one
   is the harness Stage 4 onward grows (starting with a bounded Max_pool case)
   to check plain observable VALUES across native, jsoo and Melange as the
   evaluator's JS branch diverges. Native, jsoo and Melange all build this
   exact source; `make expr_probe.runtest` diffs jsoo/Melange's output against
   the native golden. Uses only the public [Expr] API -- no [Expr_internal]
   access -- so it exercises exactly what a real consumer can. *)

open Expr

(* [load_index]/[Data] never appear in any case below, so that half of the
   resolver stays unreachable -- a real one is order_probe.ml's job. [load]
   IS reached, by the Stage 4 max-pool case: it returns [h*1000+w], strictly
   increasing in row-major (h, w) order over the case's window, so that case's
   expected max/index are closed forms and need no reference evaluator run. *)
let env =
  {
    Eval.Env.load =
      (fun _ c ->
        Ok (float_of_int ((Coord.get c Axis.H * 1000) + Coord.get c Axis.W)));
    load_index = (fun _ _ -> assert false);
  }

let out_coord = Coord.of_fn (fun _ -> 0)
let eval e = Eval.value env ~output:out_coord e

let show_result = function
  | Ok v -> Printf.sprintf "ok %g" v
  | Error e ->
      Printf.sprintf "error %s"
        (Format.asprintf "%a" Eval.pp_error (Err.Error.kind e))

let case name e = Printf.printf "%s: %s\n" name (show_result (eval e))

(* ---- arithmetic, built with the public smart constructors ---- *)

let arithmetic_case =
  Value.add (Value.const 2.) (Value.mul (Value.const 3.) (Value.const 4.))

(* ---- a reduction, built with [Builder] rather than the raw constructor:
   [Reduction.t] is private, so [Builder.reduction] is the only public way to
   mint the bound variable and hand it to the body. Sums the bound index
   0+1+2+3+4 = 10. ---- *)

let reduction_case =
  Builder.run
    (Builder.reduction ~kind:Reduction.Sum ~lo:Index.zero ~hi:(Index.const 5)
       (fun i -> Builder.return (Value.value_of_index (Index.of_position i))))

(* ---- [Rewrite.freshen] renames every bound reducer identity; [Value.equal]
   is alpha-equivalence-aware, so a freshened reduction must still compare
   equal to the original and evaluate to the same result. ---- *)

let freshened_reduction_case = Builder.run (Rewrite.freshen reduction_case)

(* ---- Max_pool with a low-hundreds window (Stage 4): under [JS_BACKEND] this
   is now one self-recursive [loop] over [pool_tag], not the mutually
   tail-recursive [rows]/[cols] pair -- see .ai/. [out] forwards every axis to
   the evaluator's own [~output], so a single output pixel at (h=0, w=0)
   selects a window covering the whole synthetic input. *)
let max_pool_source = Source.create 0
let max_pool_kernel_h = 120
let max_pool_kernel_w = 150

let max_pool ~result =
  Err.or_raise ~pp_error:Intrinsic.pp_error
    (Intrinsic.max_pool ~source:max_pool_source ~in_h:max_pool_kernel_h
       ~in_w:max_pool_kernel_w ~kernel_h:max_pool_kernel_h
       ~kernel_w:max_pool_kernel_w ~stride_h:1 ~stride_w:1 ~pad_h:0 ~pad_w:0
       ~out:(Coord.of_fn Index.output) ~result)

let max_pool_value_case = Value.intrinsic (max_pool ~result:Value)
let max_pool_index_case = Value.intrinsic (max_pool ~result:Index)

(* ---- [--deep] mode (Stage 7): depth-200,000 Value/Bool cases checked
   against closed forms, plus the deep-index negative control. Built with a
   loop, never recursion, so construction itself cannot overflow -- only
   evaluation is meant to, and only for [deep_index_case] (see
   Stack_fault's own doc comment). Never run against native by any Makefile
   target: [Eval.value] is still ordinary recursion there, and would
   genuinely overflow on [deep_value_case]/[deep_bool_case], which the
   installed JS driver (Stage 6) is the whole point of surviving. ---- *)

let deep_n = 200_000

let deep_value_chain () =
  let e = ref (Value.const 0.) in
  for _ = 1 to deep_n do
    e := Value.add !e (Value.const 1.0)
  done;
  !e

let deep_value_case = deep_value_chain ()

let deep_bool_case =
  Value.select
    (Bool.value_lt (deep_value_chain ()) (Value.const 1e9))
    (Value.const 1.) (Value.const 0.)

(* [I64_binary]/[Select] nest on the int64 side exactly like [Binary]/[Select]
   do on the float side, but only gained cutoff/machine-handoff treatment on
   the JS backends after the float/bool grammar did (see
   [Eval_js_machine.Eval_i64_state]) -- this case is what actually proves
   that treatment, rather than relying on [deep_value_case]/[deep_bool_case]
   to stand in for a spine they never exercise. *)
let deep_i64_chain () =
  let e = ref (Value.i64_const 0L) in
  for _ = 1 to deep_n do
    e := Value.i64_add !e (Value.i64_const 1L)
  done;
  !e

let deep_i64_value_case = Value.i64_to_float (deep_i64_chain ())

let deep_i64_select_case =
  Value.i64_to_float
    (Value.select
       (Bool.i64_lt (deep_i64_chain ()) (Value.i64_const 1_000_000_000L))
       (Value.i64_const 1L) (Value.i64_const 0L))

let deep_index_case =
  let idx = ref (Index.of_position Index.zero) in
  for _ = 1 to deep_n do
    idx := Index.add !idx (Index.const 1)
  done;
  Value.value_of_index !idx

let check_closed_form name ~expected actual =
  let ok =
    match actual with Ok v -> Float.equal v expected | Error _ -> false
  in
  Printf.printf "%s: %s\n" name
    (if ok then "ok" else "MISMATCH " ^ show_result actual);
  ok

let check_exhausted name e =
  if not Stack_fault.run_in_process then (
    Printf.printf "%s: skipped (needs the separate-process route)\n" name;
    true)
  else
    match eval e with
    | (Ok _ | Error _) as r ->
        Printf.printf "%s: UNEXPECTED COMPLETION %s\n" name (show_result r);
        false
    | exception exn when Stack_fault.is_exhausted exn ->
        Printf.printf "%s: exhausted as expected\n" name;
        true
    | exception exn ->
        Printf.printf "%s: WRONG EXCEPTION %s\n" name (Printexc.to_string exn);
        false

let run_deep () =
  let ok1 =
    check_closed_form "deep_value" ~expected:(float_of_int deep_n)
      (eval deep_value_case)
  in
  let ok2 = check_closed_form "deep_bool" ~expected:1. (eval deep_bool_case) in
  let ok3 =
    check_closed_form "deep_i64_value" ~expected:(float_of_int deep_n)
      (eval deep_i64_value_case)
  in
  let ok4 =
    check_closed_form "deep_i64_select" ~expected:1. (eval deep_i64_select_case)
  in
  let ok5 = check_exhausted "deep_index" deep_index_case in
  exit (if ok1 && ok2 && ok3 && ok4 && ok5 then 0 else 1)

let run_shallow () =
  case "arithmetic" arithmetic_case;
  case "reduction" reduction_case;
  case "freshened_reduction" freshened_reduction_case;
  Printf.printf "freshen preserves structural equality: %b\n"
    (Value.equal reduction_case freshened_reduction_case);
  case "max_pool_value" max_pool_value_case;
  case "max_pool_index" max_pool_index_case

let () =
  if Array.exists (String.equal "--deep") Sys.argv then run_deep ()
  else run_shallow ()
