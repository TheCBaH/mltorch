(* Runs Corpus.cases through the reference evaluator and every candidate in
   [Corpus.candidate_evaluators], on every applicable backend (tail-call
   conversion, Stage 5; see .ai/). Every case here is deliberately below
   every evaluator's stack frontier -- there is no above-frontier skip logic
   yet, unlike the design record's eventual generated deep corpus. Exits
   nonzero on any mismatch, so this is a real regression gate, not a demo.

   [--bench] switches to timing mode (design record: "[--bench] uses
   [Sys.time] and suppresses correctness output"): the shallow corpus cases
   plus a few above-frontier [deep_chain] depths, reference included for the
   corpus (safe, shallow) but candidates only for the deep depths (the
   reference evaluator is never characterized above the frontier -- see the
   [deep_smoke] note below). This is timing only: allocation and
   bounce/frame-growth instrumentation the design record also asks for is
   not yet built -- see .ai/. *)

(* A real stack-safety check, not just a correctness one: every corpus case
   above is deliberately shallow, so passing it alone would not distinguish
   a candidate from the direct recursion it is meant to replace on deep
   input. Built via a loop, not recursion, so construction itself can't be
   what overflows. One hand-picked depth well past the ~1536 node frontier
   Stage 0 measured for this shape. Every candidate is checked at this
   depth, not just [eval_machine]: a delayed trampoline's safety is a
   property of its own [threshold], not something one candidate passing
   implies for another.

   [reference] (= [Eval.value], the production evaluator) is now ALSO
   checked here (Stage 6; see .ai/'s design record): its JS build installs
   [eval_hybrid ~cutoff:50] directly, so it is no longer the unsafe direct
   recursion this smoke case was originally written to route around --
   this is exactly the "check a case ... above [the cutoff] uses the safe
   path and completes beyond the direct frontier" verification the design
   record's Stage 6 asks for, run against the real production entry point,
   not a copy of it. Native's own [Eval.value] is UNCHANGED and still
   genuinely unsafe at this depth, so this smoke case still only runs on
   the JS routes, same as every other [expr_bench] case. *)
open Expr_internal

let deep_smoke_depth = 20_000

let deep_chain n =
  let e = ref (Value.const 0.) in
  for _ = 1 to n do
    e := Value.add !e (Value.const 1.)
  done;
  !e

(* [I64_binary]/[Select] nest on the int64 side exactly like [Binary]/[Select]
   do on the float side (see [Eval_js_machine.Eval_i64_state]'s doc comment),
   so this checks the SAME stack-safety property for the int64 spine that
   [deep_chain] above checks for the float one -- against every registered
   candidate, not just the two [expr_probe.deep-runtest] compares to each
   other, since a delayed trampoline's safety is a property of its own
   [threshold] (this module's own top comment). [i64_to_float] at the root
   keeps the result comparable through the same [float]-returning
   [Corpus.evaluator] signature every other case here uses. *)
let deep_chain_i64 n =
  let e = ref (Value.i64_const 0L) in
  for _ = 1 to n do
    e := Value.i64_add !e (Value.i64_const 1L)
  done;
  Value.i64_to_float !e

let run_deep_smoke name (ev : Corpus.evaluator) e ~expected failures =
  match
    Err.or_raise ~pp_error:Eval_common.pp_error
      (ev Corpus.dead_env ~output:Corpus.origin e)
  with
  | v when v = expected ->
      Printf.printf "deep_smoke: %s survives depth %d, value %g\n" name
        deep_smoke_depth v
  | v ->
      incr failures;
      Printf.printf "FAIL deep_smoke (%s): expected %g, got %g\n" name expected
        v
  | exception exn ->
      incr failures;
      Printf.printf "FAIL deep_smoke (%s): raised %s\n" name
        (Printexc.to_string exn)

let run_correctness () =
  let failures = ref 0 in
  (let expected = float_of_int deep_smoke_depth in
   let e = deep_chain deep_smoke_depth in
   List.iter
     (fun (name, ev) -> run_deep_smoke name ev e ~expected failures)
     (("reference", Eval.value) :: Corpus.candidate_evaluators));
  (let expected = float_of_int deep_smoke_depth in
   let e = deep_chain_i64 deep_smoke_depth in
   List.iter
     (fun (name, ev) ->
       run_deep_smoke (name ^ " (i64)") ev e ~expected failures)
     (("reference", Eval.value) :: Corpus.candidate_evaluators));
  List.iter
    (fun (c : Corpus.case) ->
      let report label got =
        if got <> c.Corpus.expected then begin
          incr failures;
          Printf.printf "FAIL %s (%s): expected %g, got %g\n" c.Corpus.name
            label c.Corpus.expected got
        end
      in
      (try report "reference" (c.Corpus.reference ())
       with exn ->
         incr failures;
         Printf.printf "FAIL %s (reference): raised %s\n" c.Corpus.name
           (Printexc.to_string exn));
      List.iter
        (fun (cname, candidate) ->
          try report cname (candidate ())
          with exn ->
            incr failures;
            Printf.printf "FAIL %s (%s): raised %s\n" c.Corpus.name cname
              (Printexc.to_string exn))
        c.Corpus.candidates)
    Corpus.cases;
  failures := !failures + Cleanup_negative.check ();
  failures := !failures + Order_check.check ();
  if !failures = 0 then
    Printf.printf "expr_bench: %d cases, reference and every candidate agree\n"
      (List.length Corpus.cases)
  else begin
    Printf.printf "expr_bench: %d failure(s)\n" !failures;
    exit 1
  end

(* Above-frontier depths for the deep-chain shape only: every entry here is
   already established safe for every registered candidate by
   [run_correctness]'s own [deep_smoke] check at [deep_smoke_depth], so
   timing them needs no fresh stack-safety proof of its own. Three points,
   not two, for a real scaling curve rather than a single slope estimate. *)
let bench_deep_depths = [ 2_000; 8_000; 20_000 ]

(* [Sys.time]'s resolution under jsoo/node is coarse enough that the
   original [bench_iterations = 200] left many shallow corpus cases
   reading exactly 0.0 or landing on a single ~5000ns quantum (Stage 5
   M9's own recorded gap; see .ai/) -- 100x the iterations pushes each
   trial's total elapsed time well past that granularity. [bench_trials]
   repeats the whole timed loop and reports the MINIMUM, the standard
   best-of-N technique for filtering out GC pauses and scheduler jitter
   without inflating the estimate the way an average would. *)
let bench_iterations = 20_000
let bench_warmup = 2_000
let bench_trials = 5

(* Deep-chain iterations stay small (each eval is itself expensive at
   depth 20,000), so trials matter more than iteration count there --
   fewer trials than the shallow corpus to keep total wall-clock bounded
   across 6 candidates * 3 depths. *)
let bench_deep_iterations = 20
let bench_deep_warmup = 3
let bench_deep_trials = 3

let time_it ~warmup ~trials ~iterations f =
  for _ = 1 to warmup do
    ignore (f ())
  done;
  let best = ref infinity in
  for _ = 1 to trials do
    let before = Sys.time () in
    for _ = 1 to iterations do
      ignore (f ())
    done;
    let elapsed = Sys.time () -. before in
    if elapsed < !best then best := elapsed
  done;
  !best

let report name elapsed iterations =
  Printf.printf "%-32s %10.1f ns/eval\n" name
    (elapsed *. 1e9 /. float_of_int iterations)

let run_bench () =
  Printf.printf
    "expr_bench --bench: %d corpus cases (iterations=%d warmup=%d trials=%d, \
     min-of-trials), deep_chain depths [%s] (iterations=%d warmup=%d trials=%d)\n"
    (List.length Corpus.cases) bench_iterations bench_warmup bench_trials
    (String.concat "; " (List.map string_of_int bench_deep_depths))
    bench_deep_iterations bench_deep_warmup bench_deep_trials;
  List.iter
    (fun (c : Corpus.case) ->
      Printf.printf "-- %s --\n" c.Corpus.name;
      report "reference"
        (time_it ~warmup:bench_warmup ~trials:bench_trials
           ~iterations:bench_iterations c.Corpus.reference)
        bench_iterations;
      List.iter
        (fun (cname, f) ->
          report cname
            (time_it ~warmup:bench_warmup ~trials:bench_trials
               ~iterations:bench_iterations f)
            bench_iterations)
        c.Corpus.candidates)
    Corpus.cases;
  List.iter
    (fun depth ->
      let e = deep_chain depth in
      Printf.printf "-- deep_chain depth=%d --\n" depth;
      List.iter
        (fun (cname, (ev : Corpus.evaluator)) ->
          let f () =
            Err.or_raise ~pp_error:Eval_common.pp_error
              (ev Corpus.dead_env ~output:Corpus.origin e)
          in
          report cname
            (time_it ~warmup:bench_deep_warmup ~trials:bench_deep_trials
               ~iterations:bench_deep_iterations f)
            bench_deep_iterations)
        Corpus.candidate_evaluators)
    bench_deep_depths

let () =
  if Array.exists (String.equal "--bench") Sys.argv then run_bench ()
  else run_correctness ()
