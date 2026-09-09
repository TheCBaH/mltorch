(* Cleanup-protocol NEGATIVE controls (tail-call plan, Stage 5; see .ai/):
   proves the exception-path cleanup sweep is not vacuous by showing it can
   be made to fail. A nested [Scan_at] (the outer scan's own [update] body
   evaluates an independent inner [Scan_at]) shares one [Scan_meter] across
   two probes: the nested evaluation itself, then a fresh row-0-only second
   scan on the SAME meter afterward -- which only succeeds if every
   reservation the nested evaluation made was actually released. Every
   candidate's own private test-only [skip_cleanup] selector (same contract
   everywhere, see [Eval_candidates.eval_machine]'s doc comment) lets this
   file deliberately withhold one release, proving the second probe's
   success in the full-cleanup cases is a real signal, not a vacuous one.

   All four candidates are exercised, not just [eval_machine]: the cleanup
   protocol is a design requirement independently implemented by each, and
   Stage 5 M7 originally treated one as a representative, sufficient proof
   -- generalized here (Stage 5, post-M9) now that all four expose the same
   [skip_cleanup] hook. The [skip_evaluator] list below intentionally omits
   [eval_trampoline_delayed]'s higher [threshold=8] and [eval_hybrid]'s
   higher [cutoff=50] configurations already covered by [Corpus]: this file
   tests the PROTOCOL, which does not vary with those tuning knobs, not
   another instance of correctness. *)

open Expr_internal

exception Injected_fault of int ref

let expected_exn = Injected_fault (ref 0)
let fault_source = Source.create 50

(* [outer_width]/[inner_width]/[second_width] and [scan_limits]'s
   [max_state] satisfy the design record's cleanup-negative-control budget:
   O = 2*outer_width, I = 2*inner_width, S = 2*second_width;
   O + I <= max_state; S <= max_state; max_state < S + min(O, I). With
   O = I = 4 and S = 6, max_state = 8 is the tightest choice: the nested
   evaluation's own two reservations fit exactly, a fully-released meter
   still has room for the second scan, and leaking either one (4) leaves
   only 4 free -- short of the second scan's 6. *)
let outer_width = 2
let inner_width = 2
let second_width = 3

let scan_limits =
  Err.or_raise ~pp_error:Scan_limits.pp_error
    (Scan_limits.create ~max_state:8 ~max_updates:100L)

let pos n = Index.clamp_low (Index.const n)

(* row0[l] = l; row1's [update] reads [fault_source], which every mode below
   resolves differently. [steps = 1] is the minimum shape with an [update]
   phase at all -- row 0 alone would never reach it. *)
let inner_descriptor =
  Err.or_raise ~pp_error:Scan.pp_error
    (Builder.run
       (Builder.scan ~limits:scan_limits ~width:inner_width ~steps:1
          ~init:(fun ~lane ->
            Builder.return (Value.value_of_index (Index.of_position lane)))
          ~update:(fun ~step:_ ~lane:_ ~previous_at:_ ->
            Builder.return
              (Value.load fault_source (Coord.of_fn (fun _ -> Index.zero))))))

let inner_scan_at = Value.scan_at inner_descriptor ~row:(pos 1) ~lane:Index.zero

(* row0[l] = 100; row1's [update] is [inner_scan_at + previous_at l]: the
   inner [Scan_at] -- and its own cleanup, restoring [local_at_ref] -- has
   always finished by the time [previous_at] is read, regardless of which
   operand a given backend evaluates first (Stage 5 M8), since [previous_at]
   is an ordinary [Load], not a [Scan_at], and never itself pushes a
   cleanup. That makes the success-mode check below a genuine
   resolver-restoration test, not just a value check: if the inner
   [Scan_at]'s cleanup failed to restore the resolver, [previous_at] would
   answer from the inner scan's rebound [prev] instead of the outer's own. *)
let outer_descriptor =
  Err.or_raise ~pp_error:Scan.pp_error
    (Builder.run
       (Builder.scan ~limits:scan_limits ~width:outer_width ~steps:1
          ~init:(fun ~lane:_ -> Builder.return (Value.const 100.))
          ~update:(fun ~step:_ ~lane ~previous_at ->
            Builder.return (Value.add inner_scan_at (previous_at lane)))))

let outer_scan_at = Value.scan_at outer_descriptor ~row:(pos 1) ~lane:Index.zero

let second_descriptor =
  Err.or_raise ~pp_error:Scan.pp_error
    (Builder.run
       (Builder.scan ~limits:scan_limits ~width:second_width ~steps:0
          ~init:(fun ~lane ->
            Builder.return (Value.value_of_index (Index.of_position lane)))
          ~update:(fun ~step:_ ~lane:_ ~previous_at:_ ->
            Builder.return (Value.const 0.))))

let second_scan_at =
  Value.scan_at second_descriptor ~row:Index.zero ~lane:Index.zero

let env fire =
  {
    Eval_common.Env.load =
      (fun s _ -> if Source.equal s fault_source then fire () else assert false);
    load_index = (fun _ _ -> assert false);
  }

let success_env = env (fun () -> Err.return 7.)
let structured_env = env (fun () -> Err.fail (`Unknown_source fault_source))
let exception_env = env (fun () -> raise expected_exn)

(* [Corpus.evaluator]'s full optional-argument shape, plus [?skip_cleanup]
   in its own declaration position (right before the mandatory [env]) --
   matching every candidate's own signature WIDTH exactly, unlike
   [Corpus.evaluator] itself, so a partial application of just
   [~threshold]/[~cutoff] (which erases none of these) or a bare candidate
   reference both unify with this type with no eta-expansion needed. *)
type skip_evaluator =
  ?local:(Local_var.t -> float option) ->
  ?local_at:(Local_var.t -> int -> float option) ->
  ?scan:Eval_common.scan_reader ->
  ?scan_meter:Scan_meter.t ->
  ?reducer:(Reduce_var.t * int) list ->
  ?on_reduction:(unit -> unit) ->
  ?skip_cleanup:(int -> bool) ->
  Eval_common.Env.t ->
  output:int Coord.t ->
  Value.t ->
  (float, Eval_common.error) Err.t

let skip_evaluators : (string * skip_evaluator) list =
  [
    ("eval_hybrid(cutoff=0)", Eval_hybrid.eval_hybrid ~cutoff:0);
    ("eval_machine", Eval_candidates.eval_machine);
    ("eval_machine_reuse", Eval_machine_reuse.eval_machine_reuse);
    ( "eval_trampoline_delayed(threshold=1)",
      Eval_trampoline_delayed.eval_trampoline_delayed ~threshold:1 );
  ]

let check () =
  let failures = ref 0 in
  let fail fmt =
    Printf.ksprintf
      (fun s ->
        incr failures;
        Printf.printf "FAIL cleanup_negative: %s\n" s)
      fmt
  in
  (* [Err.payload] unwraps [Err.Error.t]'s detection-stack wrapper down to
     the bare error row, so a caller here can pattern-match
     [Eval_common.error]'s own constructors directly instead of the
     wrapper's. *)
  let run_second (ev : skip_evaluator) meter =
    Err.payload
      (ev ~scan_meter:meter Corpus.dead_env ~output:Corpus.origin second_scan_at)
  in
  let expect_second_ok ev label meter =
    match run_second ev meter with
    | Ok v when v = 0. -> ()
    | Ok v -> fail "%s: second-scan value %g, expected 0" label v
    | Error e ->
        fail "%s: second-scan unexpectedly failed: %s" label
          (Fmt.str "%a" Eval_common.pp_error e)
  in
  let expect_second_over_limit ev label meter =
    match run_second ev meter with
    | Ok v ->
        fail
          "%s: second-scan unexpectedly succeeded with %g (leak not detected)"
          label v
    | Error (`Scan_meter (Scan_meter.State_over_limit _)) -> ()
    | Error e ->
        fail "%s: second-scan failed with the wrong error: %s" label
          (Fmt.str "%a" Eval_common.pp_error e)
  in
  List.iter
    (fun (cname, (ev : skip_evaluator)) ->
      let label suffix = Printf.sprintf "%s: %s" cname suffix in
      (* 1. Success: correct value, resolver restored, meter fully
         released. *)
      (let meter = Scan_meter.create ~limits:scan_limits in
       match
         Err.payload
           (ev ~scan_meter:meter success_env ~output:Corpus.origin outer_scan_at)
       with
       | Ok v when v = 107. -> expect_second_ok ev (label "success") meter
       | Ok v -> fail "%s: expected 107, got %g" (label "success") v
       | Error e ->
           fail "%s: unexpectedly failed: %s" (label "success")
             (Fmt.str "%a" Eval_common.pp_error e));
      (* 2. Structured failure, full cleanup: a clean [Error], not a
         raise. *)
      (let meter = Scan_meter.create ~limits:scan_limits in
       (match
          Err.payload
            (ev ~scan_meter:meter structured_env ~output:Corpus.origin
               outer_scan_at)
        with
       | Ok v -> fail "%s: expected Error, got Ok %g" (label "structured") v
       | Error _ -> ()
       | exception exn ->
           fail "%s: raised %s instead of returning Error" (label "structured")
             (Printexc.to_string exn));
       expect_second_ok ev (label "structured full cleanup") meter);
      (* 3. Ordinary exception, full cleanup: exact physical identity. *)
      (let meter = Scan_meter.create ~limits:scan_limits in
       (try
          ignore
            (ev ~scan_meter:meter exception_env ~output:Corpus.origin
               outer_scan_at);
          fail "%s: did not raise" (label "exception")
        with
       | Injected_fault _ as exn when exn == expected_exn -> ()
       | Injected_fault _ ->
           fail "%s: raised a DIFFERENT Injected_fault" (label "exception")
       | exn ->
           fail "%s: raised %s, not Injected_fault" (label "exception")
             (Printexc.to_string exn));
       expect_second_ok ev (label "exception full cleanup") meter);
      (* 4/5. skip_cleanup: leaking either reservation must make the
         second scan fail with State_over_limit -- proof (2)/(3)'s
         success is not vacuous. Index 0 is the most-recently-pushed
         cleanup (inner's, since the inner Scan_at is entered from
         inside the outer's own update); index 1 is the outer's. This
         nesting relationship is structural (inner is always entered
         while filling outer's row), not evaluation-order-dependent, so
         it holds identically across every candidate. *)
      List.iter
        (fun (suffix, skip_index) ->
          let meter = Scan_meter.create ~limits:scan_limits in
          (try
             ignore
               (ev
                  ~skip_cleanup:(fun i -> i = skip_index)
                  ~scan_meter:meter exception_env ~output:Corpus.origin
                  outer_scan_at);
             fail "%s: did not raise" (label suffix)
           with
          | Injected_fault _ as exn when exn == expected_exn -> ()
          | exn -> fail "%s: raised %s" (label suffix) (Printexc.to_string exn));
          expect_second_over_limit ev (label suffix) meter)
        [ ("skip_outer", 1); ("skip_inner", 0) ])
    skip_evaluators;
  !failures
