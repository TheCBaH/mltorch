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
  let open Core.Geometry in
  let kernel =
    Hw.
      {
        h = Core.Dim.extent max_pool_kernel_h;
        w = Core.Dim.extent max_pool_kernel_w;
      }
  in
  Intrinsic.max_pool ~source:max_pool_source ~input:kernel ~kernel
    ~stride:Hw.{ h = Pos.of_int 1; w = Pos.of_int 1 }
    ~pad:Hw.{ h = Nonneg.of_int 0; w = Nonneg.of_int 0 }
    ~out:(Coord.of_fn Index.output) ~result

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

(* An int64 sum (0 + 1 + 2 + 3) at the bottom of a [deep_n]-deep chain: the
   handoff to the JS machine happens above it, so the machine's own int64 sum
   frame produces the answer. *)
let i64_sum_core ~step =
  Builder.run
    (Builder.i64_sum ~lo:Index.zero ~hi:(Index.const 4) (fun r ->
         Builder.return
           (Value.i64_add (Value.i64_const step)
              (Value.i64_of_index (Index.of_position r)))))

let deep_i64_sum_case =
  let e = ref (i64_sum_core ~step:0L) in
  for _ = 1 to deep_n do
    e := Value.i64_add !e (Value.i64_const 1L)
  done;
  Value.i64_to_float !e

(* An int64 argmax (of -(i-2)^2 over i < 4, so index 2) at the bottom of the same
   chain: the machine's argmax frame carries best value and position. *)
let deep_i64_argmax_case =
  let core =
    Builder.run
      (Builder.i64_reduction ~kind:Reduction.Argmax_index ~lo:Index.zero
         ~hi:(Index.const 4) (fun r ->
           let d =
             Value.i64_sub
               (Value.i64_of_index (Index.of_position r))
               (Value.i64_const 2L)
           in
           Builder.return
             (Value.i64_sub (Value.i64_const 0L) (Value.i64_mul d d))))
  in
  let e = ref core in
  for _ = 1 to deep_n do
    e := Value.i64_add !e (Value.i64_const 1L)
  done;
  Value.i64_to_float !e

let deep_i64_select_case =
  Value.i64_to_float
    (Value.select
       (Bool.i64_lt (deep_i64_chain ()) (Value.i64_const 1_000_000_000L))
       (Value.i64_const 1L) (Value.i64_const 0L))

(* One deep case per remaining nesting constructor. Each is built with a loop,
   like the chains above, and has a closed-form value. *)
let deep_chain ~seed step =
  let e = ref seed in
  for _ = 1 to deep_n do
    e := step !e
  done;
  !e

let deep_unary_case = deep_chain ~seed:(Value.const 1.) Value.sqrt
let deep_round_f32_case = deep_chain ~seed:(Value.const 1.) Value.round_f32

(* Each round trip adds two nodes, so half the rounds reach [deep_n]. *)
let deep_float_to_i64_case =
  let e = ref (Value.const 1.) in
  for _ = 1 to deep_n / 2 do
    e := Value.i64_to_float (Value.float_to_i64 !e)
  done;
  !e

let deep_i64_div_case =
  Value.i64_to_float
    (deep_chain ~seed:(Value.i64_const 7L) (fun e ->
         Value.i64_div e (Value.i64_const 1L)))

let deep_value_eq_case =
  Value.select
    (Bool.value_eq (deep_value_chain ()) (Value.const (float_of_int deep_n)))
    (Value.const 1.) (Value.const 0.)

let deep_i64_eq_case =
  Value.select
    (Bool.i64_eq (deep_i64_chain ()) (Value.i64_const (Int64.of_int deep_n)))
    (Value.const 1.) (Value.const 0.)

(* [Select] nests through its branch as well as its condition; the predicate is
   always true, so every round descends into [e]. *)
let always = Bool.value_lt (Value.const 0.) (Value.const 1.)

let deep_select_branch_case =
  deep_chain ~seed:(Value.const 5.) (fun e ->
      Value.select always e (Value.const 0.))

let deep_i64_select_branch_case =
  Value.i64_to_float
    (deep_chain ~seed:(Value.i64_const 5L) (fun e ->
         Value.select always e (Value.i64_const 0L)))

let deep_reduce_case =
  deep_chain
    ~seed:
      (Builder.run
         (Builder.reduction ~kind:Reduction.Sum ~lo:Index.zero
            ~hi:(Index.const 4) (fun i ->
              Builder.return (Value.value_of_index (Index.of_position i)))))
    (fun e -> Value.add e (Value.const 1.))

let scan_limits = Scan_limits.default

(* row0[l] = l; row_{r+1}[l] = row_r[l] + 1, so row_r[l] = l + r. *)
let deep_scan_at_case =
  let descriptor =
    Err.or_raise ~pp_error:Scan.pp_error
      (Builder.run
         (Builder.scan ~limits:scan_limits ~width:3 ~steps:4
            ~init:(fun ~lane ->
              Builder.return (Value.value_of_index (Index.of_position lane)))
            ~update:(fun ~step:_ ~lane ~previous_at ->
              Builder.return (Value.add (previous_at lane) (Value.const 1.)))))
  in
  deep_chain
    ~seed:
      (Value.scan_at descriptor
         ~row:(Index.clamp_low (Index.const 2))
         ~lane:(Index.clamp_low (Index.const 1)))
    (fun e -> Value.add e (Value.const 1.))

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

let eval_scan e =
  Eval.value
    ~scan_meter:(Scan_meter.create ~limits:scan_limits)
    env ~output:out_coord e

(* Every [Value]/[Bool] constructor that nests must name the deep case that
   exercises its machine frame. [value_nests]/[bool_nests] have no wildcard, so
   a new constructor fails to compile until it is classified, and [run_case]
   has none either, so a new case fails to compile until it has a runner.
   [all_deep_cases] is the one link the compiler does not check: a case left
   out of it is classified and runnable but never run. A leaf (a constructor
   whose only children are [Index.t], bounded by admission) names none. *)
type deep_case =
  | Bool
  | Float_to_i64
  | I64_argmax
  | I64_div
  | I64_eq
  | I64_select
  | I64_select_branch
  | I64_sum
  | I64_value
  | Reduce
  | Round_f32
  | Scan_at
  | Select_branch
  | Unary
  | Value
  | Value_eq

let bool_nests : Bool.t -> deep_case list = function
  | Bool.I64_eq _ -> [ I64_eq ]
  | Bool.I64_lt _ -> [ I64_select ]
  | Bool.Index_eq _ -> []
  | Bool.Value_eq _ -> [ Value_eq ]
  | Bool.Value_lt _ -> [ Bool ]

let value_nests : type a. a Value.t -> deep_case list = function
  | Value.Binary _ -> [ Value ]
  | Value.Const _ -> []
  | Value.Float_to_i64 _ -> [ Float_to_i64 ]
  | Value.I64_binary (Value.I64_div, _, _) -> [ I64_div ]
  | Value.I64_binary ((Value.I64_add | Value.I64_mul | Value.I64_sub), _, _) ->
      [ I64_value ]
  | Value.I64_const _ | Value.I64_load _ | Value.I64_local _
  | Value.I64_local_at _ | Value.I64_of_index _ ->
      []
  | Value.I64_sum _ -> [ I64_argmax; I64_sum ]
  | Value.I64_to_float _ -> [ I64_value ]
  | Value.Intrinsic _ | Value.Load _ | Value.Local _ | Value.Local_at _
  | Value.Local_scan_at _ | Value.Value_of_index _ ->
      []
  | Value.Reduce _ -> [ Reduce ]
  | Value.Round_f32 _ -> [ Round_f32 ]
  | Value.Scan_at _ -> [ Scan_at ]
  | Value.Select _ -> [ Bool; I64_select; I64_select_branch; Select_branch ]
  | Value.Unary _ -> [ Unary ]

let all_deep_cases =
  [
    Bool;
    Float_to_i64;
    I64_argmax;
    I64_div;
    I64_eq;
    I64_select;
    I64_select_branch;
    I64_sum;
    I64_value;
    Reduce;
    Round_f32;
    Scan_at;
    Select_branch;
    Unary;
    Value;
    Value_eq;
  ]

let run_case = function
  | Bool -> check_closed_form "deep_bool" ~expected:1. (eval deep_bool_case)
  | Float_to_i64 ->
      check_closed_form "deep_float_to_i64" ~expected:1.
        (eval deep_float_to_i64_case)
  | I64_argmax ->
      check_closed_form "deep_i64_argmax"
        ~expected:(float_of_int (deep_n + 2))
        (eval deep_i64_argmax_case)
  | I64_div ->
      check_closed_form "deep_i64_div" ~expected:7. (eval deep_i64_div_case)
  | I64_eq ->
      check_closed_form "deep_i64_eq" ~expected:1. (eval deep_i64_eq_case)
  | I64_select ->
      check_closed_form "deep_i64_select" ~expected:1.
        (eval deep_i64_select_case)
  | I64_select_branch ->
      check_closed_form "deep_i64_select_branch" ~expected:5.
        (eval deep_i64_select_branch_case)
  | I64_sum ->
      check_closed_form "deep_i64_sum"
        ~expected:(float_of_int (deep_n + 6))
        (eval deep_i64_sum_case)
  | I64_value ->
      check_closed_form "deep_i64_value" ~expected:(float_of_int deep_n)
        (eval deep_i64_value_case)
  | Reduce ->
      check_closed_form "deep_reduce"
        ~expected:(float_of_int (deep_n + 6))
        (eval deep_reduce_case)
  | Round_f32 ->
      check_closed_form "deep_round_f32" ~expected:1. (eval deep_round_f32_case)
  | Scan_at ->
      check_closed_form "deep_scan_at"
        ~expected:(float_of_int (deep_n + 3))
        (eval_scan deep_scan_at_case)
  | Select_branch ->
      check_closed_form "deep_select_branch" ~expected:5.
        (eval deep_select_branch_case)
  | Unary -> check_closed_form "deep_unary" ~expected:1. (eval deep_unary_case)
  | Value ->
      check_closed_form "deep_value" ~expected:(float_of_int deep_n)
        (eval deep_value_case)
  | Value_eq ->
      check_closed_form "deep_value_eq" ~expected:1. (eval deep_value_eq_case)

(* One instance of each nesting constructor: what the classification names, in
   union, must be exactly what [all_deep_cases] runs. *)
let claimed =
  let i64 = Value.i64_const 1L and f = Value.const 1. in
  let index = Index.const 0 in
  let reduction =
    Builder.run
      (Builder.reduction ~kind:Reduction.Sum ~lo:Index.zero ~hi:(Index.const 1)
         (fun _ -> Builder.return f))
  in
  let scan =
    Err.or_raise ~pp_error:Scan.pp_error
      (Builder.run
         (Builder.scan ~limits:scan_limits ~width:1 ~steps:1
            ~init:(fun ~lane:_ -> Builder.return f)
            ~update:(fun ~step:_ ~lane:_ ~previous_at:_ -> Builder.return f)))
  in
  let i64_sum =
    Builder.run
      (Builder.i64_sum ~lo:Index.zero ~hi:(Index.const 1) (fun _ ->
           Builder.return i64))
  in
  List.concat
    [
      bool_nests (Bool.i64_eq i64 i64);
      bool_nests (Bool.i64_lt i64 i64);
      bool_nests (Bool.index_eq index index);
      bool_nests (Bool.value_eq f f);
      bool_nests (Bool.value_lt f f);
      value_nests (Value.add f f);
      value_nests (Value.float_to_i64 f);
      value_nests (Value.i64_add i64 i64);
      value_nests (Value.i64_div i64 i64);
      value_nests (Value.i64_mul i64 i64);
      value_nests (Value.i64_sub i64 i64);
      value_nests i64_sum;
      value_nests (Value.i64_to_float i64);
      value_nests reduction;
      value_nests (Value.round_f32 f);
      value_nests
        (Value.scan_at scan
           ~row:(Index.clamp_low (Index.const 0))
           ~lane:(Index.clamp_low (Index.const 0)));
      value_nests (Value.select (Bool.value_lt f f) f f);
      value_nests (Value.sqrt f);
    ]

let run_deep () =
  let unclaimed =
    List.filter (fun c -> not (List.mem c claimed)) all_deep_cases
  in
  let unrun = List.filter (fun c -> not (List.mem c all_deep_cases)) claimed in
  if unclaimed <> [] || unrun <> [] then
    print_endline "deep classification: a case is unclaimed or never run";
  let results = List.map run_case all_deep_cases in
  let ok_index = check_exhausted "deep_index" deep_index_case in
  exit
    (if unclaimed = [] && unrun = [] && List.for_all Fun.id results && ok_index
     then 0
     else 1)

let run_shallow () =
  case "arithmetic" arithmetic_case;
  case "reduction" reduction_case;
  case "freshened_reduction" freshened_reduction_case;
  Printf.printf "freshen preserves structural equality: %b\n"
    (Value.equal reduction_case freshened_reduction_case);
  case "max_pool_value" max_pool_value_case;
  case "max_pool_index" max_pool_index_case;
  (* 3 * 2^53 + 3 less 3 * 2^53: a float accumulator cannot leave 3. *)
  case "i64_sum_exact"
    (Value.i64_to_float
       (Value.i64_sub
          (Builder.run
             (Builder.i64_sum ~lo:Index.zero ~hi:(Index.const 3) (fun r ->
                  Builder.return
                    (Value.i64_add
                       (Value.i64_const 9_007_199_254_740_992L)
                       (Value.i64_of_index (Index.of_position r))))))
          (Value.i64_const 27_021_597_764_222_976L)))

let () =
  if Array.exists (String.equal "--deep") Sys.argv then run_deep ()
  else run_shallow ()
