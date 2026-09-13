(* Stage 5 correctness corpus for [Eval_candidates] (tail-call conversion;
   see .ai/): a first, hand-written set of cases covering every [float Value.t]/
   [Bool.t] constructor that recurses through [go]/[guard], including nested
   and multi-row [Reduce]/[Scan_at] -- not yet the generated deep/order
   corpus the design record calls for (that needs the per-backend goldens
   and closed-form deep generators Stage 5 still owes). Every case owns its
   own evaluation config; nothing here is shared mutable state except each
   case's own fresh [Scan_meter.t], built at call time so the reference and
   candidate runs never see each other's charges.

   [Expr_internal] is this route's own shim ([include Expr_internal_js] for
   jsoo, [include Expr_internal_mel] for Melange), so every name below
   resolves against real internal types identically on both backends. *)

open Expr_internal

(* Every candidate under test shares this shape with [Eval.value] itself
   (modulo [eval_trampoline_delayed]'s extra [~threshold], partially applied
   below so each entry in [candidate_evaluators] fits it exactly). *)
type evaluator =
  ?local:(Local_var.t -> float option) ->
  ?local_at:(Local_var.t -> int -> float option) ->
  ?local_i64:(Local_var.t -> int64 option) ->
  ?local_at_i64:(Local_var.t -> int -> int64 option) ->
  ?scan:Eval_common.scan_reader ->
  ?scan_meter:Scan_meter.t ->
  ?reducer:(Reduce_var.t * int) list ->
  ?on_reduction:(unit -> unit) ->
  Eval_common.Env.t ->
  output:int Coord.t ->
  float Value.t ->
  (float, Eval_common.error) Err.t

(* Alphabetical by name (CLAUDE.md); a [threshold] group sorts by its own
   value. *)
(* Every candidate carries an extra optional [?skip_cleanup] (the cleanup
   protocol's private test-only hook, exercised by [Cleanup_negative]; see
   .ai/) that [evaluator]'s type doesn't mention -- these thin eta-expansions
   hide it, so [candidate_evaluators] below can stay a uniform
   [(string * evaluator) list]. *)
let eval_machine : evaluator =
 fun ?local ?local_at ?local_i64 ?local_at_i64 ?scan ?scan_meter ?reducer
     ?on_reduction env ~output e ->
  Eval_candidates.eval_machine ?local ?local_at ?local_i64 ?local_at_i64 ?scan
    ?scan_meter ?reducer ?on_reduction env ~output e

let eval_machine_reuse : evaluator =
 fun ?local ?local_at ?local_i64 ?local_at_i64 ?scan ?scan_meter ?reducer
     ?on_reduction env ~output e ->
  Eval_machine_reuse.eval_machine_reuse ?local ?local_at ?local_i64
    ?local_at_i64 ?scan ?scan_meter ?reducer ?on_reduction env ~output e

let eval_hybrid ~cutoff : evaluator =
 fun ?local ?local_at ?local_i64 ?local_at_i64 ?scan ?scan_meter ?reducer
     ?on_reduction env ~output e ->
  Eval_hybrid.eval_hybrid ~cutoff ?local ?local_at ?local_i64 ?local_at_i64
    ?scan ?scan_meter ?reducer ?on_reduction env ~output e

let eval_trampoline_delayed ~threshold : evaluator =
 fun ?local ?local_at ?local_i64 ?local_at_i64 ?scan ?scan_meter ?reducer
     ?on_reduction env ~output e ->
  Eval_trampoline_delayed.eval_trampoline_delayed ~threshold ?local ?local_at
    ?local_i64 ?local_at_i64 ?scan ?scan_meter ?reducer ?on_reduction env
    ~output e

let candidate_evaluators : (string * evaluator) list =
  [
    ("eval_hybrid(cutoff=0)", eval_hybrid ~cutoff:0);
    ("eval_hybrid(cutoff=50)", eval_hybrid ~cutoff:50);
    ("eval_machine", eval_machine);
    ("eval_machine_reuse", eval_machine_reuse);
    ( "eval_trampoline_delayed(threshold=1)",
      eval_trampoline_delayed ~threshold:1 );
    ( "eval_trampoline_delayed(threshold=8)",
      eval_trampoline_delayed ~threshold:8 );
  ]

type case = {
  name : string;
  expected : float;
  reference : unit -> float;
  candidates : (string * (unit -> float)) list;
}

let origin = Coord.of_fn (fun _ -> 0)

let dead_env =
  {
    Eval_common.Env.load = (fun _ _ -> assert false);
    load_index = (fun _ _ -> assert false);
  }

(* [h*1000+w], matching js/probe/probe_expr.ml's Stage 4 max-pool case: cheap,
   deterministic, and strictly increasing in row-major order so a max-pool
   case's expected value is a closed form. *)
let load_env =
  {
    Eval_common.Env.load =
      (fun _ c ->
        Ok (float_of_int ((Coord.get c Axis.H * 1000) + Coord.get c Axis.W)));
    load_index = (fun _ _ -> assert false);
  }

let pos n = Index.clamp_low (Index.const n)

let make_case ~name ~expected ~env ?local ?local_at ?scan
    ?(scan_meter = fun () -> None) ?reducer ?on_reduction ~output e =
  let run (ev : evaluator) =
    Err.or_raise ~pp_error:Eval_common.pp_error
      (ev ?local ?local_at ?scan ?scan_meter:(scan_meter ()) ?reducer
         ?on_reduction env ~output e)
  in
  {
    name;
    expected;
    reference = (fun () -> run Eval.value);
    candidates =
      List.map
        (fun (cname, ev) -> (cname, fun () -> run ev))
        candidate_evaluators;
  }

(* ---- arithmetic / guard / unary --------------------------------------- *)

let arithmetic =
  make_case ~name:"arithmetic" ~expected:14. ~env:dead_env ~output:origin
    (Value.add (Value.const 2.) (Value.mul (Value.const 3.) (Value.const 4.)))

let select_true =
  make_case ~name:"select_true" ~expected:10. ~env:dead_env ~output:origin
    (Value.select
       (Bool.value_lt (Value.const 1.) (Value.const 2.))
       (Value.const 10.) (Value.const 20.))

let select_false =
  make_case ~name:"select_false" ~expected:20. ~env:dead_env ~output:origin
    (Value.select
       (Bool.value_lt (Value.const 5.) (Value.const 2.))
       (Value.const 10.) (Value.const 20.))

let index_eq_true =
  make_case ~name:"index_eq_true" ~expected:1. ~env:dead_env ~output:origin
    (Value.select
       (Bool.index_eq (Index.const 3) (Index.const 3))
       (Value.const 1.) (Value.const 0.))

(* [Select] generalized to [int64 t], guarded by [Bool.i64_lt] -- exercises
   the new [eval_i64]/[guard] mutual dependency ([Select]'s [int64 t]
   branches, [I64_lt]'s [int64 t] operands) through every candidate, not
   just the reference evaluator. *)
let i64_select_true =
  make_case ~name:"i64_select_true" ~expected:10. ~env:dead_env ~output:origin
    (Value.i64_to_float
       (Value.select
          (Bool.i64_lt (Value.i64_const 1L) (Value.i64_const 2L))
          (Value.i64_const 10L) (Value.i64_const 20L)))

(* [Bool.i64_eq], and an [I64_lt] operand that is itself a [Float_to_i64] --
   the carrier-crossing case [Value.eval_i64]'s own doc comment calls out,
   here reached through [guard] rather than a standalone [eval_i64] call. *)
let i64_eq_through_cast =
  make_case ~name:"i64_eq_through_cast" ~expected:1. ~env:dead_env
    ~output:origin
    (Value.i64_to_float
       (Value.select
          (Bool.i64_eq
             (Value.float_to_i64 (Value.const 5.))
             (Value.i64_const 5L))
          (Value.i64_const 1L) (Value.i64_const 0L)))

let round_f32_case =
  make_case ~name:"round_f32" ~expected:1. ~env:dead_env ~output:origin
    (Value.round_f32 (Value.exp (Value.const 0.)))

(* ---- Local / Local_at -------------------------------------------------- *)

let local_var = Builder.run Builder.fresh_local

let local_case =
  make_case ~name:"local" ~expected:42. ~env:dead_env ~output:origin
    ~local:(fun v -> if Local_var.equal v local_var then Some 42. else None)
    (Value.local local_var)

let local_at_case =
  make_case ~name:"local_at" ~expected:7. ~env:dead_env ~output:origin
    ~local_at:(fun v p ->
      if Local_var.equal v local_var && p = 5 then Some 7. else None)
    (Value.local_at local_var (pos 5))

(* [Local_scan_at] reads a caller-supplied trace TABLE (unlike inline
   [Scan_at], which fills its own rows), so it needs no [Scan_meter] -- it
   is a leaf transition in every candidate, uncovered by any of the other
   17 cases. *)
let local_scan_id = Builder.run Builder.fresh_local

let local_scan_at_case =
  make_case ~name:"local_scan_at" ~expected:23. ~env:dead_env ~output:origin
    ~scan:(fun v ~row ~lane ->
      if Local_var.equal v local_scan_id then
        Err.return (float_of_int ((row * 10) + lane))
      else assert false)
    (Value.local_scan_at local_scan_id ~row:(pos 2) ~lane:(pos 3))

(* ---- Load / Value_of_index / Intrinsic --------------------------------- *)

let load_source = Source.create 0

let load_case =
  make_case ~name:"load" ~expected:3004. ~env:load_env ~output:origin
    (Value.load load_source
       (Coord.of_fn (function
         | Axis.H -> pos 3
         | Axis.W -> pos 4
         | _ -> Index.zero)))

let value_of_index_case =
  make_case ~name:"value_of_index" ~expected:5. ~env:dead_env ~output:origin
    (Value.value_of_index (Index.add (Index.const 2) (Index.const 3)))

(* [Index.max]/[Index.min]: [Index.add] above is the only [Index.t]
   arithmetic node any of the other 17 cases exercises. *)
let index_max_case =
  make_case ~name:"index_max" ~expected:7. ~env:dead_env ~output:origin
    (Value.value_of_index (Index.max (Index.const 3) (Index.const 7)))

let index_min_case =
  make_case ~name:"index_min" ~expected:3. ~env:dead_env ~output:origin
    (Value.value_of_index (Index.min (Index.const 3) (Index.const 7)))

let max_pool_source = Source.create 1

let max_pool_case =
  make_case ~name:"max_pool" ~expected:4004. ~env:load_env ~output:origin
    (Value.intrinsic
       (Err.or_raise ~pp_error:Intrinsic.pp_error
          (Intrinsic.max_pool ~source:max_pool_source ~in_h:5 ~in_w:5
             ~kernel_h:5 ~kernel_w:5 ~stride_h:1 ~stride_w:1 ~pad_h:0 ~pad_w:0
             ~out:(Coord.of_fn Index.output) ~result:Intrinsic.Max_pool.Value)))

(* ---- Reduce ------------------------------------------------------------- *)

let reduce_sum =
  make_case ~name:"reduce_sum" ~expected:10. ~env:dead_env ~output:origin
    (Builder.run
       (Builder.reduction ~kind:Reduction.Sum ~lo:Index.zero ~hi:(Index.const 5)
          (fun i -> Builder.return (Value.value_of_index (Index.of_position i)))))

let reduce_max =
  make_case ~name:"reduce_max" ~expected:4. ~env:dead_env ~output:origin
    (Builder.run
       (Builder.reduction ~kind:Reduction.Max ~lo:Index.zero ~hi:(Index.const 5)
          (fun i -> Builder.return (Value.value_of_index (Index.of_position i)))))

let reduce_empty =
  make_case ~name:"reduce_empty" ~expected:0. ~env:dead_env ~output:origin
    (Builder.run
       (Builder.reduction ~kind:Reduction.Sum ~lo:(pos 3) ~hi:(Index.const 3)
          (fun i -> Builder.return (Value.value_of_index (Index.of_position i)))))

(* sum_{i=0}^{2} sum_{j=0}^{2} (i+j) = 3+6+9 = 18. *)
let reduce_nested =
  make_case ~name:"reduce_nested" ~expected:18. ~env:dead_env ~output:origin
    (Builder.run
       (Builder.reduction ~kind:Reduction.Sum ~lo:Index.zero ~hi:(Index.const 3)
          (fun i ->
            Builder.reduction ~kind:Reduction.Sum ~lo:Index.zero
              ~hi:(Index.const 3) (fun j ->
                Builder.return
                  (Value.add
                     (Value.value_of_index (Index.of_position i))
                     (Value.value_of_index (Index.of_position j)))))))

let reduce_long =
  make_case ~name:"reduce_long" ~expected:1225. ~env:dead_env ~output:origin
    (Builder.run
       (Builder.reduction ~kind:Reduction.Sum ~lo:Index.zero
          ~hi:(Index.const 50) (fun i ->
            Builder.return (Value.value_of_index (Index.of_position i)))))

(* ---- Scan_at ------------------------------------------------------------ *)

let scan_limits = Scan_limits.default

(* row0[l] = l; row_{r+1}[l] = row_r[l] + 1, so row_r[l] = l + r. *)
let scan_descriptor =
  Err.or_raise ~pp_error:Scan.pp_error
    (Builder.run
       (Builder.scan ~limits:scan_limits ~width:3 ~steps:4
          ~init:(fun ~lane ->
            Builder.return (Value.value_of_index (Index.of_position lane)))
          ~update:(fun ~step:_ ~lane ~previous_at ->
            Builder.return (Value.add (previous_at lane) (Value.const 1.)))))

let scan_case =
  make_case ~name:"scan_row2_lane1" ~expected:3. ~env:dead_env ~output:origin
    ~scan_meter:(fun () -> Some (Scan_meter.create ~limits:scan_limits))
    (Value.scan_at scan_descriptor ~row:(pos 2) ~lane:(pos 1))

let scan_row0_case =
  make_case ~name:"scan_row0_lane2" ~expected:2. ~env:dead_env ~output:origin
    ~scan_meter:(fun () -> Some (Scan_meter.create ~limits:scan_limits))
    (Value.scan_at scan_descriptor ~row:Index.zero ~lane:(pos 2))

(* Inner: a fixed, one-row scan (no update body ever charged). Outer's
   [update] reads it at a constant projection and adds it to [previous_at] --
   two scans simultaneously reserved during the outer's update, so the
   nesting-peak reservation, the LIFO cleanup order and the [prev]
   resolver's save/restore across the inner scan's own lifetime are all
   exercised, not just each scan in isolation. *)
let inner_scan_descriptor =
  Err.or_raise ~pp_error:Scan.pp_error
    (Builder.run
       (Builder.scan ~limits:scan_limits ~width:2 ~steps:0
          ~init:(fun ~lane:_ -> Builder.return (Value.const 100.))
          ~update:(fun ~step:_ ~lane ~previous_at ->
            Builder.return (previous_at lane))))

let inner_scan_at =
  Value.scan_at inner_scan_descriptor ~row:Index.zero ~lane:Index.zero

(* row0 = [0,1]; row_{r+1}[l] = row_r[l] + 100; row2 = [200,201]. *)
let nested_scan_descriptor =
  Err.or_raise ~pp_error:Scan.pp_error
    (Builder.run
       (Builder.scan ~limits:scan_limits ~width:2 ~steps:2
          ~init:(fun ~lane ->
            Builder.return (Value.value_of_index (Index.of_position lane)))
          ~update:(fun ~step:_ ~lane ~previous_at ->
            Builder.return (Value.add (previous_at lane) inner_scan_at))))

let nested_scan_case =
  make_case ~name:"nested_scan_row2_lane1" ~expected:201. ~env:dead_env
    ~output:origin
    ~scan_meter:(fun () -> Some (Scan_meter.create ~limits:scan_limits))
    (Value.scan_at nested_scan_descriptor ~row:(pos 2) ~lane:(pos 1))

let cases =
  [
    arithmetic;
    select_true;
    select_false;
    index_eq_true;
    i64_select_true;
    i64_eq_through_cast;
    round_f32_case;
    local_case;
    local_at_case;
    local_scan_at_case;
    load_case;
    value_of_index_case;
    index_max_case;
    index_min_case;
    max_pool_case;
    reduce_sum;
    reduce_max;
    reduce_empty;
    reduce_nested;
    reduce_long;
    scan_case;
    scan_row0_case;
    nested_scan_case;
  ]
