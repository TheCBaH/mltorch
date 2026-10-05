open Ssa_bridge
open Ssa_fixtures
open Ssa_ir
module B = Ssa_builder
module Fx = Ssa_ir_test.Ssa_fixtures

(* Independent-output vectorization: that it fires where it should, that every
   vector program agrees with the reference, the scalar program and its own
   scalar-lane expansion, that the logical work is unchanged, and that each
   condition it checks is the one that refuses. *)

let alias = Ssa_effects.Distinct_buffers
let target = Ssa_target.wasm128

let vectorized_pipeline ?(target = target) p =
  fst (Ssa_opt.run ~alias ~target p)

let counters plan program ~bind =
  let c = Ssa_interp.Counters.create () in
  ignore (Ssa_lower.Ssa_exec.run ~counters:c plan program ~bind);
  c

let marks c = List.map (Ssa_interp.Counters.mark c) Ssa_mark.all

let lowered plan =
  Err.or_raise ~pp_error:Ssa_lower.Ssa_lower_plan.pp_error
    (Ssa_lower.Ssa_lower_plan.lower plan)

let has_vector (p : Ssa_program.t) =
  let found = ref false in
  let rec region (r : Ssa_region.t) = List.iter stmt r.Ssa_region.body
  and stmt = function
    | Ssa_stmt.Instr
        { Ssa_instr.op = Ssa_op.Vec_load _ | Ssa_op.Vec_store _; _ } ->
        found := true
    | Ssa_stmt.Instr _ -> ()
    | Ssa_stmt.For { body; _ } | Ssa_stmt.Ordered_sum { body; _ } -> region body
    | Ssa_stmt.If { then_; else_; _ } ->
        region then_;
        region else_
  in
  region p.Ssa_program.entry;
  !found

(* One plan, three ways: the reference, the vectorized program, and the scalar
   lanes it expands to. *)
let check ?(target = target) name plan ~bind =
  let scalar = lowered plan in
  let vectorized = vectorized_pipeline ~target scalar in
  let expanded = Ssa_vec_expand.program vectorized in
  let verdict p =
    Fmt.str "%a" Ssa_check.pp_verdict
      (Ssa_check.run ~prepare:(fun _ -> p) plan ~bind)
  in
  let c_scalar = counters plan scalar ~bind
  and c_vector = counters plan vectorized ~bind in
  Fmt.pr "%s: vectorized=%b; reference %s; expansion %s; same work: %b@." name
    (has_vector vectorized) (verdict vectorized) (verdict expanded)
    (marks c_scalar = marks c_vector)

let%expect_test "matmul: full groups, remainders, and too short to vectorize" =
  List.iter
    (fun (m, k, n) ->
      let plan = Fusion_plan.default (matmul_kernel ~m ~k ~n) in
      let a = operand 3 (m * k) and b = operand 5 (k * n) in
      check
        (Fmt.str "matmul %dx%dx%d" m k n)
        plan
        ~bind:(matmul_bind ~m ~k ~n ~a ~b))
    [ (4, 5, 8); (3, 7, 11); (2, 3, 4); (2, 3, 3); (1, 1, 1); (5, 2, 17) ];
  [%expect
    {|
    matmul 4x5x8: vectorized=true; reference agree; expansion agree; same work: true
    matmul 3x7x11: vectorized=true; reference agree; expansion agree; same work: true
    matmul 2x3x4: vectorized=true; reference agree; expansion agree; same work: true
    matmul 2x3x3: vectorized=false; reference agree; expansion agree; same work: true
    matmul 1x1x1: vectorized=false; reference agree; expansion agree; same work: true
    matmul 5x2x17: vectorized=true; reference agree; expansion agree; same work: true |}]

(* ---- the condition that refuses ----------------------------------------------- *)

let bufs =
  [
    Fx.buffer 0 ~h:1L ~w:16L Ssa_format.F32 Ssa_buffer.Input;
    Fx.buffer 1 ~h:1L ~w:16L Ssa_format.F32 Ssa_buffer.Output;
    Fx.buffer 2 ~h:1L ~w:16L Ssa_format.F32 Ssa_buffer.Scratch;
  ]

let idx bld n = B.index bld (Int64.of_int n)

let load_at bld id i =
  B.load_f64 bld (Fx.buf id) ~decode:Ssa_op.Decode.F32_to_f64
    (Fx.at bld ~h:(idx bld 0) ~w:i)

let store_at bld id i x =
  B.store_f64 bld (Fx.buf id) ~encode:Ssa_op.Encode.F32_round
    (Fx.at bld ~h:(idx bld 0) ~w:i)
    x

(* [for w in [0, trips)] around [body], one iteration's statements. *)
let loop ?(trips = 8) body =
  Err.or_raise ~pp_error:Ssa_verify.pp_error
    (B.program ~buffers:bufs (fun bld ->
         let B.Nil =
           B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld trips) ~init:B.Nil
             (fun bld w B.Nil ->
               body bld w;
               B.Nil)
         in
         ()))

let decisions ?(alias = alias) ?(target = target) p =
  let q, report = Ssa_vectorize.program ~alias ~target p in
  ignore q;
  match report with
  | [] -> Fmt.pr "no loop@."
  | l ->
      List.iter
        (fun (d : Ssa_vectorize.Decision.t) ->
          match d.Ssa_vectorize.Decision.outcome with
          | Ssa_vectorize.Decision.Vectorized ->
              Fmt.pr "vectorized (%Ld trips)@." d.Ssa_vectorize.Decision.trips
          | Ssa_vectorize.Decision.Kept_scalar r ->
              Fmt.pr "scalar: %a@." Ssa_vectorize.Reason.pp r)
        l

(* The loop bodies below have already been through the guards and the hoisting
   the vectorizer relies on, as in the pipeline. *)
let prepared p =
  fst
    (Ssa_opt.run ~alias
       ~passes:
         [
           Ssa_opt.simplify;
           Ssa_opt.guards;
           Ssa_opt.simplify;
           Ssa_opt.hoist ~alias;
         ]
       p)

let%expect_test "each refusal names the condition that failed" =
  let d ?alias ?target p = decisions ?alias ?target (prepared p) in
  let scale bld w =
    store_at bld 1 w
      (B.f64_binary bld Expr.Value.Mul (load_at bld 0 w) (B.f64 bld 2.))
  in
  d (loop scale);
  (* caller has not said the buffers are distinct *)
  d ~alias:Ssa_effects.Conservative (loop scale);
  d ~target:Ssa_target.scalar (loop scale);
  d (loop ~trips:3 scale);
  d (loop ~trips:10 scale);
  (* the iteration reads the cell the next one writes *)
  d
    (loop (fun bld w ->
         store_at bld 2 w (load_at bld 2 (B.index_add bld w (idx bld 1)))));
  (* an index that is not a stride of the iteration *)
  d
    (loop (fun bld w ->
         store_at bld 1 w (load_at bld 0 (B.index_min bld w (idx bld 3)))));
  (* every lane would write one cell *)
  d (loop (fun bld w -> store_at bld 1 (idx bld 0) (load_at bld 0 w)));
  (* a branch *)
  d
    (loop (fun bld w ->
         let c =
           B.float_compare bld Ssa_op.Compare.Lt (load_at bld 0 w)
             (B.f64 bld 0.)
         in
         let v =
           B.if_ bld c
             ~then_:(fun _ -> B.Cons (B.f64 bld 1., B.Nil))
             ~else_:(fun _ -> B.Cons (B.f64 bld 2., B.Nil))
         in
         match v with B.Cons (x, B.Nil) -> store_at bld 1 w x));
  (* an operation that can fail *)
  d
    (loop (fun bld w ->
         let n = B.float_to_i64 bld (load_at bld 0 w) in
         store_at bld 1 w (B.i64_to_f64 bld n)));
  [%expect
    {|
    vectorized (8 trips)
    scalar: aliasing
    scalar: unprofitable
    scalar: too_short
    vectorized (10 trips)
    scalar: loop_carried
    scalar: non_affine_access
    scalar: store_through_broadcast
    scalar: branch
    scalar: no_vector_form:float.to_i64 |}]

(* ---- values: the vector program is the scalar program, cell for cell ----------- *)

let run_cells p input =
  let out = Array.make 16 0. in
  let counters = Ssa_interp.Counters.create () in
  let memory =
    Fx.memory
      [
        (0, Fx.floats (Array.copy input));
        (1, Fx.floats out);
        (2, Fx.floats (Array.make 16 0.));
      ]
  in
  match Err.payload (Ssa_interp.run ~counters p ~memory) with
  | Ok () -> Some (out, marks counters)
  | Error _ -> None

let forced = Ssa_target.forced Ssa_target.wasm128

let same_cells a b =
  match (a, b) with
  | Some (x, mx), Some (y, my) ->
      mx = my && Array.for_all2 Core.Float_bits.equal_portable x y
  | None, None -> true
  | _ -> false

let input =
  Array.init 16 (fun i ->
      Ssa_const.round_f32 ((float_of_int (i - 5) *. 0.37) +. 0.1))

let cases name p =
  let scalar = prepared p in
  let vec = fst (Ssa_opt.run ~alias ~target:forced p) in
  let expanded = Ssa_vec_expand.program vec in
  let reference = run_cells scalar input in
  Fmt.pr "%-22s vectorized=%b; vector=%b expansion=%b@." name (has_vector vec)
    (same_cells reference (run_cells vec input))
    (same_cells reference (run_cells expanded input))

let%expect_test "vector loops compute what their scalar iterations did" =
  let scale = B.f64_binary in
  cases "contiguous"
    (loop ~trips:12 (fun bld w ->
         store_at bld 1 w
           (scale bld Expr.Value.Mul (load_at bld 0 w) (B.f64 bld 2.))));
  cases "strided read"
    (loop ~trips:7 (fun bld w ->
         store_at bld 1 w (load_at bld 0 (B.index_scale bld 2L w))));
  cases "reversed read"
    (loop ~trips:10 (fun bld w ->
         let r = B.index_add bld (idx bld 15) (B.index_scale bld (-1L) w) in
         store_at bld 1 w (load_at bld 0 r)));
  cases "iota"
    (loop ~trips:9 (fun bld w -> store_at bld 1 w (B.index_to_f64 bld w)));
  cases "iota with a stride"
    (loop ~trips:9 (fun bld w ->
         store_at bld 1 w
           (B.index_to_f64 bld
              (B.index_add bld (idx bld 1) (B.index_scale bld 3L w)))));
  cases "mask and select"
    (loop ~trips:13 (fun bld w ->
         let x = load_at bld 0 w in
         let neg = B.float_compare bld Ssa_op.Compare.Lt x (B.f64 bld 0.) in
         store_at bld 1 w (B.select bld neg (B.f64 bld 0.) x)));
  cases "broadcast read"
    (loop ~trips:8 (fun bld w ->
         store_at bld 1 w
           (scale bld Expr.Value.Mul (load_at bld 0 w)
              (load_at bld 0 (idx bld 3)))));
  cases "inner reduction"
    (loop ~trips:8 (fun bld w ->
         let sum =
           B.ordered_sum bld ~lo:(idx bld 0) ~hi:(idx bld 4)
             ~seed:(B.f64 bld 0.) (fun bld k ->
               B.mark bld Ssa_mark.Reduction;
               scale bld Expr.Value.Mul
                 (load_at bld 0 (B.index_add bld w k))
                 (load_at bld 0 k))
         in
         store_at bld 1 w sum));
  cases "inner loop with a carried value"
    (loop ~trips:8 (fun bld w ->
         match
           B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 3)
             ~init:(B.Cons (B.f64 bld 1., B.Nil))
             (fun bld k (B.Cons (acc, B.Nil)) ->
               let x = load_at bld 0 (B.index_add bld w k) in
               B.Cons
                 ( scale bld Expr.Value.Add
                     (scale bld Expr.Value.Mul acc (B.f64 bld 0.5))
                     x,
                   B.Nil ))
         with
         | B.Cons (acc, B.Nil) -> store_at bld 1 w acc));
  [%expect
    {|
    contiguous             vectorized=true; vector=true expansion=true
    strided read           vectorized=true; vector=true expansion=true
    reversed read          vectorized=true; vector=true expansion=true
    iota                   vectorized=true; vector=true expansion=true
    iota with a stride     vectorized=true; vector=true expansion=true
    mask and select        vectorized=true; vector=true expansion=true
    broadcast read         vectorized=true; vector=true expansion=true
    inner reduction        vectorized=true; vector=true expansion=true
    inner loop with a carried value vectorized=true; vector=true expansion=true |}]

let%expect_test "more refusals" =
  let d ?alias ?target p = decisions ?alias ?target (prepared p) in
  (* an inner loop whose bounds are the iteration *)
  d
    (loop (fun bld w ->
         let sum =
           B.ordered_sum bld ~lo:(idx bld 0) ~hi:w ~seed:(B.f64 bld 0.)
             (fun bld k -> load_at bld 0 k)
         in
         store_at bld 1 w sum));
  (* a value carried from one iteration to the next *)
  (match
     Err.payload
       (B.program ~buffers:bufs (fun bld ->
            let (B.Cons (_, B.Nil)) =
              B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 8)
                ~init:(B.Cons (B.f64 bld 0., B.Nil))
                (fun bld w (B.Cons (acc, B.Nil)) ->
                  let v =
                    B.f64_binary bld Expr.Value.Add acc (load_at bld 0 w)
                  in
                  store_at bld 1 w v;
                  B.Cons (v, B.Nil))
            in
            ()))
   with
  | Ok p -> d p
  | Error _ -> Fmt.pr "does not build@.");
  (* a target that keeps loops out of vector bodies *)
  d ~target:Ssa_target.neon128
    (loop (fun bld w ->
         let sum =
           B.ordered_sum bld ~lo:(idx bld 0) ~hi:(idx bld 3)
             ~seed:(B.f64 bld 0.) (fun bld k ->
               load_at bld 0 (B.index_add bld w k))
         in
         store_at bld 1 w sum));
  [%expect
    {|
    scalar: varying_bounds
    scalar: carries_values
    scalar: inner_loops_declined |}]
