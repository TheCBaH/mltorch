open Ssa_bridge
open Ssa_fixtures
open Ssa_ir

(* Register blocking over rows: that it fires on a vectorized dense kernel, that
   the blocked program agrees with the reference and the unblocked program, and
   that the logical work is unchanged. *)

let alias = Ssa_effects.Distinct_buffers
let target = Ssa_target.wasm128

let lowered plan =
  Err.or_raise ~pp_error:Ssa_lower.Ssa_lower_plan.pp_error
    (Ssa_lower.Ssa_lower_plan.lower plan)

let counters plan program ~bind =
  let c = Ssa_interp.Counters.create () in
  ignore (Ssa_lower.Ssa_exec.run ~counters:c plan program ~bind);
  c

let marks c = List.map (Ssa_interp.Counters.mark c) Ssa_mark.all

let check name ~rows plan ~bind =
  let vectorized = fst (Ssa_opt.run ~alias ~target (lowered plan)) in
  let blocked, n = Ssa_opt_rows.program ~rows ~policy:alias vectorized in
  (match Err.payload (Ssa_verify.check blocked) with
  | Ok () -> ()
  | Error e -> Fmt.pr "does not verify: %a@." Ssa_verify.pp_error e);
  let verdict p =
    Fmt.str "%a" Ssa_check.pp_verdict
      (Ssa_check.run ~prepare:(fun _ -> p) plan ~bind)
  in
  Fmt.pr "%s rows=%d: blocked=%d; reference %s; same work: %b@." name rows n
    (verdict blocked)
    (marks (counters plan vectorized ~bind)
    = marks (counters plan blocked ~bind))

let%expect_test "matmul: row blocks, remainders and too few rows" =
  List.iter
    (fun (m, k, n, rows) ->
      let plan = Fusion_plan.default (matmul_kernel ~m ~k ~n) in
      let a = operand 3 (m * k) and b = operand 5 (k * n) in
      check
        (Fmt.str "matmul %dx%dx%d" m k n)
        ~rows plan
        ~bind:(matmul_bind ~m ~k ~n ~a ~b))
    [ (4, 5, 8, 2); (3, 7, 11, 2); (5, 2, 17, 4); (1, 3, 8, 2); (6, 3, 8, 3) ];
  [%expect
    {|
    matmul 4x5x8 rows=2: blocked=1; reference agree; same work: true
    matmul 3x7x11 rows=2: blocked=0; reference agree; same work: true
    matmul 5x2x17 rows=4: blocked=0; reference agree; same work: true
    matmul 1x3x8 rows=2: blocked=0; reference agree; same work: true
    matmul 6x3x8 rows=3: blocked=1; reference agree; same work: true |}]

(* ---- what the pass refuses, by the condition that fails ---------------------- *)

module B = Ssa_builder
open Ssa_ir_test.Ssa_fixtures

let bufs =
  [
    buffer 0 ~h:8L ~w:3L Ssa_format.F32 Ssa_buffer.Input;
    buffer 1 ~h:3L ~w:8L Ssa_format.F32 Ssa_buffer.Input;
    buffer 2 ~h:8L ~w:8L Ssa_format.F32 Ssa_buffer.Output;
  ]

let idx bld n = B.index bld (Int64.of_int n)

let load_at bld id ~h ~w =
  B.load_f64 bld (buf id) ~decode:Ssa_op.Decode.F32_to_f64 (at bld ~h ~w)

(* for h in [0, rows): for w in [0, 8): out[h, w] = sum over k of a[h, k] *
   b[k, w], with the stored row, the reduction's bound and the row count
   varied. *)
let dense ?(rows = 6) ?(store_row = fun _ h -> h)
    ?(sum_hi = fun bld _ -> idx bld 3) () =
  match
    Err.payload
      (B.program ~buffers:bufs (fun bld ->
           let _ =
             B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld rows) ~init:B.Nil
               (fun bld h B.Nil ->
                 let _ =
                   B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 8) ~init:B.Nil
                     (fun bld w B.Nil ->
                       let s =
                         B.ordered_sum bld ~lo:(idx bld 0) ~hi:(sum_hi bld h)
                           ~seed:(B.f64 bld 0.) (fun bld k ->
                             B.f64_binary bld Expr.Value.Mul
                               (load_at bld 0 ~h ~w:k) (load_at bld 1 ~h:k ~w))
                       in
                       B.store_f64 bld (buf 2) ~encode:Ssa_op.Encode.F32_round
                         (at bld ~h:(store_row bld h) ~w)
                         s;
                       B.Nil)
                 in
                 B.Nil)
           in
           ()))
  with
  | Ok p -> p
  | Error e -> Fmt.failwith "%a" Ssa_verify.pp_error e

let decisions ?(rows = 2) ?(policy = alias) ?(vectorize = true) p =
  let p =
    fst
      (Ssa_opt.run ~alias
         ~passes:
           ([ Ssa_opt.simplify; Ssa_opt.guards; Ssa_opt.simplify ]
           @ [ Ssa_opt.hoist ~alias ]
           @
           if vectorize then
             [ Ssa_opt.vectorize ~alias ~target:(Ssa_target.forced target) ]
           else [])
         p)
  in
  match Ssa_opt_rows.analyze ~rows ~policy p with
  | [] -> Fmt.pr "not a candidate@."
  | l ->
      List.iter
        (function
          | _, Ok n -> Fmt.pr "blocks by %d@." n
          | _, Error r -> Fmt.pr "refused: %s@." (Ssa_opt_rows.refusal_name r))
        l

let%expect_test "row blocking is refused by the condition that fails" =
  decisions (dense ());
  (* the rows left over the block size are not a refusal *)
  decisions ~rows:4 (dense ~rows:6 ());
  (* the reduction's length depends on the row *)
  decisions (dense ~sum_hi:(fun bld h -> B.index_min bld h (idx bld 3)) ());
  (* every row stores to the same cells *)
  decisions (dense ~store_row:(fun bld _ -> idx bld 0) ());
  (* the output may be what the rows read *)
  decisions ~policy:Ssa_effects.Conservative (dense ());
  (* a block of one row is no block *)
  decisions ~rows:1 (dense ());
  (* a loop over scalar outputs is the column blocking's, not this pass's *)
  decisions ~vectorize:false (dense ());
  [%expect
    {|
    blocks by 2
    blocks by 4
    refused: bounds vary with the row
    refused: stores are not per row
    refused: the body reads a buffer it may write
    refused: fewer rows than one block
    not a candidate |}]

(* The vectorizer never keeps a loop that can fail, so a vector loop that can is
   built by hand: its scalar operand goes through a float-to-integer conversion. *)
let failing_vector_rows () =
  let lanes = Ssa_type.Lanes.of_int 4 in
  let raw (type a) (x : a B.value) : Ssa_value.t = (x :> Ssa_value.t) in
  let z bld = raw (idx bld 0) in
  let coord bld ~h ~w =
    Expr.Coord.make ~n:(z bld) ~t:(z bld) ~d:(z bld) ~h:(raw h) ~w:(raw w)
      ~c:(z bld)
  in
  match
    Err.payload
      (B.program ~buffers:bufs (fun bld ->
           let _ =
             B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 6) ~init:B.Nil
               (fun bld h B.Nil ->
                 let _ =
                   B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 5) ~init:B.Nil
                     (fun bld w B.Nil ->
                       let seed = B.vec_splat bld ~lanes (raw (B.f64 bld 0.)) in
                       let s =
                         B.ordered_sum_dyn bld ~lo:(idx bld 0) ~hi:(idx bld 3)
                           ~seed (fun bld k ->
                             let a = load_at bld 0 ~h ~w:k in
                             let a = B.i64_to_f64 bld (B.float_to_i64 bld a) in
                             let a = B.vec_splat bld ~lanes (raw a) in
                             let b =
                               B.vec_load bld (buf 1)
                                 ~decode:Ssa_op.Decode.F32_to_f64 ~lanes
                                 ~steps:
                                   (Expr.Coord.make ~n:0L ~t:0L ~d:0L ~h:0L
                                      ~w:1L ~c:0L)
                                 (coord bld ~h:k ~w)
                             in
                             B.lanewise bld
                               (Ssa_op.Float_binary (Expr.Value.Mul, a, b)))
                       in
                       B.vec_store bld (buf 2) ~encode:Ssa_op.Encode.F32_round
                         ~lanes
                         ~steps:
                           (Expr.Coord.make ~n:0L ~t:0L ~d:0L ~h:0L ~w:1L ~c:0L)
                         (coord bld ~h ~w) s;
                       B.Nil)
                 in
                 B.Nil)
           in
           ()))
  with
  | Ok p -> p
  | Error e -> Fmt.failwith "%a" Ssa_verify.pp_error e

let%expect_test "a vector loop that can fail is refused" =
  (match
     Ssa_opt_rows.analyze ~rows:2 ~policy:alias (failing_vector_rows ())
   with
  | [] -> Fmt.pr "not a candidate@."
  | l ->
      List.iter
        (function
          | _, Ok n -> Fmt.pr "blocks by %d@." n
          | _, Error r -> Fmt.pr "refused: %s@." (Ssa_opt_rows.refusal_name r))
        l);
  [%expect {| refused: the body can fail |}]

(* ---- the planner ------------------------------------------------------------- *)

let neon = Ssa_target.forced Ssa_target.neon128

let%expect_test "planned: neon blocks two rows, bitwise the unblocked plan" =
  List.iter
    (fun (m, k, n) ->
      let plan = Fusion_plan.default (matmul_kernel ~m ~k ~n) in
      let a = operand 3 (m * k) and b = operand 5 (k * n) in
      let bind = matmul_bind ~m ~k ~n ~a ~b in
      List.iter
        (fun (name, numerics) ->
          let run target =
            let verdict, p =
              Ssa_check.run_planned ~alias ~numerics ~target plan ~bind
            in
            ( Fmt.str "%a" Ssa_check.pp_verdict verdict,
              match p with Some p -> p.Ssa_plan.blocked | None -> -1 )
          in
          let v1, b1 = run (Ssa_target.with_row_block 1 neon)
          and v2, b2 = run neon in
          Fmt.pr "matmul %dx%dx%d %s: unblocked %s (%d), blocked %s (%d)@." m k
            n name v1 b1 v2 b2)
        [
          ("reference", Ssa_numerics.Reference_f64);
          ("ordered", Ssa_numerics.Simd_fp32_ordered);
          ("relaxed", Ssa_numerics.Simd_fp32_relaxed);
        ])
    [ (4, 5, 8); (6, 3, 16); (5, 4, 8); (7, 2, 16); (3, 2, 11) ];
  [%expect
    {|
    matmul 4x5x8 reference: unblocked agree (0), blocked agree (0)
    matmul 4x5x8 ordered: unblocked agree (0), blocked agree (0)
    matmul 4x5x8 relaxed: unblocked agree (0), blocked agree (0)
    matmul 6x3x16 reference: unblocked agree (0), blocked agree (0)
    matmul 6x3x16 ordered: unblocked agree (0), blocked agree (1)
    matmul 6x3x16 relaxed: unblocked agree (0), blocked agree (1)
    matmul 5x4x8 reference: unblocked agree (0), blocked agree (0)
    matmul 5x4x8 ordered: unblocked agree (0), blocked agree (0)
    matmul 5x4x8 relaxed: unblocked agree (0), blocked agree (0)
    matmul 7x2x16 reference: unblocked agree (0), blocked agree (0)
    matmul 7x2x16 ordered: unblocked agree (0), blocked agree (1)
    matmul 7x2x16 relaxed: unblocked agree (0), blocked agree (1)
    matmul 3x2x11 reference: unblocked agree (0), blocked agree (0)
    matmul 3x2x11 ordered: unblocked agree (0), blocked agree (0)
    matmul 3x2x11 relaxed: unblocked agree (0), blocked agree (0) |}]
