open Ssa_ir
open Loop_ir_test
open Loop_fixtures
open Loop_programs

(* The optimizer against failures. The sweep's walks never fail, so they cannot
   tell a pass that dropped a check it needed from one that dropped a check it
   did not: these fixtures do. Each is run unoptimized and optimized, and the
   two must agree on the value or on the failure, row and payload. *)

let alias = Ssa_effects.Distinct_buffers
let optimize p = fst (Ssa_opt.run ~alias p)

let lowered plan =
  Err.or_raise ~pp_error:Ssa_lower.Ssa_lower_plan.pp_error
    (Ssa_lower.Ssa_lower_plan.lower plan)

let same a b =
  match (Err.payload a, Err.payload b) with
  | Ok x, Ok y -> Tensor_id.Map.equal Loop_ir.Loop_check.tensors_equal x y
  | Error x, Error y -> Stdlib.( = ) x y
  | Ok _, Error _ | Error _, Ok _ -> false

let describe = function
  | Ok _ -> "ok"
  | Error e -> Fmt.str "%a" Ssa_lower.Ssa_exec.pp_error e

let check name plan ~bind =
  let program = lowered plan in
  let optimized = optimize program in
  let before = Ssa_lower.Ssa_exec.run plan program ~bind in
  let after = Ssa_lower.Ssa_exec.run plan optimized ~bind in
  let stats p = Ssa_stats.of_program p in
  Fmt.pr "%s: %s; optimized agrees: %b; checks %d -> %d@." name
    (describe (Err.payload after))
    (same before after) (stats program).Ssa_stats.checked
    (stats optimized).Ssa_stats.checked

let default kernel = Fusion_plan.default kernel
let const_pos n = Expr.Index.assume_position (Expr.Index.const n)

let%expect_test "an access that can fail keeps the check that reports it" =
  check "shifted load" (default shifted_kernel)
    ~bind:(Ssa_fixtures.bind_data ~shape:(shape_w 4) [| 1.; 2.; 3.; 4. |]);
  check "vector read past its extent"
    (default (region_kernel_of (vector_program ~extent:3 ~pick:(pick 3))))
    ~bind:Ssa_scan_data.bind;
  check "cached projection past the trace"
    (default
       (region_kernel_of
          (trace_program_at ~steps:2 ~row:(const_pos 3) ~lane:(const_pos 0))))
    ~bind:Ssa_scan_data.bind;
  check "inline projection past the width"
    (default
       (region_kernel_of
          (Region_program.pixel
             (inline_scan_body_at ~steps:2 ~row:(const_pos 0)
                ~lane:(const_pos 3)))))
    ~bind:Ssa_scan_data.bind;
  check "gather out of range" (default gather_kernel)
    ~bind:
      (i64_bind ~floats:[| 10.; 20.; 30.; 40. |] ~cells:[| 4L; 0L; 0L; 0L |]);
  check "i64 division by zero"
    (default
       (i64_body_kernel (Expr.Value.i64_div i64_here (Expr.Value.i64_const 0L))))
    ~bind:(i64_bind ~floats:[| 0.; 0.; 0.; 0. |] ~cells:[| 7L; 7L; 7L; 7L |]);
  check "float to i64 of NaN"
    (default
       (i64_body_kernel
          (Expr.Value.float_to_i64
             (Expr.Value.load (Expr_bridge.source_of_id (tid 0)) out_coord))))
    ~bind:(i64_bind ~floats:[| nan; 1.; 2.; 3. |] ~cells:[| 0L; 0L; 0L; 0L |]);
  (* an index that leaves the domain: the check on the scale stays, since w can
     reach 3 and 3 * 2^30 does not fit *)
  check "index overflow" (default overflow_kernel)
    ~bind:(Ssa_fixtures.bind_data ~shape:(shape_w 4) [| 0.; 0.; 0.; 0. |]);
  [%expect
    {|
    shifted load: t0: coordinate W = 4 is outside the buffer, at (0,0,0,0,4,0); optimized agrees: true; checks 2 -> 1
    vector read past its extent: local #0 is unbound here; optimized agrees: true; checks 3 -> 1
    cached projection past the trace: scan row 3 out of range [0,3); optimized agrees: true; checks 11 -> 1
    inline projection past the width: scan lane 3 out of range [0,3) at row 0; optimized agrees: true; checks 5 -> 1
    gather out of range: gather index 4 is outside [-4, 4); optimized agrees: true; checks 4 -> 3
    i64 division by zero: I64 division by zero; optimized agrees: true; checks 2 -> 1
    float to i64 of NaN: Float-to-I64 cast of NaN; optimized agrees: true; checks 2 -> 1
    index overflow: index mul overflows on 1073741824 and 2; optimized agrees: true; checks 1 -> 1 |}]

(* ---- the passes on small programs -------------------------------------------- *)

module B = Ssa_builder
open Ssa_ir_test.Ssa_fixtures

let row_buffer id n format role = buffer id ~h:1L ~w:n format role
let idx bld n = B.index bld (Int64.of_int n)

let load_at bld id i =
  B.load_f64 bld (buf id) ~decode:Ssa_op.Decode.F32_to_f64
    (at bld ~h:(idx bld 0) ~w:i)

let store_at bld id i x =
  B.store_f64 bld (buf id) ~encode:Ssa_op.Encode.F32_round
    (at bld ~h:(idx bld 0) ~w:i)
    x

let bufs =
  [
    row_buffer 0 4L Ssa_format.F32 Ssa_buffer.Input;
    row_buffer 1 4L Ssa_format.F32 Ssa_buffer.Output;
  ]

let program f = build ~buffers:bufs f

let show ?(passes = [ Ssa_opt.simplify; Ssa_opt.guards; Ssa_opt.simplify ]) p =
  let q, _ = Ssa_opt.run ~alias ~passes p in
  Fmt.pr "%s@." (Ssa_pp.to_string q);
  (* both versions run the same on these inputs *)
  let run p =
    let out = Array.make 4 0. in
    let memory = memory [ (0, floats [| 1.; 2.; 3.; 4. |]); (1, floats out) ] in
    (Err.payload (Ssa_interp.run p ~memory), Array.copy out)
  in
  Fmt.pr "same result: %b@." (run p = run q)

let%expect_test
    "folding evaluates constants, checked operations included, and reconnects \
     the chain" =
  show
    (program (fun bld ->
         let two = B.f64 bld 2. and three = B.f64 bld 3. in
         let six = B.f64_binary bld Expr.Value.Mul two three in
         let i = B.index_add bld (idx bld 1) (idx bld 2) in
         let pick = B.select bld (B.pred bld true) six (B.f64 bld 9.) in
         store_at bld 1 i pick));
  [%expect
    {|
    buffer b0 in f32 [1, 1, 1, 1, 4, 1]
    buffer b1 out f32 [1, 1, 1, 1, 4, 1]
    entry(%0:effect) {
      (%1:f64) = const 0x1.8p+2:f64
      (%2:index) = const 3:index
      (%3:index) = const 0:index
      (%4:effect) = store.f32_round b1[%3, %3, %3, %3, %2, %3], %1 effect %0
      yield %4
    }

    same result: true |}]

let%expect_test "pure subexpressions are shared, effects and marks never are" =
  show
    (program (fun bld ->
         let x = load_at bld 0 (idx bld 1) in
         let a = B.f64_binary bld Expr.Value.Add x x in
         let b = B.f64_binary bld Expr.Value.Add x x in
         B.mark bld Ssa_mark.Reduction;
         B.mark bld Ssa_mark.Reduction;
         store_at bld 1 (idx bld 0) (B.f64_binary bld Expr.Value.Mul a b)));
  [%expect
    {|
    buffer b0 in f32 [1, 1, 1, 1, 4, 1]
    buffer b1 out f32 [1, 1, 1, 1, 4, 1]
    entry(%0:effect) {
      (%1:index) = const 1:index
      (%2:index) = const 0:index
      (%3:f64, %4:effect) = load.in_bounds.f32_to_f64 b0[%2, %2, %2, %2, %1, %2] effect %0
      (%5:f64) = float.add %3, %3
      (%6:effect) = mark reduction effect %4
      (%7:effect) = mark reduction effect %6
      (%8:f64) = float.mul %5, %5
      (%9:effect) = store.f32_round b1[%2, %2, %2, %2, %2, %2], %8 effect %7
      yield %9
    }

    same result: true |}]

let%expect_test "a dead pure value goes; a dead checked operation stays" =
  show
    (program (fun bld ->
         let _dead =
           B.f64_binary bld Expr.Value.Add (B.f64 bld 1.) (B.f64 bld 2.)
         in
         let unused_load = load_at bld 0 (idx bld 9) in
         ignore unused_load;
         store_at bld 1 (idx bld 0) (B.f64 bld 1.)));
  [%expect
    {|
    buffer b0 in f32 [1, 1, 1, 1, 4, 1]
    buffer b1 out f32 [1, 1, 1, 1, 4, 1]
    entry(%0:effect) {
      (%1:f64) = const 0x1p+0:f64
      (%2:index) = const 9:index
      (%3:index) = const 0:index
      (%4:f64, %5:effect) = load.f32_to_f64 b0[%3, %3, %3, %3, %2, %3] effect %0
      (%6:effect) = store.f32_round b1[%3, %3, %3, %3, %3, %3], %1 effect %5
      yield %6
    }

    same result: true |}]

let%expect_test
    "ranges remove checks that cannot fire and keep the one that can" =
  show
    (program (fun bld ->
         let B.Nil =
           B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 3) ~init:B.Nil
             (fun bld i B.Nil ->
               (* i + 1 is in [1, 3] inside a buffer of 4: no check needed *)
               let next = B.index_add bld i (idx bld 1) in
               store_at bld 1 i (load_at bld 0 next);
               (* i + 2 can be 4: the check stays *)
               let far = B.index_add bld i (idx bld 2) in
               store_at bld 1 i (load_at bld 0 far);
               B.Nil)
         in
         ()));
  [%expect
    {|
    buffer b0 in f32 [1, 1, 1, 1, 4, 1]
    buffer b1 out f32 [1, 1, 1, 1, 4, 1]
    entry(%0:effect) {
      (%1:index) = const 3:index
      (%2:index) = const 0:index
      (%3:effect) = for %2 to %1 step 1 iter(%4:index, %5:effect := %0) {
        (%6:index) = const 1:index
        (%7:index) = index.add_in_domain %4, %6
        (%8:f64, %9:effect) = load.in_bounds.f32_to_f64 b0[%2, %2, %2, %2, %7, %2] effect %5
        (%10:effect) = store.f32_round b1[%2, %2, %2, %2, %4, %2], %8 effect %9
        (%11:index) = const 2:index
        (%12:index) = index.add_in_domain %4, %11
        (%13:f64, %14:effect) = load.f32_to_f64 b0[%2, %2, %2, %2, %12, %2] effect %10
        (%15:effect) = store.f32_round b1[%2, %2, %2, %2, %4, %2], %13 effect %14
        yield %15
      }
      yield %3
    }

    same result: true |}]

let%expect_test
    "a loop that cannot run vanishes, and one that runs once is its body" =
  show
    (program (fun bld ->
         let B.Nil =
           B.for_ bld ~lo:(idx bld 2) ~hi:(idx bld 2) ~init:B.Nil
             (fun bld _ B.Nil ->
               store_at bld 1 (idx bld 0) (B.f64 bld 7.);
               B.Nil)
         in
         let B.Nil =
           B.for_ bld ~lo:(idx bld 1) ~hi:(idx bld 2) ~init:B.Nil
             (fun bld i B.Nil ->
               B.mark bld Ssa_mark.Key;
               store_at bld 1 i (load_at bld 0 i);
               B.Nil)
         in
         ()));
  [%expect
    {|
    buffer b0 in f32 [1, 1, 1, 1, 4, 1]
    buffer b1 out f32 [1, 1, 1, 1, 4, 1]
    entry(%0:effect) {
      (%1:index) = const 1:index
      (%2:effect) = mark key effect %0
      (%3:index) = const 0:index
      (%4:f64, %5:effect) = load.in_bounds.f32_to_f64 b0[%3, %3, %3, %3, %1, %3] effect %2
      (%6:effect) = store.f32_round b1[%3, %3, %3, %3, %1, %3], %4 effect %5
      yield %6
    }

    same result: true |}]

let nest f =
  program (fun bld ->
      let B.Nil =
        B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 2) ~init:B.Nil
          (fun bld i B.Nil ->
            f bld i;
            B.Nil)
      in
      ())

let hoisting ~alias p =
  let q, _ =
    Ssa_opt.run ~alias
      ~passes:
        [
          Ssa_opt.guards;
          Ssa_opt.simplify;
          Ssa_opt.hoist ~alias;
          Ssa_opt.simplify;
        ]
      p
  in
  Fmt.pr "%s@." (Ssa_pp.to_string q)

let%expect_test "an invariant pure value leaves the loop, a dependent one stays"
    =
  hoisting ~alias:Ssa_effects.Conservative
    (nest (fun bld i ->
         let base = B.index_scale bld 2L (idx bld 1) in
         let here = B.index_add bld base i in
         ignore (B.f64_binary bld Expr.Value.Add (B.f64 bld 1.) (B.f64 bld 2.));
         store_at bld 1 here (B.f64 bld 3.)));
  [%expect
    {|
    buffer b0 in f32 [1, 1, 1, 1, 4, 1]
    buffer b1 out f32 [1, 1, 1, 1, 4, 1]
    entry(%0:effect) {
      (%1:index) = const 2:index
      (%2:index) = const 0:index
      (%3:index) = const 1:index
      (%4:index) = index.scale_in_domain 2, %3
      (%5:f64) = const 0x1.8p+1:f64
      (%6:effect) = for %2 to %1 step 1 iter(%7:index, %8:effect := %0) {
        (%9:index) = index.add_in_domain %4, %7
        (%10:effect) = store.f32_round b1[%2, %2, %2, %2, %9, %2], %5 effect %8
        yield %10
      }
      yield %6
    } |}]

let%expect_test
    "a load leaves only a loop that runs, never writes it, and cannot fail" =
  let body bld i =
    let x = load_at bld 0 (idx bld 1) in
    store_at bld 1 i x
  in
  (* the loop writes b1, the load reads b0: under the conservative policy the
     two may overlap, under distinct buffers they cannot *)
  hoisting ~alias:Ssa_effects.Conservative (nest body);
  hoisting ~alias:Ssa_effects.Distinct_buffers (nest body);
  (* an empty loop never reads: nothing moves *)
  hoisting ~alias:Ssa_effects.Distinct_buffers
    (program (fun bld ->
         let B.Nil =
           B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 0) ~init:B.Nil
             (fun bld i B.Nil ->
               store_at bld 1 i (load_at bld 0 (idx bld 1));
               B.Nil)
         in
         ()));
  (* a loop that writes the buffer it reads keeps the read inside *)
  hoisting ~alias:Ssa_effects.Distinct_buffers
    (nest (fun bld i ->
         let x = load_at bld 1 (idx bld 1) in
         store_at bld 1 i x));
  [%expect
    {|
    buffer b0 in f32 [1, 1, 1, 1, 4, 1]
    buffer b1 out f32 [1, 1, 1, 1, 4, 1]
    entry(%0:effect) {
      (%1:index) = const 2:index
      (%2:index) = const 0:index
      (%3:index) = const 1:index
      (%4:effect) = for %2 to %1 step 1 iter(%5:index, %6:effect := %0) {
        (%7:f64, %8:effect) = load.in_bounds.f32_to_f64 b0[%2, %2, %2, %2, %3, %2] effect %6
        (%9:effect) = store.f32_round b1[%2, %2, %2, %2, %5, %2], %7 effect %8
        yield %9
      }
      yield %4
    }

    buffer b0 in f32 [1, 1, 1, 1, 4, 1]
    buffer b1 out f32 [1, 1, 1, 1, 4, 1]
    entry(%0:effect) {
      (%1:index) = const 2:index
      (%2:index) = const 0:index
      (%3:index) = const 1:index
      (%4:f64, %5:effect) = load.in_bounds.f32_to_f64 b0[%2, %2, %2, %2, %3, %2] effect %0
      (%6:effect) = for %2 to %1 step 1 iter(%7:index, %8:effect := %5) {
        (%9:effect) = store.f32_round b1[%2, %2, %2, %2, %7, %2], %4 effect %8
        yield %9
      }
      yield %6
    }

    buffer b0 in f32 [1, 1, 1, 1, 4, 1]
    buffer b1 out f32 [1, 1, 1, 1, 4, 1]
    entry(%0:effect) {
      yield %0
    }

    buffer b0 in f32 [1, 1, 1, 1, 4, 1]
    buffer b1 out f32 [1, 1, 1, 1, 4, 1]
    entry(%0:effect) {
      (%1:index) = const 2:index
      (%2:index) = const 0:index
      (%3:index) = const 1:index
      (%4:effect) = for %2 to %1 step 1 iter(%5:index, %6:effect := %0) {
        (%7:f64, %8:effect) = load.in_bounds.f32_to_f64 b1[%2, %2, %2, %2, %3, %2] effect %6
        (%9:effect) = store.f32_round b1[%2, %2, %2, %2, %5, %2], %7 effect %8
        yield %9
      }
      yield %4
    } |}]

let%expect_test "an earlier load is shared until something may write its buffer"
    =
  let p =
    program (fun bld ->
        let a = load_at bld 0 (idx bld 1) in
        let b = load_at bld 0 (idx bld 1) in
        store_at bld 1 (idx bld 0) (B.f64_binary bld Expr.Value.Add a b);
        let c = load_at bld 1 (idx bld 0) in
        store_at bld 1 (idx bld 1) c;
        let d = load_at bld 1 (idx bld 0) in
        store_at bld 1 (idx bld 2) d)
  in
  let q, _ =
    Ssa_opt.run ~alias ~passes:[ Ssa_opt.share ~alias; Ssa_opt.simplify ] p
  in
  Fmt.pr "%s@." (Ssa_pp.to_string q);
  (* a store to the same buffer between two loads of it separates them *)
  let r =
    program (fun bld ->
        let a = load_at bld 1 (idx bld 0) in
        store_at bld 1 (idx bld 0)
          (B.f64_binary bld Expr.Value.Add a (B.f64 bld 1.));
        let b = load_at bld 1 (idx bld 0) in
        store_at bld 1 (idx bld 1) b)
  in
  let s, _ = Ssa_opt.run ~alias ~passes:[ Ssa_opt.share ~alias ] r in
  Fmt.pr "loads %d -> %d@." (Ssa_stats.of_program r).Ssa_stats.loads
    (Ssa_stats.of_program s).Ssa_stats.loads;
  [%expect
    {|
    buffer b0 in f32 [1, 1, 1, 1, 4, 1]
    buffer b1 out f32 [1, 1, 1, 1, 4, 1]
    entry(%0:effect) {
      (%1:index) = const 1:index
      (%2:index) = const 0:index
      (%3:f64, %4:effect) = load.f32_to_f64 b0[%2, %2, %2, %2, %1, %2] effect %0
      (%5:f64) = float.add %3, %3
      (%6:effect) = store.f32_round b1[%2, %2, %2, %2, %2, %2], %5 effect %4
      (%7:f64, %8:effect) = load.f32_to_f64 b1[%2, %2, %2, %2, %2, %2] effect %6
      (%9:effect) = store.f32_round b1[%2, %2, %2, %2, %1, %2], %7 effect %8
      (%10:f64, %11:effect) = load.f32_to_f64 b1[%2, %2, %2, %2, %2, %2] effect %9
      (%12:index) = const 2:index
      (%13:effect) = store.f32_round b1[%2, %2, %2, %2, %12, %2], %10 effect %11
      yield %13
    }

    loads 2 -> 2 |}]

let%expect_test
    "sharing keeps what differs: the same operator with another literal" =
  show
    (program (fun bld ->
         let i = B.index_scale bld 2L (idx bld 1) in
         ignore i;
         let x =
           B.index_of_i64 bld (B.float_to_i64 bld (load_at bld 0 (idx bld 0)))
         in
         let a = B.index_floor_div bld 2L x in
         let b = B.index_floor_div bld 3L x in
         store_at bld 1 (idx bld 0) (B.index_to_f64 bld a);
         store_at bld 1 (idx bld 1) (B.index_to_f64 bld b)));
  [%expect
    {|
    buffer b0 in f32 [1, 1, 1, 1, 4, 1]
    buffer b1 out f32 [1, 1, 1, 1, 4, 1]
    entry(%0:effect) {
      (%1:index) = const 1:index
      (%2:index) = const 0:index
      (%3:f64, %4:effect) = load.in_bounds.f32_to_f64 b0[%2, %2, %2, %2, %2, %2] effect %0
      (%5:i64, %6:effect) = float.to_i64 %3 effect %4
      (%7:index, %8:effect) = index.of_i64 %5 effect %6
      (%9:index) = index.floor_div 2, %7
      (%10:index) = index.floor_div 3, %7
      (%11:f64) = convert.index_to_f64 %9
      (%12:effect) = store.f32_round b1[%2, %2, %2, %2, %2, %2], %11 effect %8
      (%13:f64) = convert.index_to_f64 %10
      (%14:effect) = store.f32_round b1[%2, %2, %2, %2, %1, %2], %13 effect %12
      yield %14
    }

    same result: true |}]

(* A loop whose bounds the analysis cannot fix may run no times: reading before
   it would add a read that was never made. *)
let%expect_test "a read is not hoisted out of a loop that might not run" =
  let p =
    program (fun bld ->
        let n =
          B.index_of_i64 bld (B.float_to_i64 bld (load_at bld 0 (idx bld 0)))
        in
        let B.Nil =
          B.for_ bld ~lo:(idx bld 0) ~hi:n ~init:B.Nil (fun bld i B.Nil ->
              store_at bld 1 i (load_at bld 0 (idx bld 1));
              B.Nil)
        in
        ())
  in
  let q, _ =
    Ssa_opt.run ~alias
      ~passes:[ Ssa_opt.guards; Ssa_opt.hoist ~alias; Ssa_opt.simplify ]
      p
  in
  (* the first input cell is 0.: the loop runs no times *)
  let loads program =
    let c = Ssa_interp.Counters.create () in
    let memory =
      memory [ (0, floats [| 0.; 2.; 3.; 4. |]); (1, floats (Array.make 4 0.)) ]
    in
    ignore (Ssa_interp.run ~counters:c program ~memory);
    Ssa_interp.Counters.loads c
  in
  Fmt.pr "loads %d -> %d@." (loads p) (loads q);
  [%expect {| loads 1 -> 1 |}]

(* An operation that returned a value did not fail, so the value is inside the
   domain: [x + 1 + 0] needs no check on the second sum, though [x + 1] alone
   spans one past the top. *)
let%expect_test
    "a checked result keeps only the range that survived its own check" =
  show
    (program (fun bld ->
         let x =
           B.index_of_i64 bld (B.float_to_i64 bld (load_at bld 0 (idx bld 0)))
         in
         let y = B.index_add bld x (idx bld 1) in
         let z = B.index_add bld y (idx bld 0) in
         store_at bld 1 (idx bld 0) (B.index_to_f64 bld z)));
  [%expect
    {|
    buffer b0 in f32 [1, 1, 1, 1, 4, 1]
    buffer b1 out f32 [1, 1, 1, 1, 4, 1]
    entry(%0:effect) {
      (%1:index) = const 0:index
      (%2:f64, %3:effect) = load.in_bounds.f32_to_f64 b0[%1, %1, %1, %1, %1, %1] effect %0
      (%4:i64, %5:effect) = float.to_i64 %2 effect %3
      (%6:index, %7:effect) = index.of_i64 %4 effect %5
      (%8:index) = const 1:index
      (%9:index, %10:effect) = index.add %6, %8 effect %7
      (%11:index) = index.add_in_domain %9, %1
      (%12:f64) = convert.index_to_f64 %11
      (%13:effect) = store.f32_round b1[%1, %1, %1, %1, %1, %1], %12 effect %10
      yield %13
    }

    same result: true |}]

(* The ranges a loaded integer gets from each total index operation are claims;
   a wrong one removes a check that data can trip. Each case reads at an index
   computed from the first input cell, which the test sets to land outside, so
   the failure must survive optimization. *)
let%expect_test "index operations keep their ranges sound for any data" =
  let case name data f =
    let p =
      program (fun bld ->
          let x =
            B.index_of_i64 bld (B.float_to_i64 bld (load_at bld 0 (idx bld 0)))
          in
          store_at bld 1 (idx bld 0) (load_at bld 0 (f bld x)))
    in
    let q = optimize p in
    let run program =
      let memory = memory [ (0, floats data); (1, floats (Array.make 4 0.)) ] in
      Err.payload (Ssa_interp.run program ~memory)
    in
    let describe = function
      | Ok () -> "ok"
      | Error e -> Fmt.str "%a" Ssa_interp.pp_error e
    in
    Fmt.pr "%-18s %s | optimized: %s@." name
      (describe (run p))
      (describe (run q))
  in
  case "min(x, 0), x = -3" [| -3.; 1.; 2.; 3. |] (fun bld x ->
      B.index_min bld x (idx bld 0));
  case "max(x, 3), x = 7" [| 7.; 1.; 2.; 3. |] (fun bld x ->
      B.index_max bld x (idx bld 3));
  case "scale -1, x = 2" [| 2.; 1.; 2.; 3. |] (fun bld x ->
      B.index_scale bld (-1L) x);
  case "floor_div 2, x = 9" [| 9.; 1.; 2.; 3. |] (fun bld x ->
      B.index_floor_div bld 2L x);
  case "ceil_div 2, x = -9" [| -9.; 1.; 2.; 3. |] (fun bld x ->
      B.index_ceil_div bld 2L x);
  case "clamp_low, x = 7" [| 7.; 1.; 2.; 3. |] (fun bld x ->
      B.index_clamp_low bld x);
  case "clamp_low, x = -7" [| -7.; 1.; 2.; 3. |] (fun bld x ->
      B.index_clamp_low bld x);
  case "x + x, x = 3" [| 3.; 1.; 2.; 3. |] (fun bld x -> B.index_add bld x x);
  [%expect
    {|
    min(x, 0), x = -3  t0: coordinate W = -3 is outside the buffer, at (0,0,0,0,-3,0) | optimized: t0: coordinate W = -3 is outside the buffer, at (0,0,0,0,-3,0)
    max(x, 3), x = 7   t0: coordinate W = 7 is outside the buffer, at (0,0,0,0,7,0) | optimized: t0: coordinate W = 7 is outside the buffer, at (0,0,0,0,7,0)
    scale -1, x = 2    t0: coordinate W = -2 is outside the buffer, at (0,0,0,0,-2,0) | optimized: t0: coordinate W = -2 is outside the buffer, at (0,0,0,0,-2,0)
    floor_div 2, x = 9 t0: coordinate W = 4 is outside the buffer, at (0,0,0,0,4,0) | optimized: t0: coordinate W = 4 is outside the buffer, at (0,0,0,0,4,0)
    ceil_div 2, x = -9 t0: coordinate W = -4 is outside the buffer, at (0,0,0,0,-4,0) | optimized: t0: coordinate W = -4 is outside the buffer, at (0,0,0,0,-4,0)
    clamp_low, x = 7   t0: coordinate W = 7 is outside the buffer, at (0,0,0,0,7,0) | optimized: t0: coordinate W = 7 is outside the buffer, at (0,0,0,0,7,0)
    clamp_low, x = -7  ok | optimized: ok
    x + x, x = 3       t0: coordinate W = 6 is outside the buffer, at (0,0,0,0,6,0) | optimized: t0: coordinate W = 6 is outside the buffer, at (0,0,0,0,6,0) |}]

(* A load before a loop is not the load inside it: the body repeats, and what it
   writes is what its next iteration reads. *)
let%expect_test "a load is not shared into a loop that writes what it reads" =
  let p =
    program (fun bld ->
        let before = load_at bld 1 (idx bld 0) in
        let B.Nil =
          B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 3) ~init:B.Nil
            (fun bld _ B.Nil ->
              let x = load_at bld 1 (idx bld 0) in
              store_at bld 1 (idx bld 0)
                (B.f64_binary bld Expr.Value.Add x (B.f64 bld 1.));
              B.Nil)
        in
        store_at bld 1 (idx bld 1) before)
  in
  let q, _ =
    Ssa_opt.run ~alias ~passes:[ Ssa_opt.share ~alias; Ssa_opt.simplify ] p
  in
  let run program =
    let out = Array.make 4 0. in
    let memory = memory [ (0, floats (Array.make 4 0.)); (1, floats out) ] in
    ignore (Ssa_interp.run program ~memory);
    Array.to_list out
  in
  let show l = String.concat " " (List.map (Fmt.str "%g") l) in
  Fmt.pr "unoptimized: %s@.optimized:   %s@." (show (run p)) (show (run q));
  [%expect {|
    unoptimized: 3 0 0 0
    optimized:   3 0 0 0 |}]

(* A check with no read behind it exists only to fail: it goes only when the
   ranges prove it cannot. *)
let%expect_test "a lone access check is dropped only when it cannot fire" =
  let check_at w =
    program (fun bld ->
        B.check_access bld (buf 0) (at bld ~h:(idx bld 0) ~w:(idx bld w));
        store_at bld 1 (idx bld 0) (B.f64 bld 1.))
  in
  List.iter
    (fun w ->
      let p = check_at w in
      let q = optimize p in
      let run program =
        let memory =
          memory
            [ (0, floats (Array.make 4 0.)); (1, floats (Array.make 4 0.)) ]
        in
        match Err.payload (Ssa_interp.run program ~memory) with
        | Ok () -> "ok"
        | Error e -> Fmt.str "%a" Ssa_interp.pp_error e
      in
      Fmt.pr "w = %d: %s | optimized (%d checks): %s@." w (run p)
        (Ssa_stats.of_program q).Ssa_stats.checked (run q))
    [ 3; 4 ];
  [%expect
    {|
    w = 3: ok | optimized (0 checks): ok
    w = 4: t0: coordinate W = 4 is outside the buffer, at (0,0,0,0,4,0) | optimized (1 checks): t0: coordinate W = 4 is outside the buffer, at (0,0,0,0,4,0) |}]
