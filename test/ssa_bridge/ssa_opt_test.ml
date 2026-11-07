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
    (* run first: a tuple's components are evaluated right to left *)
    let r = Err.payload (Ssa_interp.run p ~memory) in
    (r, Array.copy out)
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

(* ---- local reads -------------------------------------------------------------- *)

(* How many local reads sit at the entry, before any loop, after hoisting; and
   whether the hoisted program computes the same as the original: the output
   cells, or the failure, on the given first input cell. *)
let hoisting_local ?(first = 5.) name p =
  let q, _ =
    Ssa_opt.run ~alias
      ~passes:[ Ssa_opt.guards; Ssa_opt.hoist ~alias; Ssa_opt.simplify ]
      p
  in
  let entry_reads (p : Ssa_program.t) =
    List.length
      (List.filter
         (function
           | Ssa_stmt.Instr { Ssa_instr.op = Ssa_op.Local_read _; _ } -> true
           | _ -> false)
         p.Ssa_program.entry.Ssa_region.body)
  in
  let run p =
    let out = Array.make 4 0. in
    let memory =
      memory [ (0, floats [| first; 2.; 3.; 4. |]); (1, floats out) ]
    in
    let r =
      match Err.payload (Ssa_interp.run p ~memory) with
      | Ok _ -> "ok"
      | Error e -> Fmt.str "%a" Ssa_interp.pp_error e
      | exception Invalid_argument defect -> defect
    in
    (r, Array.to_list out)
  in
  let r, out = run q in
  Fmt.pr "%s: reads at entry %d -> %d; %s [%s]; same result: %b@." name
    (entry_reads p) (entry_reads q) r
    (String.concat "; " (List.map (Fmt.str "%g") out))
    (run p = run q)

(* a one-slot object written before the loop with the first input cell *)
let with_local ?(slots = 1L) bld =
  let l = B.local_alloc bld ~slots in
  B.local_write bld l (idx bld 0) (load_at bld 0 (idx bld 0));
  l

let loop2 bld f =
  let B.Nil =
    B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 2) ~init:B.Nil (fun bld i B.Nil ->
        f bld i;
        B.Nil)
  in
  ()

let%expect_test "a local read leaves a loop under the four conditions only" =
  hoisting_local "invariant one-cell local"
    (program (fun bld ->
         let l = with_local bld in
         loop2 bld (fun bld i ->
             store_at bld 1 i (B.local_read bld l (idx bld 0)))));
  (* the loop writes the object: each trip reads what the last one wrote *)
  hoisting_local "written in the loop"
    (program (fun bld ->
         let l = with_local bld in
         loop2 bld (fun bld i ->
             let x = B.local_read bld l (idx bld 0) in
             store_at bld 1 i x;
             B.local_write bld l (idx bld 0)
               (B.f64_binary bld Expr.Value.Add x (B.f64 bld 1.)))));
  (* a loop that may run no times over a cell nothing wrote: the original reads
     nothing; a hoisted read would reach the unset cell *)
  hoisting_local ~first:0. "possibly empty loop, unset cell"
    (program (fun bld ->
         let l = B.local_alloc bld ~slots:1L in
         let n =
           B.index_of_i64 bld (B.float_to_i64 bld (load_at bld 0 (idx bld 0)))
         in
         let B.Nil =
           B.for_ bld ~lo:(idx bld 0) ~hi:n ~init:B.Nil (fun bld i B.Nil ->
               store_at bld 1 i (B.local_read bld l (idx bld 0));
               B.Nil)
         in
         ()));
  (* the index is past the object's one slot: the read can fail *)
  hoisting_local "index outside the object"
    (program (fun bld ->
         let l = with_local bld in
         loop2 bld (fun bld i ->
             store_at bld 1 i (B.local_read bld l (idx bld 1)))));
  (* an invariant index that is not a constant: no range proof is tried *)
  hoisting_local ~first:0. "non-constant index"
    (program (fun bld ->
         let l = with_local bld in
         let k =
           B.index_of_i64 bld (B.float_to_i64 bld (load_at bld 0 (idx bld 0)))
         in
         loop2 bld (fun bld i -> store_at bld 1 i (B.local_read bld l k))));
  [%expect
    {|
    invariant one-cell local: reads at entry 0 -> 1; ok [5; 5; 0; 0]; same result: true
    written in the loop: reads at entry 0 -> 0; ok [5; 6; 0; 0]; same result: true
    possibly empty loop, unset cell: reads at entry 0 -> 0; ok [0; 0; 0; 0]; same result: true
    index outside the object: reads at entry 0 -> 0; Ssa_interp: a local read outside its object [0; 0; 0; 0]; same result: true
    non-constant index: reads at entry 0 -> 0; ok [0; 0; 0; 0]; same result: true |}]

(* ---- collapsing dense rows ------------------------------------------------------ *)

let grid id ~h ~w role = buffer id ~h ~w Ssa_format.F32 role

(* out[h, w] = f(a[h, w], b[...]) over an [rows x cols] nest; [b_at] says
   where [b] is read. *)
let nest2 ~rows ~cols ~b_ext ~b_at =
  build
    ~buffers:
      [
        grid 0 ~h:2L ~w:6L Ssa_buffer.Input;
        grid 1 ~h:(fst b_ext) ~w:(snd b_ext) Ssa_buffer.Input;
        grid 2 ~h:2L ~w:6L Ssa_buffer.Output;
      ]
    (fun bld ->
      let B.Nil =
        B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld rows) ~init:B.Nil
          (fun bld r B.Nil ->
            let B.Nil =
              B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld cols) ~init:B.Nil
                (fun bld c B.Nil ->
                  let ld id h w =
                    B.load_f64 bld (buf id) ~decode:Ssa_op.Decode.F32_to_f64
                      (at bld ~h ~w)
                  in
                  let bh, bw = b_at bld r c in
                  B.store_f64 bld (buf 2) ~encode:Ssa_op.Encode.F32_round
                    (at bld ~h:r ~w:c)
                    (B.f64_binary bld Expr.Value.Add (ld 0 r c) (ld 1 bh bw));
                  B.Nil)
            in
            B.Nil)
      in
      ())

let collapsing name p =
  let passes =
    [ Ssa_opt.guards; Ssa_opt.simplify; Ssa_opt.hoist ~alias; Ssa_opt.collapse ]
  in
  let q, _ = Ssa_opt.run ~alias ~passes p in
  let loops (p : Ssa_program.t) =
    let rec count (r : Ssa_region.t) =
      List.fold_left
        (fun n -> function
          | Ssa_stmt.For { body; _ } -> n + 1 + count body | _ -> n)
        0 r.Ssa_region.body
    in
    count p.Ssa_program.entry
  in
  let run p =
    let out = Array.make 12 0. in
    let memory =
      memory
        [
          (0, floats (Array.init 12 (fun i -> float_of_int (i * i))));
          (1, floats (Array.init 12 (fun i -> 0.5 +. float_of_int i)));
          (2, floats out);
        ]
    in
    (* run first: a tuple's components are evaluated right to left *)
    let r = Err.payload (Ssa_interp.run p ~memory) in
    (r, Array.to_list out)
  in
  Fmt.pr "%s: loops %d -> %d; extents [%s]; same result: %b@." name (loops p)
    (loops q)
    (String.concat "; "
       (List.map
          (fun (b : Ssa_buffer.t) ->
            String.concat "x"
              (List.map Int64.to_string
                 [
                   Expr.Coord.get b.Ssa_buffer.extents Expr.Axis.H;
                   Expr.Coord.get b.Ssa_buffer.extents Expr.Axis.W;
                 ]))
          q.Ssa_program.buffers))
    (run p = run q)

let%expect_test "dense rows collapse into one loop, and only they" =
  collapsing "both rows whole"
    (nest2 ~rows:2 ~cols:6 ~b_ext:(2L, 6L) ~b_at:(fun _ r c -> (r, c)));
  collapsing "b broadcast over rows"
    (nest2 ~rows:2 ~cols:6 ~b_ext:(1L, 6L) ~b_at:(fun bld _ c -> (idx bld 0, c)));
  collapsing "b transposed"
    (nest2 ~rows:2 ~cols:2 ~b_ext:(6L, 2L) ~b_at:(fun _ r c -> (c, r)));
  collapsing "partial rows"
    (nest2 ~rows:2 ~cols:4 ~b_ext:(2L, 6L) ~b_at:(fun _ r c -> (r, c)));
  [%expect
    {|
    both rows whole: loops 2 -> 1; extents [1x12; 1x12; 1x12]; same result: true
    b broadcast over rows: loops 2 -> 2; extents [2x6; 1x6; 2x6]; same result: true
    b transposed: loops 2 -> 2; extents [2x6; 6x2; 2x6]; same result: true
    partial rows: loops 2 -> 2; extents [2x6; 2x6; 2x6]; same result: true |}]

(* ---- unroll and interleave ------------------------------------------------------ *)

(* x[w] = x[w - 16] + 1 over w in [16, 112): a recurrence at a distance of
   one vector of sixteen. Vectorized at sixteen lanes each trip reads what the
   previous trip wrote; two copies interleaved would read before the earlier
   copy stores, so the pass must leave the loop as it is. *)
let recurrence_at d =
  build
    ~buffers:[ buffer 0 ~h:1L ~w:112L Ssa_format.F32 Ssa_buffer.Output ]
    (fun bld ->
      let B.Nil =
        B.for_ bld ~lo:(idx bld 16) ~hi:(idx bld 112) ~init:B.Nil
          (fun bld w B.Nil ->
            let back = B.index_add bld w (B.index bld (Int64.neg d)) in
            let x =
              B.load_f64 bld (buf 0) ~decode:Ssa_op.Decode.F32_to_f64
                (at bld ~h:(idx bld 0) ~w:back)
            in
            B.store_f64 bld (buf 0) ~encode:Ssa_op.Encode.F32_round
              (at bld ~h:(idx bld 0) ~w)
              (B.f64_binary bld Expr.Value.Add x (B.f64 bld 1.));
            B.Nil)
      in
      ())

let unrolled name ~unroll p =
  let target =
    Ssa_target.with_unroll unroll (Ssa_target.forced Ssa_target.neon128)
  in
  match Ssa_plan.fp32_vector ~alias ~target p with
  | Error _ -> Fmt.pr "%s: not admitted@." name
  | Ok plan ->
      let q = plan.Ssa_plan.program in
      let steps =
        let rec go (r : Ssa_region.t) =
          List.concat_map
            (function
              | Ssa_stmt.For { step; body; _ } -> step :: go body | _ -> [])
            r.Ssa_region.body
        in
        go q.Ssa_program.entry
      in
      let run p =
        let cells = Array.init 112 (fun i -> float_of_int (i mod 16)) in
        let memory = memory [ (0, floats cells) ] in
        let r = Err.payload (Ssa_interp.run p ~memory) in
        (r, Array.to_list cells)
      in
      List.iter
        (fun (d : Ssa_vectorize.Decision.t) ->
          Fmt.pr "  %s@."
            (match d.Ssa_vectorize.Decision.outcome with
            | Ssa_vectorize.Decision.Vectorized -> "vectorized"
            | Ssa_vectorize.Decision.Kept_scalar r ->
                Ssa_vectorize.Reason.name r))
        plan.Ssa_plan.vectorized;
      Fmt.pr "%s: loop steps [%s]; same result: %b@." name
        (String.concat "; " (List.map Int64.to_string steps))
        (run p = run q)

(* The vectorizer refuses every loop carried through memory (above), so a
   vector loop whose store feeds a later trip's load is built directly:
   x[w + 16 ..) = x[w ..), a vector step of 16. Interleaving two copies would
   load the second copy's x[w + 16 ..) before the first stores it. *)
let vector_recurrence =
  let lanes = Ssa_type.Lanes.of_int 16 in
  let p =
    build
      ~buffers:[ buffer 0 ~h:1L ~w:64L Ssa_format.F32 Ssa_buffer.Output ]
      (fun bld ->
        let zero = idx bld 0 in
        let coord (w : Ssa_type.index B.value) =
          let z = (zero :> Ssa_value.t) in
          Expr.Coord.make ~n:z ~t:z ~d:z ~h:z ~w:(w :> Ssa_value.t) ~c:z
        in
        let steps = Expr.Coord.make ~n:0L ~t:0L ~d:0L ~h:0L ~w:1L ~c:0L in
        let B.Nil =
          B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 33) ~init:B.Nil
            (fun bld w B.Nil ->
              let ahead = B.index_add bld w (B.index bld 16L) in
              let x =
                B.vec_load bld (buf 0) ~decode:Ssa_op.Decode.F32_exact ~lanes
                  ~steps (coord w)
              in
              B.vec_store bld (buf 0) ~encode:Ssa_op.Encode.F32_exact ~lanes
                ~steps (coord ahead) x;
              B.Nil)
        in
        ())
  in
  let entry = p.Ssa_program.entry in
  {
    p with
    Ssa_program.entry =
      {
        entry with
        Ssa_region.body =
          List.map
            (function
              | Ssa_stmt.For f -> Ssa_stmt.For { f with step = 16L } | s -> s)
            entry.Ssa_region.body;
      };
  }

let interleaved name p =
  (* the guards first, as in the pipeline: the index arithmetic in domain *)
  let p, _ =
    Ssa_opt.run ~alias ~passes:[ Ssa_opt.guards; Ssa_opt.simplify ] p
  in
  let q, _ = Ssa_opt_unroll.pass ~unroll:2 ~alias p in
  let steps (p : Ssa_program.t) =
    List.filter_map
      (function Ssa_stmt.For { step; _ } -> Some step | _ -> None)
      p.Ssa_program.entry.Ssa_region.body
  in
  let run p =
    let cells = Array.init 64 float_of_int in
    let r =
      Err.payload (Ssa_interp.run p ~memory:(memory [ (0, floats cells) ]))
    in
    (r, Array.to_list cells)
  in
  Fmt.pr "%s: steps [%s]; same result: %b@." name
    (String.concat "; " (List.map Int64.to_string (steps q)))
    (run p = run q)

let%expect_test "a vector loop carried through memory keeps its trips in order"
    =
  interleaved "x[w + 16..) = x[w..)" vector_recurrence;
  [%expect {| x[w + 16..) = x[w..): steps [16]; same result: true |}]

let%expect_test "a recurrence at a vector's distance is not interleaved" =
  List.iter
    (fun d ->
      unrolled (Fmt.str "distance %Ld, unroll 2" d) ~unroll:2 (recurrence_at d))
    [ 16L; 20L; 24L; 32L ];
  [%expect
    {|
      loop_carried
    distance 16, unroll 2: loop steps [1]; same result: true
      no_vector_form:load
    distance 20, unroll 2: loop steps [1]; same result: true
      no_vector_form:load
    distance 24, unroll 2: loop steps [1]; same result: true
      no_vector_form:load
    distance 32, unroll 2: loop steps [1]; same result: true |}]
