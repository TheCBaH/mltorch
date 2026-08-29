open Ssa_bridge
open Ssa_fixtures
open Ssa_ir_test.Ssa_fixtures
open Loop_ir
open Loop_ir_test
open Ssa_ir
module B = Ssa_builder

(* ---- Loop -> SSA: the bridge program, run against the reference ------------- *)

let bridged plan =
  match Err.payload (Loop_lower.lower plan) with
  | Error _ -> Error `Loop_refused
  | Ok loop -> (
      match Err.payload (Ssa_of_loop.convert loop) with
      | Error (`Unsupported u) -> Error (`Unsupported u)
      | Ok p -> Ok (loop, p))

let bridge_verdict plan ~bind =
  match bridged plan with
  | Error `Loop_refused -> Fmt.pr "loop refused@."
  | Error (`Unsupported u) ->
      Fmt.pr "refused: %a@." Ssa_of_loop_unsupported.pp u
  | Ok (_, p) ->
      Fmt.pr "%a@." Ssa_check.pp_verdict
        (Ssa_check.compare
           ~reference:(Kernel_eval.run_plan plan ~bind)
           ~ssa:(Ssa_lower.Ssa_exec.run plan p ~bind))

let%expect_test "the Loop program, converted to SSA, agrees with the reference"
    =
  let pointwise ~data kernel =
    bridge_verdict
      (Fusion_plan.default kernel)
      ~bind:(bind_data ~shape:(Loop_fixtures.shape_w 4) data)
  in
  pointwise ~data:[| -0.; 1.5; nan; 3. |] Loop_programs.kernel;
  pointwise ~data:[| 1e30; -1e30; 0.1; 16777217. |] Loop_programs.kernel;
  List.iter
    (fun (m, k, n) ->
      let a = operand 3 (m * k) and b = operand 5 (k * n) in
      bridge_verdict
        (Fusion_plan.default (matmul_kernel ~m ~k ~n))
        ~bind:(matmul_bind ~m ~k ~n ~a ~b))
    [ (1, 1, 1); (1, 3, 2); (5, 7, 3); (4, 4, 4) ];
  [%expect
    {|
    agree
    agree
    agree
    agree
    agree
    agree
    |}]

let%expect_test
    "a structured sum converts to an ordered sum, and an unknown guard is \
     refused" =
  let m = 3 and k = 4 and n = 2 in
  let plan = Fusion_plan.default (matmul_kernel ~m ~k ~n) in
  let a = operand 3 (m * k) and b = operand 5 (k * n) in
  let bind = matmul_bind ~m ~k ~n ~a ~b in
  (match Err.payload (Loop_lower.lower_structured plan) with
  | Error _ -> Fmt.pr "loop refused@."
  | Ok loop -> (
      Fmt.pr "structured sums: %d@." (Loop_sum.count loop);
      match Err.payload (Ssa_of_loop.convert loop) with
      | Error (`Unsupported u) ->
          Fmt.pr "refused: %a@." Ssa_of_loop_unsupported.pp u
      | Ok p ->
          let has_sum =
            let rec stmt = function
              | Ssa_stmt.Ordered_sum _ -> true
              | Ssa_stmt.For { body; _ } ->
                  List.exists stmt body.Ssa_region.body
              | Ssa_stmt.If { then_; else_; _ } ->
                  List.exists stmt then_.Ssa_region.body
                  || List.exists stmt else_.Ssa_region.body
              | Ssa_stmt.Instr _ -> false
            in
            List.exists stmt p.Ssa_program.entry.Ssa_region.body
          in
          Fmt.pr "ordered_sum present: %b@." has_sum;
          Fmt.pr "%a@." Ssa_check.pp_verdict
            (Ssa_check.compare
               ~reference:(Kernel_eval.run_plan plan ~bind)
               ~ssa:(Ssa_lower.Ssa_exec.run plan p ~bind));
          let counters = Ssa_interp.Counters.create () in
          ignore (Ssa_lower.Ssa_exec.run ~counters plan p ~bind);
          let loop_counters = Loop_interp.counters () in
          ignore (Loop_interp.run ~counters:loop_counters loop ~bind);
          Fmt.pr "reductions ssa=%d loop=%d@."
            (Ssa_interp.Counters.mark counters Ssa_mark.Reduction)
            loop_counters.Loop_interp.reductions));
  (* the shifted load keeps its bounds guard, which becomes a checked access *)
  bridge_verdict
    (Fusion_plan.default Loop_programs.shifted_kernel)
    ~bind:(bind_data ~shape:(Loop_fixtures.shape_w 4) [| 0.; 0.; 0.; 0. |]);
  (* a guard the converter has no recipe for is refused, never dropped *)
  let unknown_guard =
    Loop_fixtures.program
      [
        Loop_stmt.Fail_if
          ( Loop_bool.Index_lt (Loop_index.Const 0, Loop_index.Const 1),
            Loop_failure.I64_division_by_zero );
      ]
  in
  (match Err.payload (Ssa_of_loop.convert unknown_guard) with
  | Error (`Unsupported u) ->
      Fmt.pr "refused: %a@." Ssa_of_loop_unsupported.pp u
  | Ok _ -> Fmt.pr "converted@.");
  [%expect
    {|
    structured sums: 1
    ordered_sum present: true
    agree
    reductions ssa=24 loop=24
    agree on failure: coord_out_of_range
    refused: guard is not converted
    |}]

let%expect_test "an inlined producer keeps its f32 boundary through the bridge"
    =
  let bind =
    bind_data ~shape:(Loop_fixtures.shape_w 2) [| Loop_programs.two24; 1. |]
  in
  let fused, _ =
    Fusion_plan.plan (Loop_programs.chain ~outputs:[ Loop_fixtures.tid 2 ])
  in
  bridge_verdict fused ~bind;
  bridge_verdict
    (Fusion_plan.default
       (Loop_programs.chain
          ~outputs:[ Loop_fixtures.tid 1; Loop_fixtures.tid 2 ]))
    ~bind;
  [%expect {|
    agree
    agree
    |}]

let%expect_test
    "the bridge program does the same logical work as the Loop program" =
  let m = 3 and k = 4 and n = 2 in
  let plan = Fusion_plan.default (matmul_kernel ~m ~k ~n) in
  let bind =
    matmul_bind ~m ~k ~n ~a:(operand 3 (m * k)) ~b:(operand 5 (k * n))
  in
  (match bridged plan with
  | Ok (loop, p) ->
      let ssa_counters = Ssa_interp.Counters.create () in
      let loop_counters = Loop_interp.counters () in
      ignore (Ssa_lower.Ssa_exec.run ~counters:ssa_counters plan p ~bind);
      ignore (Loop_interp.run ~counters:loop_counters loop ~bind);
      Fmt.pr "reductions ssa=%d loop=%d@."
        (Ssa_interp.Counters.mark ssa_counters Ssa_mark.Reduction)
        loop_counters.Loop_interp.reductions
  | Error _ -> Fmt.pr "not converted@.");
  [%expect {| reductions ssa=24 loop=24 |}]

(* ---- SSA -> Loop: the projection, checked against the SSA interpreter -------- *)

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
    row_buffer 1 8L Ssa_format.F32 Ssa_buffer.Output;
  ]

let input = [| 1.5; 2.5; -3.; 0.25 |]

(* Both executors over one SSA program: what the SSA interpreter says, and what
   the Loop interpreter says of its projection. Failures are compared as rows,
   not as text; only the first four cells are read, because the Loop
   interpreter's fresh output holds unwritten cells as they came. *)
let both ?(cells = 4) p =
  let out = Array.make 8 0. in
  let memory = memory [ (0, floats (Array.copy input)); (1, floats out) ] in
  let ssa =
    match Err.payload (Ssa_interp.run p ~memory) with
    | Ok () -> Ok (Array.to_list (Array.sub out 0 cells))
    | Error (#Ssa_interp.failure as e) -> Error (e :> Kernel_eval.error)
    | Error (`Invalid_program _) -> invalid_arg "invalid program"
  in
  let shape = Loop_fixtures.shape_w 4 in
  let bind id =
    if Tensor_id.equal id (Loop_fixtures.tid 0) then
      Some
        (Loop_fixtures.f32_tensor shape (fun c ->
             input.((Vec6.offset shape c :> int))))
    else None
  in
  let loop =
    match Err.payload (Loop_interp.run (Loop_of_ssa.convert p) ~bind) with
    | Ok m ->
        let t = Tensor_id.Map.find (Loop_fixtures.tid 1) m in
        Ok
          (List.init cells (fun w ->
               Tensor.read t (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w ~c:0)))
    | Error e -> Error (e :> Kernel_eval.error)
  in
  (match ssa with
  | Ok cells ->
      Fmt.pr "ok %s" (String.concat " " (List.map (Fmt.str "%g") cells))
  | Error e -> Fmt.pr "%s" (Loop_check.kind e));
  Fmt.pr " | same row: %b@." (Stdlib.( = ) ssa loop)

let program f = build ~buffers:bufs f

let%expect_test "the projection runs a swap recurrence as a parallel transfer" =
  both ~cells:2
    (program (fun bld ->
         let a0 = B.f64 bld 1. and b0 = B.f64 bld 2. in
         let (B.Cons (a, B.Cons (b, B.Nil))) =
           B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 5)
             ~init:(B.Cons (a0, B.Cons (b0, B.Nil)))
             (fun _ _ (B.Cons (a, B.Cons (b, B.Nil))) ->
               B.Cons (b, B.Cons (a, B.Nil)))
         in
         store_at bld 1 (idx bld 0) a;
         store_at bld 1 (idx bld 1) b));
  [%expect {|
    ok 2 1 | same row: true
    |}]

let%expect_test "the projection keeps sums, zero-trip loops and rounding" =
  both
    (program (fun bld ->
         let sum =
           B.ordered_sum bld ~lo:(idx bld 0) ~hi:(idx bld 4)
             ~seed:(B.f64 bld 0.) (fun bld i ->
               B.mark bld Ssa_mark.Reduction;
               load_at bld 0 i)
         in
         store_at bld 1 (idx bld 0) sum;
         let empty =
           B.ordered_sum bld ~lo:(idx bld 3) ~hi:(idx bld 3)
             ~seed:(B.f64 bld 7.) (fun bld _ -> load_at bld 0 (idx bld 1000))
         in
         store_at bld 1 (idx bld 1) empty;
         let (B.Cons (z, B.Nil)) =
           B.for_ bld ~lo:(idx bld 2) ~hi:(idx bld 0)
             ~init:(B.Cons (B.f64 bld 5., B.Nil))
             (fun bld _ (B.Cons (x, B.Nil)) ->
               B.Cons
                 ( B.f64_binary bld Expr.Value.Add x
                     (load_at bld 0 (idx bld 1000)),
                   B.Nil ))
         in
         store_at bld 1 (idx bld 2) z;
         store_at bld 1 (idx bld 3) (B.f64 bld 0.1)));
  [%expect {|
    ok 1.25 7 5 0.1 | same row: true
    |}]

let%expect_test "the projection reproduces failures at the same site" =
  (* a load outside, a checked add past the domain and a checked scale *)
  both ~cells:1
    (program (fun bld -> store_at bld 1 (idx bld 0) (load_at bld 0 (idx bld 9))));
  both ~cells:1
    (program (fun bld ->
         let over = B.index_add bld (B.index bld 0x7FFF_FFFFL) (idx bld 1) in
         store_at bld 1 over (B.f64 bld 1.)));
  both ~cells:1
    (program (fun bld ->
         let over = B.index_scale bld 2L (B.index bld 0x4000_0000L) in
         store_at bld 1 over (B.f64 bld 1.)));
  both ~cells:1
    (program (fun bld ->
         let (B.Cons (x, B.Nil)) =
           B.if_ bld (B.pred bld false)
             ~then_:(fun bld -> B.Cons (load_at bld 0 (idx bld 100), B.Nil))
             ~else_:(fun bld -> B.Cons (B.f64 bld 3., B.Nil))
         in
         store_at bld 1 (idx bld 0) x));
  [%expect
    {|
    coord_out_of_range | same row: true
    index_overflow | same row: true
    index_overflow | same row: true
    ok 3 | same row: true
    |}]

(* ---- flat accesses ----------------------------------------------------------- *)

let%expect_test
    "a flat program converts to SSA and projects back, with the same cells" =
  let loop = Loop_programs.reversed_flat in
  let data = [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let shape = Loop_programs.shape_hw in
  let bind id =
    if Tensor_id.equal id (tid 0) then
      Some
        (Loop_fixtures.f32_tensor shape (fun c ->
             data.((Vec6.offset shape c :> int))))
    else None
  in
  let cells result =
    match Err.payload result with
    | Ok m ->
        let t = Tensor_id.Map.find (tid 1) m in
        List.init 6 (fun i ->
            Tensor.read t
              (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:(i / 3) ~w:(i mod 3) ~c:0))
    | Error e -> Fmt.failwith "%a" Loop_interp.pp_error e
  in
  let reference = cells (Loop_interp.run loop ~bind) in
  (* through SSA *)
  let ssa_program =
    match Err.payload (Ssa_of_loop.convert loop) with
    | Ok p -> p
    | Error (`Unsupported u) -> Fmt.failwith "%a" Ssa_of_loop_unsupported.pp u
  in
  let out = Array.make 6 0. in
  let memory = memory [ (0, floats (Array.copy data)); (1, floats out) ] in
  run ssa_program ~memory;
  (* and back to Loop *)
  let projected =
    cells (Loop_interp.run (Loop_of_ssa.convert ssa_program) ~bind)
  in
  let show l = String.concat " " (List.map (Fmt.str "%g") l) in
  Fmt.pr "loop:      %s@.ssa:       %s@.projected: %s@." (show reference)
    (show (Array.to_list out))
    (show projected);
  [%expect
    {|
    loop:      12 10 8 6 4 2
    ssa:       12 10 8 6 4 2
    projected: 12 10 8 6 4 2 |}]
