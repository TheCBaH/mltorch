(* Source kernels as RV32 processes: each case is selected, linear-scan
   allocated, framed, published and run under qemu over the case's bound bytes;
   its status, failure row and outputs are compared with the generic route and
   the SSA oracle (whose event counts a process does not report). *)

open Loop_ir_test
open Ssa_bridge_test.Ssa_fixtures
open Machine_ir
module Src = Machine_source_test.Mir_source
module Cases = Machine_source_test.Mir_source_test
module A64T = Machine_aarch64_test.A64_select_test
module R = Machine_rivet_riscv32_mir.Rv32_mir_route
module Art = Machine_model.Mir_artifact

let ( let* ) = Result.bind

let qemu ?mutation ?cpu (case : Src.Case.t) =
  let g = case.Src.Case.lowered.Machine_lower.Mir_lower.program in
  let* published = R.publish ~allocation:R.Allocation.Scanned ~sites:[||] g in
  let* prepared = R.prepare ?mutation published in
  let phys = Art.program published.R.artifact in
  let memory = Machine_interp.Mir_memory.create () in
  let* binding =
    Machine_interp.Mir_interp.instantiate (R.regions_program phys) memory
      ~bound:case.Src.Case.bound
  in
  let* status = R.run_once ?cpu prepared ~published memory binding in
  Ok
    (Src.selected_observation
       ~layout:case.Src.Case.lowered.Machine_lower.Mir_lower.layout ~sites:[||]
       ~record:published.R.record memory binding
       (Machine_interp.Mir_interp.Outcome.Success
          [
            Machine_interp.Mir_datum.Bits
              (Int64.logand (Int64.of_int32 status) 0xFFFF_FFFFL);
          ])
       [])

let no_events (r : Src.Route.t) =
  {
    r with
    Src.Route.observation =
      { r.Src.Route.observation with Mir_observation.events = [] };
  }

let report ?mutation ?cpu (case : Src.Case.t) =
  match qemu ?mutation ?cpu case with
  | Error e -> e
  | Ok obs -> (
      let x = { Src.Route.name = "qemu"; observation = obs } in
      let ds =
        List.filter_map Fun.id
          [
            Src.verdict ~expected:(no_events case.Src.Case.generic) ~actual:x;
            Src.verdict ~expected:(no_events case.Src.Case.oracle) ~actual:x;
          ]
      in
      Src.status_name obs
      ^ match ds with [] -> "" | d -> " DISAGREE " ^ String.concat "; " d)

let planned ?mutation kernel ~bind =
  match
    Src.case_of_planned ~target:Ssa_ir.Ssa_target.scalar
      ~numerics:Ssa_ir.Ssa_numerics.Fp32_ordered
      (Fusion_plan.default kernel)
      ~bind
  with
  | Error e -> e
  | Ok c -> report ?mutation c

let program p ~inputs =
  match Src.case_of_program p ~inputs () with Error e -> e | Ok c -> report c

let data_bind = A64T.data_bind
let zeros = data_bind [| 0.; 0.; 0.; 0. |]
let i64s xs = Ssa_ir.Ssa_memory.Int64s (Array.of_list xs)

let%expect_test "kernels and failures under qemu" =
  List.iter
    (fun (n, k, bind) -> Fmt.pr "%s: %s@." n (planned k ~bind))
    [
      ("pointwise", Loop_programs.kernel, data_bind [| -0.; 1.5; nan; 3. |]);
      ( "pointwise bounds",
        Loop_programs.kernel,
        data_bind [| 1e30; -1e30; 0.1; 16777217. |] );
      ( "noncommutative",
        Cases.noncommutative,
        data_bind [| 7.; -0.; 1e-40; 3.4e38 |] );
      ("shifted", Loop_programs.shifted_kernel, zeros);
      ("negative shift", A64T.shifted ~dh:0 ~dw:(-1), zeros);
      ("competing axes", A64T.shifted ~dh:1 ~dw:4, zeros);
      ("index overflow", Loop_programs.overflow_kernel, zeros);
      ( "matmul 5x7x3",
        matmul_kernel ~m:5 ~k:7 ~n:3,
        matmul_bind ~m:5 ~k:7 ~n:3 ~a:(operand 3 35) ~b:(operand 5 21) );
    ];
  [%expect
    {|
    pointwise: ok
    pointwise bounds: ok
    noncommutative: ok
    shifted: coord_out_of_range(t0, W)
    negative shift: coord_out_of_range(t0, W)
    competing axes: coord_out_of_range(t0, H)
    index overflow: index_overflow(mul)
    matmul 5x7x3: ok |}]

let%expect_test "i64 pairs under qemu" =
  let module B = Ssa_ir.Ssa_builder in
  let arith op = A64T.i64_program (fun bld x y -> B.i64_arith bld op x y) in
  let lt =
    A64T.i64_program (fun bld x y ->
        B.select bld (B.i64_compare bld Ssa_ir.Ssa_op.Compare.Lt x y) x y)
  in
  List.iter
    (fun (n, p) ->
      List.iter
        (fun (x, y) ->
          Fmt.pr "%s %Lx %Lx: %s@." n x y
            (program p ~inputs:[ (0, i64s [ x; y ]) ]))
        [
          (0xFFFF_FFFFL, 1L); (Int64.min_int, Int64.max_int); (-3L, 0x7FFF_FFFFL);
        ])
    Ssa_ir.Ssa_op.
      [
        ("add", arith I64_op.Add);
        ("sub", arith I64_op.Sub);
        ("mul", arith I64_op.Mul);
        ("min", lt);
      ];
  [%expect
    {|
    add ffffffff 1: ok
    add 8000000000000000 7fffffffffffffff: ok
    add fffffffffffffffd 7fffffff: ok
    sub ffffffff 1: ok
    sub 8000000000000000 7fffffffffffffff: ok
    sub fffffffffffffffd 7fffffff: ok
    mul ffffffff 1: ok
    mul 8000000000000000 7fffffffffffffff: ok
    mul fffffffffffffffd 7fffffff: ok
    min ffffffff 1: ok
    min 8000000000000000 7fffffffffffffff: ok
    min fffffffffffffffd 7fffffff: ok |}]

let vplanned ?mutation ?cpu kernel ~bind =
  match
    Src.case_of_planned ~target:Ssa_ir.Ssa_target.zve32f128
      ~numerics:Ssa_ir.Ssa_numerics.Fp32_ordered
      (Fusion_plan.default kernel)
      ~bind
  with
  | Error e -> e
  | Ok c -> report ?mutation ?cpu c

let vector_cases () =
  let x = Loop_fixtures.load_t0 in
  let data n =
    Array.init n (fun i -> (float_of_int (i * 7 mod 11) -. 5.) /. 4.)
  in
  List.concat_map
    (fun (name, body) ->
      List.map
        (fun n ->
          let d = data n in
          d.(0) <- Float.nan;
          d.(1) <- -0.;
          d.(2) <- 3.4e38;
          let shape = Loop_fixtures.shape_w n in
          ( Fmt.str "%s, w=%d" name n,
            Loop_fixtures.pixel_kernel ~shape body,
            bind_data ~shape d ))
        [ 16; 37 ])
    [
      ("x / 3 - 1.5", Expr.Value.(sub (div x (const 3.)) (const 1.5)));
      ("sqrt x", Expr.Value.sqrt x);
      ("trunc x", Expr.Value.trunc x);
      ( "x < 0 ? -x : x / 3",
        Expr.Value.(
          select
            (Expr.Bool.value_lt x (const 0.))
            (sub (const 0.) x)
            (div x (const 3.))) );
    ]
  @ List.map
      (fun (m, k, n) ->
        let a = operand 3 (m * k) and b = operand 5 (k * n) in
        ( Fmt.str "%dx%dx%d" m k n,
          matmul_kernel ~m ~k ~n,
          matmul_bind ~m ~k ~n ~a ~b ))
      [ (2, 3, 16); (3, 5, 17); (5, 7, 33) ]

let%expect_test "Zve32f kernels under qemu" =
  List.iter
    (fun (n, k, bind) -> Fmt.pr "%s: %s@." n (vplanned k ~bind))
    (vector_cases ());
  [%expect
    {|
    x / 3 - 1.5, w=16: ok
    x / 3 - 1.5, w=37: ok
    sqrt x, w=16: ok
    sqrt x, w=37: ok
    trunc x, w=16: ok
    trunc x, w=37: ok
    x < 0 ? -x : x / 3, w=16: ok
    x < 0 ? -x : x / 3, w=37: ok
    2x3x16: ok
    3x5x17: ok
    5x7x33: ok |}]

(* The same image at every VLEN from 128 up, qemu filling tail and masked-off
   elements with ones: the code depends on neither. Below 128 the harness
   refuses to run it (exit 105). *)
let%expect_test "Zve32f kernels at every VLEN" =
  List.iter
    (fun vlen ->
      let cpu =
        Fmt.str
          "rv32,d=false,zve32f=true,vlen=%d,elen=32,rvv_ta_all_1s=true,rvv_ma_all_1s=true"
          vlen
      in
      let results =
        List.map (fun (_, k, bind) -> vplanned ~cpu k ~bind) (vector_cases ())
      in
      let ok = List.length (List.filter (fun r -> r = "ok") results) in
      Fmt.pr "vlen %d: %d of %d ok%s@." vlen ok (List.length results)
        (match List.filter (fun r -> r <> "ok") results with
        | r :: _ -> "; first other: " ^ r
        | [] -> ""))
    [ 64; 128; 256; 512; 1024 ];
  [%expect
    {|
    vlen 64: 0 of 11 ok; first other: the process exited 105
    vlen 128: 11 of 11 ok
    vlen 256: 11 of 11 ok
    vlen 512: 11 of 11 ok
    vlen 1024: 11 of 11 ok |}]

let%expect_test "vector configuration and lane mutations are caught" =
  let find n =
    List.find (fun (m, _, _) -> String.equal m n) (vector_cases ())
  in
  List.iter
    (fun (case, mutation) ->
      let _, k, bind = find case in
      Fmt.pr "%s: %s@." case (vplanned ~mutation k ~bind))
    Machine_rivet_riscv32_mir.Rv32_mir_form.Mutation.
      [
        ("5x7x33", Unset_vtype);
        ("x < 0 ? -x : x / 3, w=37", Select_arms);
        ("x / 3 - 1.5, w=37", Commuted_sub);
      ];
  [%expect
    {|
    timeout: the monitored command dumped core
    5x7x33: the process was stopped by signal -5
    x < 0 ? -x : x / 3, w=37: ok DISAGREE generic vs qemu: output t1[1]: -0x0p+0:f32 vs 0x0p+0:f32; oracle vs qemu: output t1[1]: -0x0p+0:f32 vs 0x0p+0:f32
    x / 3 - 1.5, w=37: ok DISAGREE generic vs qemu: output t1[1]: -0x1.8p+0:f32 vs 0x1.8p+0:f32; oracle vs qemu: output t1[1]: -0x1.8p+0:f32 vs 0x1.8p+0:f32 |}]

(* Dequantization and binary16/bfloat16 as integer lanes under qemu: the
   integer register loads at their own width, the slides and extensions, the
   lane shifts, at VLEN 128 and 1024. *)
let%expect_test "integer-lane decodes under qemu" =
  let f32 x = fst (Quant.round_scale x) in
  let run name fmt ?quant n cells =
    let bind id =
      if Tensor_id.equal id (Loop_fixtures.tid 0) then
        Some (Loop_programs.raw_tensor fmt ?quant n cells)
      else None
    in
    List.iter
      (fun vlen ->
        let cpu =
          Fmt.str
            "rv32,d=false,zve32f=true,vlen=%d,elen=32,rvv_ta_all_1s=true,rvv_ma_all_1s=true"
            vlen
        in
        Fmt.pr "%s, vlen %d: %s@." name vlen
          (vplanned ~cpu (Loop_programs.format_kernel ~fmt ?quant n) ~bind))
      [ 128; 1024 ]
  in
  let q n = Array.init n (fun i -> Int64.of_int ((i * 37 mod 256) - 128)) in
  run "i8 per tensor, 37 lanes" (Payload.Fmt Payload.I8)
    ~quant:(Quant.per_tensor ~scale:(f32 0.05) ~zero_point:1.25)
    37 (q 37);
  run "i16 per tensor, 32 lanes" (Payload.Fmt Payload.I16)
    ~quant:(Quant.per_tensor ~scale:(f32 1e-3) ~zero_point:7.)
    32 (q 32);
  run "f16" (Payload.Fmt Payload.F16) 65536 (Array.init 65536 Int64.of_int);
  run "bf16" (Payload.Fmt Payload.BF16) 65536 (Array.init 65536 Int64.of_int);
  [%expect
    {|
    i8 per tensor, 37 lanes, vlen 128: ok
    i8 per tensor, 37 lanes, vlen 1024: ok
    i16 per tensor, 32 lanes, vlen 128: ok
    i16 per tensor, 32 lanes, vlen 1024: ok
    f16, vlen 128: ok
    f16, vlen 1024: ok
    bf16, vlen 128: ok
    bf16, vlen 1024: ok |}]
