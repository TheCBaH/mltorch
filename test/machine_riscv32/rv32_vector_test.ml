open Loop_ir_test
open Ssa_bridge_test.Ssa_fixtures

(* Planned binary32 vector kernels selected as Zve32f forms: selected,
   allocated and realized against the generic route and the plan's oracle. *)

module H = Rv32_harness

let show ?mutation ?(target = Ssa_ir.Ssa_target.zve32f128) ?(stage = H.Selected)
    kernel ~bind =
  Fmt.pr "%s@."
    (H.planned ?mutation ~stage ~target
       ~numerics:Ssa_ir.Ssa_numerics.Fp32_ordered kernel ~bind)

let data n = Array.init n (fun i -> (float_of_int (i * 7 mod 11) -. 5.) /. 4.)
let x = Loop_fixtures.load_t0

let cases () =
  List.concat_map
    (fun (name, body) ->
      List.map
        (fun n ->
          let d = data n in
          if n > 2 then (
            d.(0) <- Float.nan;
            d.(1) <- -0.;
            d.(2) <- 3.4e38);
          let shape = Loop_fixtures.shape_w n in
          ( Fmt.str "%s, w=%d" name n,
            Loop_fixtures.pixel_kernel ~shape body,
            bind_data ~shape d ))
        [ 16; 37 ])
    [
      ("x / 3 - 1.5", Expr.Value.(sub (div x (const 3.)) (const 1.5)));
      ("sqrt x", Expr.Value.sqrt x);
      ("x * x + x", Expr.Value.(add (mul x x) x));
      ("trunc x", Expr.Value.trunc x);
      ( "x < 0 ? -x : x / 3",
        Expr.Value.(
          select
            (Expr.Bool.value_lt x (const 0.))
            (sub (const 0.) x)
            (div x (const 3.))) );
      ( "x == 0 ? 0 : 1",
        Expr.Value.(
          select (Expr.Bool.value_eq x (const 0.)) (const 0.) (const 1.)) );
    ]
  @ List.map
      (fun (m, k, n) ->
        let a = operand 3 (m * k) and b = operand 5 (k * n) in
        ( Fmt.str "%dx%dx%d" m k n,
          matmul_kernel ~m ~k ~n,
          matmul_bind ~m ~k ~n ~a ~b ))
      [ (2, 3, 16); (3, 5, 17); (5, 7, 33) ]

let%expect_test "vector kernels, every stage" =
  List.iter
    (fun stage ->
      List.iter
        (fun (name, k, bind) ->
          Fmt.pr "%s: " name;
          show ~stage k ~bind)
        (cases ()))
    [ H.Selected; H.Allocated; H.Realized ];
  [%expect
    {|
    x / 3 - 1.5, w=16: ok
    x / 3 - 1.5, w=37: ok
    sqrt x, w=16: ok
    sqrt x, w=37: ok
    x * x + x, w=16: ok
    x * x + x, w=37: ok
    trunc x, w=16: ok
    trunc x, w=37: ok
    x < 0 ? -x : x / 3, w=16: ok
    x < 0 ? -x : x / 3, w=37: ok
    x == 0 ? 0 : 1, w=16: ok
    x == 0 ? 0 : 1, w=37: ok
    2x3x16: ok
    3x5x17: ok
    5x7x33: ok
    x / 3 - 1.5, w=16: ok
    x / 3 - 1.5, w=37: ok
    sqrt x, w=16: ok
    sqrt x, w=37: ok
    x * x + x, w=16: ok
    x * x + x, w=37: ok
    trunc x, w=16: ok
    trunc x, w=37: ok
    x < 0 ? -x : x / 3, w=16: ok
    x < 0 ? -x : x / 3, w=37: ok
    x == 0 ? 0 : 1, w=16: ok
    x == 0 ? 0 : 1, w=37: ok
    2x3x16: ok
    3x5x17: ok
    5x7x33: ok
    x / 3 - 1.5, w=16: ok
    x / 3 - 1.5, w=37: ok
    sqrt x, w=16: ok
    sqrt x, w=37: ok
    x * x + x, w=16: ok
    x * x + x, w=37: ok
    trunc x, w=16: ok
    trunc x, w=37: ok
    x < 0 ? -x : x / 3, w=16: ok
    x < 0 ? -x : x / 3, w=37: ok
    x == 0 ? 0 : 1, w=16: ok
    x == 0 ? 0 : 1, w=37: ok
    2x3x16: ok
    3x5x17: ok
    5x7x33: ok |}]

module Src = Machine_source_test.Mir_source
module Sel = Machine_target_riscv32.Rv32_select

(* How many of the selected forms are vector forms, and the program's
   features: the plan's vectors reach the target as Zve32f forms. *)
let vector_forms kernel ~bind =
  match
    Src.case_of_planned ~target:Ssa_ir.Ssa_target.zve32f128
      ~numerics:Ssa_ir.Ssa_numerics.Fp32_ordered
      (Fusion_plan.default kernel)
      ~bind
  with
  | Error e -> e
  | Ok case -> (
      match
        Err.payload
          (Sel.program case.Src.Case.lowered.Machine_lower.Mir_lower.program)
      with
      | Error r -> Fmt.str "%a" Sel.Refusal.pp r
      | Ok res ->
          let s =
            Machine_target_riscv32.Rv32_stage.Sel.Verified.selected
              res.Sel.selected
          in
          let n = ref 0 and all = ref 0 in
          List.iter
            (fun (f : (_, _) Machine_ir.Mir_func.t) ->
              List.iter
                (fun (b : (_, _) Machine_ir.Mir_block.t) ->
                  List.iter
                    (fun (i : _ Machine_ir.Mir_instr.t) ->
                      match i.Machine_ir.Mir_instr.op with
                      | Machine_ir.Mir_sel.Op.Machine op -> (
                          incr all;
                          match op with
                          | Machine_target_riscv32.Rv32_op.Vec _ -> incr n
                          | _ -> ())
                      | _ -> ())
                    b.Machine_ir.Mir_block.body)
                f.Machine_ir.Mir_func.blocks)
            s.Machine_target_riscv32.Rv32_stage.Sel.program
              .Machine_ir.Mir_program.funcs;
          Fmt.str "%d of %d forms vector; features [%s]" !n !all
            (String.concat ", "
               (List.map Machine_ir.Mir_target.Feature.name
                  s.Machine_target_riscv32.Rv32_stage.Sel.features)))

let%expect_test "the plans' vectors are Zve32f forms" =
  List.iter
    (fun (name, k, bind) -> Fmt.pr "%s: %s@." name (vector_forms k ~bind))
    (cases ());
  [%expect
    {|
    x / 3 - 1.5, w=16: 24 of 47 forms vector; features [zve32f]
    x / 3 - 1.5, w=37: 24 of 59 forms vector; features [zve32f]
    sqrt x, w=16: 12 of 31 forms vector; features [zve32f]
    sqrt x, w=37: 12 of 42 forms vector; features [zve32f]
    x * x + x, w=16: 24 of 43 forms vector; features [zve32f]
    x * x + x, w=37: 24 of 55 forms vector; features [zve32f]
    trunc x, w=16: 36 of 63 forms vector; features [zve32f]
    trunc x, w=37: 36 of 81 forms vector; features [zve32f]
    x < 0 ? -x : x / 3, w=16: 40 of 63 forms vector; features [zve32f]
    x < 0 ? -x : x / 3, w=37: 40 of 77 forms vector; features [zve32f]
    x == 0 ? 0 : 1, w=16: 24 of 47 forms vector; features [zve32f]
    x == 0 ? 0 : 1, w=37: 24 of 59 forms vector; features [zve32f]
    2x3x16: 38 of 96 forms vector; features [zve32f]
    3x5x17: 21 of 99 forms vector; features [zve32f]
    5x7x33: 21 of 99 forms vector; features [zve32f] |}]

let%expect_test "vector selection mutations are caught" =
  let find n = List.find (fun (m, _, _) -> String.equal m n) (cases ()) in
  List.iter
    (fun (case, mutation) ->
      let _, k, bind = find case in
      Fmt.pr "%s: " case;
      show ~mutation k ~bind)
    Sel.Mutation.[ ("trunc x, w=37", Unguarded_trunc); ("5x7x33", Contract) ];
  [%expect
    {|
    trunc x, w=37: ok DISAGREE generic vs riscv32: output t1[0]: nan:f32 vs 0x1p+31:f32; oracle vs riscv32: output t1[0]: nan:f32 vs 0x1p+31:f32
    5x7x33: ok DISAGREE generic vs riscv32: output t2[32]: -0x1.0046dcp+4:f32 vs -0x1.0046dep+4:f32; oracle vs riscv32: output t2[32]: -0x1.0046dcp+4:f32 vs -0x1.0046dep+4:f32 |}]

(* Dequantization and binary16/bfloat16 as integer lanes: the integer loads,
   extensions and lane shifts the plan's decodes make. *)
let%expect_test "integer-lane decodes" =
  let f32 x = fst (Quant.round_scale x) in
  let run name fmt ?quant n cells =
    let bind id =
      if Tensor_id.equal id (Loop_fixtures.tid 0) then
        Some (Loop_programs.raw_tensor fmt ?quant n cells)
      else None
    in
    List.iter
      (fun stage ->
        Fmt.pr "%s: %s@." name
          (H.planned ~stage ~target:Ssa_ir.Ssa_target.zve32f128
             ~numerics:Ssa_ir.Ssa_numerics.Fp32_ordered
             (Loop_programs.format_kernel ~fmt ?quant n)
             ~bind))
      [ H.Selected; H.Realized ]
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
    i8 per tensor, 37 lanes: ok
    i8 per tensor, 37 lanes: ok
    i16 per tensor, 32 lanes: ok
    i16 per tensor, 32 lanes: ok
    f16: ok
    f16: ok
    bf16: ok
    bf16: ok |}]
