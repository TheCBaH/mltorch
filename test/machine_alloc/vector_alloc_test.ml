open Loop_ir_test
open Ssa_bridge_test.Ssa_fixtures

(* M11.3: planned vector kernels reference-allocated and realized on AArch64:
   register-wide values in Q registers and 16-byte slots, against the selected
   run and the plan's oracle. *)

module Src = Machine_source_test.Mir_source

let case kernel ~bind =
  Result.get_ok
    (Src.case_of_planned ~target:Ssa_ir.Ssa_target.neon128
       ~numerics:Ssa_ir.Ssa_numerics.Simd_fp32_ordered
       (Fusion_plan.default kernel)
       ~bind)

let data n = Array.init n (fun i -> (float_of_int (i * 7 mod 11) -. 5.) /. 4.)

let kernels () =
  let x = Loop_fixtures.load_t0 in
  let shape = Loop_fixtures.shape_w 37 in
  [
    ( "x * x + x, w=37",
      case
        (Loop_fixtures.pixel_kernel ~shape Expr.Value.(add (mul x x) x))
        ~bind:(bind_data ~shape (data 37)) );
  ]
  @ List.map
      (fun (m, k, n) ->
        ( Fmt.str "matmul %dx%dx%d" m k n,
          case (matmul_kernel ~m ~k ~n)
            ~bind:
              (matmul_bind ~m ~k ~n
                 ~a:(operand 3 (m * k))
                 ~b:(operand 5 (k * n))) ))
      [ (2, 3, 16); (3, 5, 17) ]

let%expect_test "reference allocation and frames of vector kernels" =
  List.iter
    (fun (name, c) ->
      Fmt.pr "%s: allocated %s; realized %s@." name (Alloc_harness.report c)
        (Alloc_harness.realized_report c))
    (kernels ());
  [%expect
    {|
    x * x + x, w=37: allocated ok; realized ok
    matmul 2x3x16: allocated ok; realized ok
    matmul 3x5x17: allocated ok; realized ok |}]
