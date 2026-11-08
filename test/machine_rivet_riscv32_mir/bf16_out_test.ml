(* bfloat16 results on RV32 under qemu, each call bitwise against the binary32
   oracle: scalar (ILP32, the 16-bit store of the rounded high half), then
   vectors under register groups at VLEN 128 and 1024 (the int32 rounding,
   [vnsrl.wi] into half a group, [vse16.v]). In a file of its own for the
   inline-test time limit. *)

module T = Machine_model_test.Model_test
module R = Machine_rivet_riscv32_mir.Rv32_mir_route

let planned target =
  Ssa_backends.Pipeline.Planned
    { numerics = Ssa_ir.Ssa_numerics.Fp32_ordered; target }

let cpu vlen =
  Fmt.str
    "rv32,d=false,zve32f=true,vlen=%d,elen=32,rvv_ta_all_1s=true,rvv_ma_all_1s=true"
    vlen

let%expect_test "bfloat16 results under qemu, scalar" =
  List.iter
    (fun (name, g) ->
      T.check ~route:(R.route ())
        ~pipeline:(planned Ssa_ir.Ssa_target.scalar)
        ~oracle:T.Fp32 ~constant:T.positive name g)
    Machine_model_test.Transformer_fixtures.bf16_out;
  [%expect
    {|
    residual add bf16->bf16 2x160 (2 invocations): bitwise, bitwise
    residual add bf16->bf16 1x67 (2 invocations): bitwise, bitwise
    residual add bf16->bf16 1x33 (2 invocations): bitwise, bitwise
    rms norm bf16->bf16 2x160 (2 invocations): bitwise, bitwise
    rms norm bf16->bf16 1x67 (2 invocations): bitwise, bitwise
    rms norm bf16->bf16 1x33 (2 invocations): bitwise, bitwise
    tanh gelu mul bf16->bf16 2x160 (3 invocations): bitwise, bitwise
    tanh gelu mul bf16->bf16 1x67 (3 invocations): bitwise, bitwise
    tanh gelu mul bf16->bf16 1x33 (3 invocations): bitwise, bitwise |}]

let%expect_test "bfloat16 results under register groups" =
  List.iter
    (fun vlen ->
      List.iter
        (fun (name, g) ->
          T.check
            ~route:(R.route ~cpu:(cpu vlen) ~registers:R.Registers.Grouped ())
            ~pipeline:(planned Ssa_ir.Ssa_target.zve32f128)
            ~oracle:T.Fp32 ~constant:T.positive ~calls:1
            (Fmt.str "vlen %d, %s" vlen name)
            g)
        Machine_model_test.Transformer_fixtures.bf16_out)
    [ 128; 1024 ];
  [%expect
    {|
    vlen 128, residual add bf16->bf16 2x160 (2 invocations): bitwise
    vlen 128, residual add bf16->bf16 1x67 (2 invocations): bitwise
    vlen 128, residual add bf16->bf16 1x33 (2 invocations): bitwise
    vlen 128, rms norm bf16->bf16 2x160 (2 invocations): bitwise
    vlen 128, rms norm bf16->bf16 1x67 (2 invocations): bitwise
    vlen 128, rms norm bf16->bf16 1x33 (2 invocations): bitwise
    vlen 128, tanh gelu mul bf16->bf16 2x160 (3 invocations): bitwise
    vlen 128, tanh gelu mul bf16->bf16 1x67 (3 invocations): bitwise
    vlen 128, tanh gelu mul bf16->bf16 1x33 (3 invocations): bitwise
    vlen 1024, residual add bf16->bf16 2x160 (2 invocations): bitwise
    vlen 1024, residual add bf16->bf16 1x67 (2 invocations): bitwise
    vlen 1024, residual add bf16->bf16 1x33 (2 invocations): bitwise
    vlen 1024, rms norm bf16->bf16 2x160 (2 invocations): bitwise
    vlen 1024, rms norm bf16->bf16 1x67 (2 invocations): bitwise
    vlen 1024, rms norm bf16->bf16 1x33 (2 invocations): bitwise
    vlen 1024, tanh gelu mul bf16->bf16 2x160 (3 invocations): bitwise
    vlen 1024, tanh gelu mul bf16->bf16 1x67 (3 invocations): bitwise
    vlen 1024, tanh gelu mul bf16->bf16 1x33 (3 invocations): bitwise |}]
