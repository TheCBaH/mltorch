module M = Machine_model.Mir_model
module T = Machine_model_test.Model_test
module C = Machine_rivet_riscv32_coral.Rv32_mir_coral

let config configuration =
  {
    C.Config.dir = Sys.getenv "CORALNPU_DATA";
    configuration;
    deadline = 600.;
    flags = [];
  }

let planned target =
  Ssa_backends.Pipeline.Planned
    { numerics = Ssa_ir.Ssa_numerics.Fp32_ordered; target }

let%expect_test "binary32 bundles on MPACT" =
  List.iter
    (fun (cname, tname, target) ->
      List.iter
        (fun (name, g) ->
          T.check
            ~route:(C.route (config cname))
            ~pipeline:(planned target) ~oracle:T.Fp32 ~constant:T.positive
            ~calls:1
            (Fmt.str "%s, %s, %s" cname tname name)
            g)
        T.vector_fixtures)
    [
      ("mpact-v2", "scalar", Ssa_ir.Ssa_target.scalar);
      ("mpact-m3", "scalar", Ssa_ir.Ssa_target.scalar);
      ("mpact-m3", "zve32f128", Ssa_ir.Ssa_target.zve32f128);
      ("mpact-v2", "zve32f128", Ssa_ir.Ssa_target.zve32f128);
    ];
  [%expect
    {|
    mpact-v2, scalar, relu then add over C=32 (2 invocations): bitwise
    mpact-v2, scalar, softmax over H, W=32 (1 invocations): bitwise
    mpact-v2, scalar, layer_norm over H, W=32 (1 invocations): bitwise
    mpact-v2, scalar, conv, batch norm, relu to C=32 (3 invocations): bitwise
    mpact-m3, scalar, relu then add over C=32 (2 invocations): bitwise
    mpact-m3, scalar, softmax over H, W=32 (1 invocations): bitwise
    mpact-m3, scalar, layer_norm over H, W=32 (1 invocations): bitwise
    mpact-m3, scalar, conv, batch norm, relu to C=32 (3 invocations): bitwise
    mpact-m3, zve32f128, relu then add over C=32 (2 invocations): bitwise
    mpact-m3, zve32f128, softmax over H, W=32 (1 invocations): bitwise
    mpact-m3, zve32f128, layer_norm over H, W=32 (1 invocations): bitwise
    mpact-m3, zve32f128, conv, batch norm, relu to C=32 (3 invocations): bitwise
    mpact-v2, zve32f128, relu then add over C=32: refused: invocation 0 (n0): configuration mpact-v2 has no Zve32f, which the kernel uses; invocation 1 (n1): configuration mpact-v2 has no Zve32f, which the kernel uses
    mpact-v2, zve32f128, softmax over H, W=32 (1 invocations): bitwise
    mpact-v2, zve32f128, layer_norm over H, W=32 (1 invocations): bitwise
    mpact-v2, zve32f128, conv, batch norm, relu to C=32: refused: invocation 0 (n0): configuration mpact-v2 has no Zve32f, which the kernel uses; invocation 1 (n1): configuration mpact-v2 has no Zve32f, which the kernel uses; invocation 2 (n2): configuration mpact-v2 has no Zve32f, which the kernel uses |}]
