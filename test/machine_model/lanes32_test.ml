module M = Machine_model.Mir_model
module T = Model_test

(* Thirty-two binary32 lanes a vector iteration on every Machine IR route,
   against the binary32 oracle: NEON's eight registers on generic, aarch64
   and x86_64, Zve32f's on generic and RV32. *)

let planned target =
  Ssa_backends.Pipeline.Planned
    { numerics = Ssa_ir.Ssa_numerics.Fp32_ordered; target }

let%expect_test "32-lane plans are the binary32 oracle" =
  List.iter
    (fun (tname, target, routes) ->
      List.iter
        (fun (name, g) ->
          List.iter
            (fun (rname, route) ->
              T.check ?route ~pipeline:(planned target) ~oracle:T.Fp32
                ~constant:T.positive ~calls:1
                (Fmt.str "%s, %s, %s" name tname rname)
                g)
            routes)
        (T.vector_fixtures @ T.transformer_fixtures
       @ T.transformer_bf16_fixtures @ Transformer_fixtures.bf16_out))
    [
      ( "neon128_x32",
        Ssa_ir.Ssa_target.neon128_x32,
        [
          ("generic", None);
          ("aarch64", Some (M.Route.Aarch64 M.Stage.Scanned));
          ("x86_64", Some (M.Route.X86_64 M.Stage.Scanned));
        ] );
      ( "zve32f128_x32",
        Ssa_ir.Ssa_target.zve32f128_x32,
        [
          ("generic", None); ("riscv32", Some (M.Route.Riscv32 M.Stage.Scanned));
        ] );
    ];
  [%expect
    {|
    relu then add over C=32, neon128_x32, generic (2 invocations): bitwise
    relu then add over C=32, neon128_x32, aarch64 (2 invocations): bitwise
    relu then add over C=32, neon128_x32, x86_64 (2 invocations): bitwise
    softmax over H, W=32, neon128_x32, generic (1 invocations): bitwise
    softmax over H, W=32, neon128_x32, aarch64 (1 invocations): bitwise
    softmax over H, W=32, neon128_x32, x86_64 (1 invocations): bitwise
    layer_norm over H, W=32, neon128_x32, generic (1 invocations): bitwise
    layer_norm over H, W=32, neon128_x32, aarch64 (1 invocations): bitwise
    layer_norm over H, W=32, neon128_x32, x86_64 (1 invocations): bitwise
    conv, batch norm, relu to C=32, neon128_x32, generic (3 invocations): bitwise
    conv, batch norm, relu to C=32, neon128_x32, aarch64 (3 invocations): bitwise
    conv, batch norm, relu to C=32, neon128_x32, x86_64 (3 invocations): bitwise
    residual add 2x160, neon128_x32, generic (1 invocations): bitwise
    residual add 2x160, neon128_x32, aarch64 (1 invocations): bitwise
    residual add 2x160, neon128_x32, x86_64 (1 invocations): bitwise
    residual add 1x67, neon128_x32, generic (1 invocations): bitwise
    residual add 1x67, neon128_x32, aarch64 (1 invocations): bitwise
    residual add 1x67, neon128_x32, x86_64 (1 invocations): bitwise
    residual add 1x33, neon128_x32, generic (1 invocations): bitwise
    residual add 1x33, neon128_x32, aarch64 (1 invocations): bitwise
    residual add 1x33, neon128_x32, x86_64 (1 invocations): bitwise
    rms norm 2x160, neon128_x32, generic (1 invocations): bitwise
    rms norm 2x160, neon128_x32, aarch64 (1 invocations): bitwise
    rms norm 2x160, neon128_x32, x86_64 (1 invocations): bitwise
    rms norm 1x67, neon128_x32, generic (1 invocations): bitwise
    rms norm 1x67, neon128_x32, aarch64 (1 invocations): bitwise
    rms norm 1x67, neon128_x32, x86_64 (1 invocations): bitwise
    rms norm 1x33, neon128_x32, generic (1 invocations): bitwise
    rms norm 1x33, neon128_x32, aarch64 (1 invocations): bitwise
    rms norm 1x33, neon128_x32, x86_64 (1 invocations): bitwise
    tanh gelu mul 2x160, neon128_x32, generic (2 invocations): bitwise
    tanh gelu mul 2x160, neon128_x32, aarch64 (2 invocations): bitwise
    tanh gelu mul 2x160, neon128_x32, x86_64 (2 invocations): bitwise
    tanh gelu mul 1x67, neon128_x32, generic (2 invocations): bitwise
    tanh gelu mul 1x67, neon128_x32, aarch64 (2 invocations): bitwise
    tanh gelu mul 1x67, neon128_x32, x86_64 (2 invocations): bitwise
    tanh gelu mul 1x33, neon128_x32, generic (2 invocations): bitwise
    tanh gelu mul 1x33, neon128_x32, aarch64 (2 invocations): bitwise
    tanh gelu mul 1x33, neon128_x32, x86_64 (2 invocations): bitwise
    residual add bf16 2x160, neon128_x32, generic (1 invocations): bitwise
    residual add bf16 2x160, neon128_x32, aarch64 (1 invocations): bitwise
    residual add bf16 2x160, neon128_x32, x86_64 (1 invocations): bitwise
    residual add bf16 1x67, neon128_x32, generic (1 invocations): bitwise
    residual add bf16 1x67, neon128_x32, aarch64 (1 invocations): bitwise
    residual add bf16 1x67, neon128_x32, x86_64 (1 invocations): bitwise
    residual add bf16 1x33, neon128_x32, generic (1 invocations): bitwise
    residual add bf16 1x33, neon128_x32, aarch64 (1 invocations): bitwise
    residual add bf16 1x33, neon128_x32, x86_64 (1 invocations): bitwise
    rms norm bf16 2x160, neon128_x32, generic (1 invocations): bitwise
    rms norm bf16 2x160, neon128_x32, aarch64 (1 invocations): bitwise
    rms norm bf16 2x160, neon128_x32, x86_64 (1 invocations): bitwise
    rms norm bf16 1x67, neon128_x32, generic (1 invocations): bitwise
    rms norm bf16 1x67, neon128_x32, aarch64 (1 invocations): bitwise
    rms norm bf16 1x67, neon128_x32, x86_64 (1 invocations): bitwise
    rms norm bf16 1x33, neon128_x32, generic (1 invocations): bitwise
    rms norm bf16 1x33, neon128_x32, aarch64 (1 invocations): bitwise
    rms norm bf16 1x33, neon128_x32, x86_64 (1 invocations): bitwise
    tanh gelu mul bf16 2x160, neon128_x32, generic (2 invocations): bitwise
    tanh gelu mul bf16 2x160, neon128_x32, aarch64 (2 invocations): bitwise
    tanh gelu mul bf16 2x160, neon128_x32, x86_64 (2 invocations): bitwise
    tanh gelu mul bf16 1x67, neon128_x32, generic (2 invocations): bitwise
    tanh gelu mul bf16 1x67, neon128_x32, aarch64 (2 invocations): bitwise
    tanh gelu mul bf16 1x67, neon128_x32, x86_64 (2 invocations): bitwise
    tanh gelu mul bf16 1x33, neon128_x32, generic (2 invocations): bitwise
    tanh gelu mul bf16 1x33, neon128_x32, aarch64 (2 invocations): bitwise
    tanh gelu mul bf16 1x33, neon128_x32, x86_64 (2 invocations): bitwise
    residual add bf16->bf16 2x160, neon128_x32, generic (2 invocations): bitwise
    residual add bf16->bf16 2x160, neon128_x32, aarch64 (2 invocations): bitwise
    residual add bf16->bf16 2x160, neon128_x32, x86_64 (2 invocations): bitwise
    residual add bf16->bf16 1x67, neon128_x32, generic (2 invocations): bitwise
    residual add bf16->bf16 1x67, neon128_x32, aarch64 (2 invocations): bitwise
    residual add bf16->bf16 1x67, neon128_x32, x86_64 (2 invocations): bitwise
    residual add bf16->bf16 1x33, neon128_x32, generic (2 invocations): bitwise
    residual add bf16->bf16 1x33, neon128_x32, aarch64 (2 invocations): bitwise
    residual add bf16->bf16 1x33, neon128_x32, x86_64 (2 invocations): bitwise
    rms norm bf16->bf16 2x160, neon128_x32, generic (2 invocations): bitwise
    rms norm bf16->bf16 2x160, neon128_x32, aarch64 (2 invocations): bitwise
    rms norm bf16->bf16 2x160, neon128_x32, x86_64 (2 invocations): bitwise
    rms norm bf16->bf16 1x67, neon128_x32, generic (2 invocations): bitwise
    rms norm bf16->bf16 1x67, neon128_x32, aarch64 (2 invocations): bitwise
    rms norm bf16->bf16 1x67, neon128_x32, x86_64 (2 invocations): bitwise
    rms norm bf16->bf16 1x33, neon128_x32, generic (2 invocations): bitwise
    rms norm bf16->bf16 1x33, neon128_x32, aarch64 (2 invocations): bitwise
    rms norm bf16->bf16 1x33, neon128_x32, x86_64 (2 invocations): bitwise
    tanh gelu mul bf16->bf16 2x160, neon128_x32, generic (3 invocations): bitwise
    tanh gelu mul bf16->bf16 2x160, neon128_x32, aarch64 (3 invocations): bitwise
    tanh gelu mul bf16->bf16 2x160, neon128_x32, x86_64 (3 invocations): bitwise
    tanh gelu mul bf16->bf16 1x67, neon128_x32, generic (3 invocations): bitwise
    tanh gelu mul bf16->bf16 1x67, neon128_x32, aarch64 (3 invocations): bitwise
    tanh gelu mul bf16->bf16 1x67, neon128_x32, x86_64 (3 invocations): bitwise
    tanh gelu mul bf16->bf16 1x33, neon128_x32, generic (3 invocations): bitwise
    tanh gelu mul bf16->bf16 1x33, neon128_x32, aarch64 (3 invocations): bitwise
    tanh gelu mul bf16->bf16 1x33, neon128_x32, x86_64 (3 invocations): bitwise
    relu then add over C=32, zve32f128_x32, generic (2 invocations): bitwise
    relu then add over C=32, zve32f128_x32, riscv32 (2 invocations): bitwise
    softmax over H, W=32, zve32f128_x32, generic (1 invocations): bitwise
    softmax over H, W=32, zve32f128_x32, riscv32 (1 invocations): bitwise
    layer_norm over H, W=32, zve32f128_x32, generic (1 invocations): bitwise
    layer_norm over H, W=32, zve32f128_x32, riscv32 (1 invocations): bitwise
    conv, batch norm, relu to C=32, zve32f128_x32, generic (3 invocations): bitwise
    conv, batch norm, relu to C=32, zve32f128_x32, riscv32 (3 invocations): bitwise
    residual add 2x160, zve32f128_x32, generic (1 invocations): bitwise
    residual add 2x160, zve32f128_x32, riscv32 (1 invocations): bitwise
    residual add 1x67, zve32f128_x32, generic (1 invocations): bitwise
    residual add 1x67, zve32f128_x32, riscv32 (1 invocations): bitwise
    residual add 1x33, zve32f128_x32, generic (1 invocations): bitwise
    residual add 1x33, zve32f128_x32, riscv32 (1 invocations): bitwise
    rms norm 2x160, zve32f128_x32, generic (1 invocations): bitwise
    rms norm 2x160, zve32f128_x32, riscv32 (1 invocations): bitwise
    rms norm 1x67, zve32f128_x32, generic (1 invocations): bitwise
    rms norm 1x67, zve32f128_x32, riscv32 (1 invocations): bitwise
    rms norm 1x33, zve32f128_x32, generic (1 invocations): bitwise
    rms norm 1x33, zve32f128_x32, riscv32 (1 invocations): bitwise
    tanh gelu mul 2x160, zve32f128_x32, generic (2 invocations): bitwise
    tanh gelu mul 2x160, zve32f128_x32, riscv32 (2 invocations): bitwise
    tanh gelu mul 1x67, zve32f128_x32, generic (2 invocations): bitwise
    tanh gelu mul 1x67, zve32f128_x32, riscv32 (2 invocations): bitwise
    tanh gelu mul 1x33, zve32f128_x32, generic (2 invocations): bitwise
    tanh gelu mul 1x33, zve32f128_x32, riscv32 (2 invocations): bitwise
    residual add bf16 2x160, zve32f128_x32, generic (1 invocations): bitwise
    residual add bf16 2x160, zve32f128_x32, riscv32 (1 invocations): bitwise
    residual add bf16 1x67, zve32f128_x32, generic (1 invocations): bitwise
    residual add bf16 1x67, zve32f128_x32, riscv32 (1 invocations): bitwise
    residual add bf16 1x33, zve32f128_x32, generic (1 invocations): bitwise
    residual add bf16 1x33, zve32f128_x32, riscv32 (1 invocations): bitwise
    rms norm bf16 2x160, zve32f128_x32, generic (1 invocations): bitwise
    rms norm bf16 2x160, zve32f128_x32, riscv32 (1 invocations): bitwise
    rms norm bf16 1x67, zve32f128_x32, generic (1 invocations): bitwise
    rms norm bf16 1x67, zve32f128_x32, riscv32 (1 invocations): bitwise
    rms norm bf16 1x33, zve32f128_x32, generic (1 invocations): bitwise
    rms norm bf16 1x33, zve32f128_x32, riscv32 (1 invocations): bitwise
    tanh gelu mul bf16 2x160, zve32f128_x32, generic (2 invocations): bitwise
    tanh gelu mul bf16 2x160, zve32f128_x32, riscv32 (2 invocations): bitwise
    tanh gelu mul bf16 1x67, zve32f128_x32, generic (2 invocations): bitwise
    tanh gelu mul bf16 1x67, zve32f128_x32, riscv32 (2 invocations): bitwise
    tanh gelu mul bf16 1x33, zve32f128_x32, generic (2 invocations): bitwise
    tanh gelu mul bf16 1x33, zve32f128_x32, riscv32 (2 invocations): bitwise
    residual add bf16->bf16 2x160, zve32f128_x32, generic (2 invocations): bitwise
    residual add bf16->bf16 2x160, zve32f128_x32, riscv32 (2 invocations): bitwise
    residual add bf16->bf16 1x67, zve32f128_x32, generic (2 invocations): bitwise
    residual add bf16->bf16 1x67, zve32f128_x32, riscv32 (2 invocations): bitwise
    residual add bf16->bf16 1x33, zve32f128_x32, generic (2 invocations): bitwise
    residual add bf16->bf16 1x33, zve32f128_x32, riscv32 (2 invocations): bitwise
    rms norm bf16->bf16 2x160, zve32f128_x32, generic (2 invocations): bitwise
    rms norm bf16->bf16 2x160, zve32f128_x32, riscv32 (2 invocations): bitwise
    rms norm bf16->bf16 1x67, zve32f128_x32, generic (2 invocations): bitwise
    rms norm bf16->bf16 1x67, zve32f128_x32, riscv32 (2 invocations): bitwise
    rms norm bf16->bf16 1x33, zve32f128_x32, generic (2 invocations): bitwise
    rms norm bf16->bf16 1x33, zve32f128_x32, riscv32 (2 invocations): bitwise
    tanh gelu mul bf16->bf16 2x160, zve32f128_x32, generic (3 invocations): bitwise
    tanh gelu mul bf16->bf16 2x160, zve32f128_x32, riscv32 (3 invocations): bitwise
    tanh gelu mul bf16->bf16 1x67, zve32f128_x32, generic (3 invocations): bitwise
    tanh gelu mul bf16->bf16 1x67, zve32f128_x32, riscv32 (3 invocations): bitwise
    tanh gelu mul bf16->bf16 1x33, zve32f128_x32, generic (3 invocations): bitwise
    tanh gelu mul bf16->bf16 1x33, zve32f128_x32, riscv32 (3 invocations): bitwise |}]
