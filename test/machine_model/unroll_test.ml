module M = Machine_model.Mir_model
module T = Model_test

(* Vector loops unrolled and interleaved ([Ssa_target.with_unroll]) on every
   Machine IR route, against the binary32 oracle: two and four copies, with
   the remainder trips of 1x67 and 1x33, on NEON's and Zve32f's lanes. *)

let planned ~unroll target =
  Ssa_backends.Pipeline.Planned
    {
      numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
      target = Ssa_ir.Ssa_target.with_unroll unroll target;
    }

let routes =
  [
    ("generic", None);
    ("aarch64 scanned", Some (M.Route.Aarch64 M.Stage.Scanned));
    ("x86_64 scanned", Some (M.Route.X86_64 M.Stage.Scanned));
    ("riscv32 scanned", Some (M.Route.Riscv32 M.Stage.Scanned));
  ]

let%expect_test "unrolled vector loops are the binary32 oracle" =
  List.iter
    (fun (tname, target) ->
      List.iter
        (fun unroll ->
          List.iter
            (fun (name, g) ->
              List.iter
                (fun (rname, route) ->
                  T.check ?route ~pipeline:(planned ~unroll target)
                    ~oracle:T.Fp32 ~constant:T.positive ~calls:1
                    (Fmt.str "%s, %s x%d, %s" name tname unroll rname)
                    g)
                (match target.Ssa_ir.Ssa_target.name with
                | "zve32f128" -> [ List.hd routes; List.nth routes 3 ]
                | _ -> List.filteri (fun i _ -> i < 3) routes))
            (T.vector_fixtures @ T.transformer_fixtures))
        [ 2; 4 ])
    [
      ("neon128", Ssa_ir.Ssa_target.neon128);
      ("zve32f128", Ssa_ir.Ssa_target.zve32f128);
    ];
  [%expect {|
    relu then add over C=32, neon128 x2, generic (2 invocations): bitwise
    relu then add over C=32, neon128 x2, aarch64 scanned (2 invocations): bitwise
    relu then add over C=32, neon128 x2, x86_64 scanned (2 invocations): bitwise
    softmax over H, W=32, neon128 x2, generic (1 invocations): bitwise
    softmax over H, W=32, neon128 x2, aarch64 scanned (1 invocations): bitwise
    softmax over H, W=32, neon128 x2, x86_64 scanned (1 invocations): bitwise
    layer_norm over H, W=32, neon128 x2, generic (1 invocations): bitwise
    layer_norm over H, W=32, neon128 x2, aarch64 scanned (1 invocations): bitwise
    layer_norm over H, W=32, neon128 x2, x86_64 scanned (1 invocations): bitwise
    conv, batch norm, relu to C=32, neon128 x2, generic (3 invocations): bitwise
    conv, batch norm, relu to C=32, neon128 x2, aarch64 scanned (3 invocations): bitwise
    conv, batch norm, relu to C=32, neon128 x2, x86_64 scanned (3 invocations): bitwise
    residual add 2x160, neon128 x2, generic (1 invocations): bitwise
    residual add 2x160, neon128 x2, aarch64 scanned (1 invocations): bitwise
    residual add 2x160, neon128 x2, x86_64 scanned (1 invocations): bitwise
    residual add 1x67, neon128 x2, generic (1 invocations): bitwise
    residual add 1x67, neon128 x2, aarch64 scanned (1 invocations): bitwise
    residual add 1x67, neon128 x2, x86_64 scanned (1 invocations): bitwise
    residual add 1x33, neon128 x2, generic (1 invocations): bitwise
    residual add 1x33, neon128 x2, aarch64 scanned (1 invocations): bitwise
    residual add 1x33, neon128 x2, x86_64 scanned (1 invocations): bitwise
    rms norm 2x160, neon128 x2, generic (1 invocations): bitwise
    rms norm 2x160, neon128 x2, aarch64 scanned (1 invocations): bitwise
    rms norm 2x160, neon128 x2, x86_64 scanned (1 invocations): bitwise
    rms norm 1x67, neon128 x2, generic (1 invocations): bitwise
    rms norm 1x67, neon128 x2, aarch64 scanned (1 invocations): bitwise
    rms norm 1x67, neon128 x2, x86_64 scanned (1 invocations): bitwise
    rms norm 1x33, neon128 x2, generic (1 invocations): bitwise
    rms norm 1x33, neon128 x2, aarch64 scanned (1 invocations): bitwise
    rms norm 1x33, neon128 x2, x86_64 scanned (1 invocations): bitwise
    tanh gelu mul 2x160, neon128 x2, generic (2 invocations): bitwise
    tanh gelu mul 2x160, neon128 x2, aarch64 scanned (2 invocations): bitwise
    tanh gelu mul 2x160, neon128 x2, x86_64 scanned (2 invocations): bitwise
    tanh gelu mul 1x67, neon128 x2, generic (2 invocations): bitwise
    tanh gelu mul 1x67, neon128 x2, aarch64 scanned (2 invocations): bitwise
    tanh gelu mul 1x67, neon128 x2, x86_64 scanned (2 invocations): bitwise
    tanh gelu mul 1x33, neon128 x2, generic (2 invocations): bitwise
    tanh gelu mul 1x33, neon128 x2, aarch64 scanned (2 invocations): bitwise
    tanh gelu mul 1x33, neon128 x2, x86_64 scanned (2 invocations): bitwise
    relu then add over C=32, neon128 x4, generic (2 invocations): bitwise
    relu then add over C=32, neon128 x4, aarch64 scanned (2 invocations): bitwise
    relu then add over C=32, neon128 x4, x86_64 scanned (2 invocations): bitwise
    softmax over H, W=32, neon128 x4, generic (1 invocations): bitwise
    softmax over H, W=32, neon128 x4, aarch64 scanned (1 invocations): bitwise
    softmax over H, W=32, neon128 x4, x86_64 scanned (1 invocations): bitwise
    layer_norm over H, W=32, neon128 x4, generic (1 invocations): bitwise
    layer_norm over H, W=32, neon128 x4, aarch64 scanned (1 invocations): bitwise
    layer_norm over H, W=32, neon128 x4, x86_64 scanned (1 invocations): bitwise
    conv, batch norm, relu to C=32, neon128 x4, generic (3 invocations): bitwise
    conv, batch norm, relu to C=32, neon128 x4, aarch64 scanned (3 invocations): bitwise
    conv, batch norm, relu to C=32, neon128 x4, x86_64 scanned (3 invocations): bitwise
    residual add 2x160, neon128 x4, generic (1 invocations): bitwise
    residual add 2x160, neon128 x4, aarch64 scanned (1 invocations): bitwise
    residual add 2x160, neon128 x4, x86_64 scanned (1 invocations): bitwise
    residual add 1x67, neon128 x4, generic (1 invocations): bitwise
    residual add 1x67, neon128 x4, aarch64 scanned (1 invocations): bitwise
    residual add 1x67, neon128 x4, x86_64 scanned (1 invocations): bitwise
    residual add 1x33, neon128 x4, generic (1 invocations): bitwise
    residual add 1x33, neon128 x4, aarch64 scanned (1 invocations): bitwise
    residual add 1x33, neon128 x4, x86_64 scanned (1 invocations): bitwise
    rms norm 2x160, neon128 x4, generic (1 invocations): bitwise
    rms norm 2x160, neon128 x4, aarch64 scanned (1 invocations): bitwise
    rms norm 2x160, neon128 x4, x86_64 scanned (1 invocations): bitwise
    rms norm 1x67, neon128 x4, generic (1 invocations): bitwise
    rms norm 1x67, neon128 x4, aarch64 scanned (1 invocations): bitwise
    rms norm 1x67, neon128 x4, x86_64 scanned (1 invocations): bitwise
    rms norm 1x33, neon128 x4, generic (1 invocations): bitwise
    rms norm 1x33, neon128 x4, aarch64 scanned (1 invocations): bitwise
    rms norm 1x33, neon128 x4, x86_64 scanned (1 invocations): bitwise
    tanh gelu mul 2x160, neon128 x4, generic (2 invocations): bitwise
    tanh gelu mul 2x160, neon128 x4, aarch64 scanned (2 invocations): bitwise
    tanh gelu mul 2x160, neon128 x4, x86_64 scanned (2 invocations): bitwise
    tanh gelu mul 1x67, neon128 x4, generic (2 invocations): bitwise
    tanh gelu mul 1x67, neon128 x4, aarch64 scanned (2 invocations): bitwise
    tanh gelu mul 1x67, neon128 x4, x86_64 scanned (2 invocations): bitwise
    tanh gelu mul 1x33, neon128 x4, generic (2 invocations): bitwise
    tanh gelu mul 1x33, neon128 x4, aarch64 scanned (2 invocations): bitwise
    tanh gelu mul 1x33, neon128 x4, x86_64 scanned (2 invocations): bitwise
    relu then add over C=32, zve32f128 x2, generic (2 invocations): bitwise
    relu then add over C=32, zve32f128 x2, riscv32 scanned (2 invocations): bitwise
    softmax over H, W=32, zve32f128 x2, generic (1 invocations): bitwise
    softmax over H, W=32, zve32f128 x2, riscv32 scanned (1 invocations): bitwise
    layer_norm over H, W=32, zve32f128 x2, generic (1 invocations): bitwise
    layer_norm over H, W=32, zve32f128 x2, riscv32 scanned (1 invocations): bitwise
    conv, batch norm, relu to C=32, zve32f128 x2, generic (3 invocations): bitwise
    conv, batch norm, relu to C=32, zve32f128 x2, riscv32 scanned (3 invocations): bitwise
    residual add 2x160, zve32f128 x2, generic (1 invocations): bitwise
    residual add 2x160, zve32f128 x2, riscv32 scanned (1 invocations): bitwise
    residual add 1x67, zve32f128 x2, generic (1 invocations): bitwise
    residual add 1x67, zve32f128 x2, riscv32 scanned (1 invocations): bitwise
    residual add 1x33, zve32f128 x2, generic (1 invocations): bitwise
    residual add 1x33, zve32f128 x2, riscv32 scanned (1 invocations): bitwise
    rms norm 2x160, zve32f128 x2, generic (1 invocations): bitwise
    rms norm 2x160, zve32f128 x2, riscv32 scanned (1 invocations): bitwise
    rms norm 1x67, zve32f128 x2, generic (1 invocations): bitwise
    rms norm 1x67, zve32f128 x2, riscv32 scanned (1 invocations): bitwise
    rms norm 1x33, zve32f128 x2, generic (1 invocations): bitwise
    rms norm 1x33, zve32f128 x2, riscv32 scanned (1 invocations): bitwise
    tanh gelu mul 2x160, zve32f128 x2, generic (2 invocations): bitwise
    tanh gelu mul 2x160, zve32f128 x2, riscv32 scanned (2 invocations): bitwise
    tanh gelu mul 1x67, zve32f128 x2, generic (2 invocations): bitwise
    tanh gelu mul 1x67, zve32f128 x2, riscv32 scanned (2 invocations): bitwise
    tanh gelu mul 1x33, zve32f128 x2, generic (2 invocations): bitwise
    tanh gelu mul 1x33, zve32f128 x2, riscv32 scanned (2 invocations): bitwise
    relu then add over C=32, zve32f128 x4, generic (2 invocations): bitwise
    relu then add over C=32, zve32f128 x4, riscv32 scanned (2 invocations): bitwise
    softmax over H, W=32, zve32f128 x4, generic (1 invocations): bitwise
    softmax over H, W=32, zve32f128 x4, riscv32 scanned (1 invocations): bitwise
    layer_norm over H, W=32, zve32f128 x4, generic (1 invocations): bitwise
    layer_norm over H, W=32, zve32f128 x4, riscv32 scanned (1 invocations): bitwise
    conv, batch norm, relu to C=32, zve32f128 x4, generic (3 invocations): bitwise
    conv, batch norm, relu to C=32, zve32f128 x4, riscv32 scanned (3 invocations): bitwise
    residual add 2x160, zve32f128 x4, generic (1 invocations): bitwise
    residual add 2x160, zve32f128 x4, riscv32 scanned (1 invocations): bitwise
    residual add 1x67, zve32f128 x4, generic (1 invocations): bitwise
    residual add 1x67, zve32f128 x4, riscv32 scanned (1 invocations): bitwise
    residual add 1x33, zve32f128 x4, generic (1 invocations): bitwise
    residual add 1x33, zve32f128 x4, riscv32 scanned (1 invocations): bitwise
    rms norm 2x160, zve32f128 x4, generic (1 invocations): bitwise
    rms norm 2x160, zve32f128 x4, riscv32 scanned (1 invocations): bitwise
    rms norm 1x67, zve32f128 x4, generic (1 invocations): bitwise
    rms norm 1x67, zve32f128 x4, riscv32 scanned (1 invocations): bitwise
    rms norm 1x33, zve32f128 x4, generic (1 invocations): bitwise
    rms norm 1x33, zve32f128 x4, riscv32 scanned (1 invocations): bitwise
    tanh gelu mul 2x160, zve32f128 x4, generic (2 invocations): bitwise
    tanh gelu mul 2x160, zve32f128 x4, riscv32 scanned (2 invocations): bitwise
    tanh gelu mul 1x67, zve32f128 x4, generic (2 invocations): bitwise
    tanh gelu mul 1x67, zve32f128 x4, riscv32 scanned (2 invocations): bitwise
    tanh gelu mul 1x33, zve32f128 x4, generic (2 invocations): bitwise
    tanh gelu mul 1x33, zve32f128 x4, riscv32 scanned (2 invocations): bitwise |}]
