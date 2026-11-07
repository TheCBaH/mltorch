module M = Machine_model.Mir_model
module T = Machine_model_test.Model_test
module C = Machine_rivet_riscv32_coral.Rv32_mir_coral

let config configuration =
  {
    C.Config.dir = Sys.getenv "CORALNPU_DATA";
    configuration;
    deadline = Deadline.seconds;
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
        (T.vector_fixtures @ T.gemma_fixtures @ T.gemma_bf16_fixtures))
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
    mpact-v2, scalar, residual add 2x160 (1 invocations): bitwise
    mpact-v2, scalar, residual add 1x67 (1 invocations): bitwise
    mpact-v2, scalar, residual add 1x33 (1 invocations): bitwise
    mpact-v2, scalar, rms norm 2x160 (1 invocations): bitwise
    mpact-v2, scalar, rms norm 1x67 (1 invocations): bitwise
    mpact-v2, scalar, rms norm 1x33 (1 invocations): bitwise
    mpact-v2, scalar, tanh gelu mul 2x160 (2 invocations): bitwise
    mpact-v2, scalar, tanh gelu mul 1x67 (2 invocations): bitwise
    mpact-v2, scalar, tanh gelu mul 1x33 (2 invocations): bitwise
    mpact-v2, scalar, residual add bf16 2x160 (1 invocations): bitwise
    mpact-v2, scalar, residual add bf16 1x67 (1 invocations): bitwise
    mpact-v2, scalar, residual add bf16 1x33 (1 invocations): bitwise
    mpact-v2, scalar, rms norm bf16 2x160 (1 invocations): bitwise
    mpact-v2, scalar, rms norm bf16 1x67 (1 invocations): bitwise
    mpact-v2, scalar, rms norm bf16 1x33 (1 invocations): bitwise
    mpact-v2, scalar, tanh gelu mul bf16 2x160 (2 invocations): bitwise
    mpact-v2, scalar, tanh gelu mul bf16 1x67 (2 invocations): bitwise
    mpact-v2, scalar, tanh gelu mul bf16 1x33 (2 invocations): bitwise
    mpact-m3, scalar, relu then add over C=32 (2 invocations): bitwise
    mpact-m3, scalar, softmax over H, W=32 (1 invocations): bitwise
    mpact-m3, scalar, layer_norm over H, W=32 (1 invocations): bitwise
    mpact-m3, scalar, conv, batch norm, relu to C=32 (3 invocations): bitwise
    mpact-m3, scalar, residual add 2x160 (1 invocations): bitwise
    mpact-m3, scalar, residual add 1x67 (1 invocations): bitwise
    mpact-m3, scalar, residual add 1x33 (1 invocations): bitwise
    mpact-m3, scalar, rms norm 2x160 (1 invocations): bitwise
    mpact-m3, scalar, rms norm 1x67 (1 invocations): bitwise
    mpact-m3, scalar, rms norm 1x33 (1 invocations): bitwise
    mpact-m3, scalar, tanh gelu mul 2x160 (2 invocations): bitwise
    mpact-m3, scalar, tanh gelu mul 1x67 (2 invocations): bitwise
    mpact-m3, scalar, tanh gelu mul 1x33 (2 invocations): bitwise
    mpact-m3, scalar, residual add bf16 2x160 (1 invocations): bitwise
    mpact-m3, scalar, residual add bf16 1x67 (1 invocations): bitwise
    mpact-m3, scalar, residual add bf16 1x33 (1 invocations): bitwise
    mpact-m3, scalar, rms norm bf16 2x160 (1 invocations): bitwise
    mpact-m3, scalar, rms norm bf16 1x67 (1 invocations): bitwise
    mpact-m3, scalar, rms norm bf16 1x33 (1 invocations): bitwise
    mpact-m3, scalar, tanh gelu mul bf16 2x160 (2 invocations): bitwise
    mpact-m3, scalar, tanh gelu mul bf16 1x67 (2 invocations): bitwise
    mpact-m3, scalar, tanh gelu mul bf16 1x33 (2 invocations): bitwise
    mpact-m3, zve32f128, relu then add over C=32 (2 invocations): bitwise
    mpact-m3, zve32f128, softmax over H, W=32 (1 invocations): bitwise
    mpact-m3, zve32f128, layer_norm over H, W=32 (1 invocations): bitwise
    mpact-m3, zve32f128, conv, batch norm, relu to C=32 (3 invocations): bitwise
    mpact-m3, zve32f128, residual add 2x160 (1 invocations): bitwise
    mpact-m3, zve32f128, residual add 1x67 (1 invocations): bitwise
    mpact-m3, zve32f128, residual add 1x33 (1 invocations): bitwise
    mpact-m3, zve32f128, rms norm 2x160 (1 invocations): bitwise
    mpact-m3, zve32f128, rms norm 1x67 (1 invocations): bitwise
    mpact-m3, zve32f128, rms norm 1x33 (1 invocations): bitwise
    mpact-m3, zve32f128, tanh gelu mul 2x160 (2 invocations): bitwise
    mpact-m3, zve32f128, tanh gelu mul 1x67 (2 invocations): bitwise
    mpact-m3, zve32f128, tanh gelu mul 1x33 (2 invocations): bitwise
    mpact-m3, zve32f128, residual add bf16 2x160 (1 invocations): bitwise
    mpact-m3, zve32f128, residual add bf16 1x67 (1 invocations): bitwise
    mpact-m3, zve32f128, residual add bf16 1x33 (1 invocations): bitwise
    mpact-m3, zve32f128, rms norm bf16 2x160 (1 invocations): bitwise
    mpact-m3, zve32f128, rms norm bf16 1x67 (1 invocations): bitwise
    mpact-m3, zve32f128, rms norm bf16 1x33 (1 invocations): bitwise
    mpact-m3, zve32f128, tanh gelu mul bf16 2x160 (2 invocations): bitwise
    mpact-m3, zve32f128, tanh gelu mul bf16 1x67 (2 invocations): bitwise
    mpact-m3, zve32f128, tanh gelu mul bf16 1x33 (2 invocations): bitwise
    mpact-v2, zve32f128, relu then add over C=32: refused: invocation 0 (n0): configuration mpact-v2 has no Zve32f, which the kernel uses; invocation 1 (n1): configuration mpact-v2 has no Zve32f, which the kernel uses
    mpact-v2, zve32f128, softmax over H, W=32 (1 invocations): bitwise
    mpact-v2, zve32f128, layer_norm over H, W=32 (1 invocations): bitwise
    mpact-v2, zve32f128, conv, batch norm, relu to C=32: refused: invocation 0 (n0): configuration mpact-v2 has no Zve32f, which the kernel uses; invocation 1 (n1): configuration mpact-v2 has no Zve32f, which the kernel uses; invocation 2 (n2): configuration mpact-v2 has no Zve32f, which the kernel uses
    mpact-v2, zve32f128, residual add 2x160: refused: invocation 0 (n0): configuration mpact-v2 has no Zve32f, which the kernel uses
    mpact-v2, zve32f128, residual add 1x67: refused: invocation 0 (n0): configuration mpact-v2 has no Zve32f, which the kernel uses
    mpact-v2, zve32f128, residual add 1x33: refused: invocation 0 (n0): configuration mpact-v2 has no Zve32f, which the kernel uses
    mpact-v2, zve32f128, rms norm 2x160 (1 invocations): bitwise
    mpact-v2, zve32f128, rms norm 1x67 (1 invocations): bitwise
    mpact-v2, zve32f128, rms norm 1x33 (1 invocations): bitwise
    mpact-v2, zve32f128, tanh gelu mul 2x160: refused: invocation 0 (n0): configuration mpact-v2 has no Zve32f, which the kernel uses; invocation 1 (n1): configuration mpact-v2 has no Zve32f, which the kernel uses
    mpact-v2, zve32f128, tanh gelu mul 1x67: refused: invocation 0 (n0): configuration mpact-v2 has no Zve32f, which the kernel uses; invocation 1 (n1): configuration mpact-v2 has no Zve32f, which the kernel uses
    mpact-v2, zve32f128, tanh gelu mul 1x33: refused: invocation 0 (n0): configuration mpact-v2 has no Zve32f, which the kernel uses; invocation 1 (n1): configuration mpact-v2 has no Zve32f, which the kernel uses
    mpact-v2, zve32f128, residual add bf16 2x160: refused: invocation 0 (n0): configuration mpact-v2 has no Zve32f, which the kernel uses
    mpact-v2, zve32f128, residual add bf16 1x67: refused: invocation 0 (n0): configuration mpact-v2 has no Zve32f, which the kernel uses
    mpact-v2, zve32f128, residual add bf16 1x33: refused: invocation 0 (n0): configuration mpact-v2 has no Zve32f, which the kernel uses
    mpact-v2, zve32f128, rms norm bf16 2x160 (1 invocations): bitwise
    mpact-v2, zve32f128, rms norm bf16 1x67 (1 invocations): bitwise
    mpact-v2, zve32f128, rms norm bf16 1x33 (1 invocations): bitwise
    mpact-v2, zve32f128, tanh gelu mul bf16 2x160: refused: invocation 0 (n0): configuration mpact-v2 has no Zve32f, which the kernel uses; invocation 1 (n1): configuration mpact-v2 has no Zve32f, which the kernel uses
    mpact-v2, zve32f128, tanh gelu mul bf16 1x67: refused: invocation 0 (n0): configuration mpact-v2 has no Zve32f, which the kernel uses; invocation 1 (n1): configuration mpact-v2 has no Zve32f, which the kernel uses
    mpact-v2, zve32f128, tanh gelu mul bf16 1x33: refused: invocation 0 (n0): configuration mpact-v2 has no Zve32f, which the kernel uses; invocation 1 (n1): configuration mpact-v2 has no Zve32f, which the kernel uses |}]

(* The same Zve32f plans with sixteen-lane vectors in groups of four
   registers (LMUL 4) on MPACT. *)
let%expect_test "register groups on MPACT" =
  List.iter
    (fun (name, g) ->
      T.check
        ~route:
          (C.route
             ~registers:
               Machine_rivet_riscv32_mir.Rv32_mir_route.Registers.Grouped
             (config "mpact-m3"))
        ~pipeline:(planned Ssa_ir.Ssa_target.zve32f128)
        ~oracle:T.Fp32 ~constant:T.positive ~calls:1
        (Fmt.str "mpact-m3, grouped, %s" name)
        g)
    (T.vector_fixtures @ T.contraction_fixtures);
  [%expect
    {|
    mpact-m3, grouped, relu then add over C=32 (2 invocations): bitwise
    mpact-m3, grouped, softmax over H, W=32 (1 invocations): bitwise
    mpact-m3, grouped, layer_norm over H, W=32 (1 invocations): bitwise
    mpact-m3, grouped, conv, batch norm, relu to C=32 (3 invocations): bitwise
    mpact-m3, grouped, linear 9x32 k64 (1 invocations): bitwise
    mpact-m3, grouped, linear 9x32 k64 bias (1 invocations): bitwise
    mpact-m3, grouped, linear 7x29 k63 bias (1 invocations): bitwise
    mpact-m3, grouped, bmm 1x9x32 k64 (1 invocations): bitwise
    mpact-m3, grouped, bmm 3x7x29 k63 (1 invocations): bitwise |}]

(* The DMA engine is the subsystem's: a core without one refuses [~dma:Allowed]
   before running, and [Forbidden] is the in-place route it always was. *)
let%expect_test "the DMA policy on a core with no DMA" =
  List.iter
    (fun (label, dma) ->
      T.check
        ~route:(C.route ~dma (config "mpact-m3"))
        ~pipeline:(planned Ssa_ir.Ssa_target.zve32f128)
        ~oracle:T.Fp32 ~constant:T.positive ~calls:1 label
        (List.assoc "relu then add over C=32" T.vector_fixtures))
    [ ("forbidden", C.Dma.Forbidden); ("allowed", C.Dma.Allowed) ];
  [%expect
    {|
    forbidden (2 invocations): bitwise
    allowed: refused: invocation 0 (n0): configuration mpact-m3 has no DMA engine (~dma Allowed); invocation 1 (n1): configuration mpact-m3 has no DMA engine (~dma Allowed) |}]

(* A DMA plan forced onto MPACT, past the route's own refusal ([~transfer] is
   a rule of one's own, which skips the check [~dma] makes): what does the
   simulator do when the image touches an engine the core does not have? *)
let%expect_test "a forced DMA plan on MPACT" =
  let module P = Coral_transfer.Transfer_plan in
  let forced ~budget:_ _ =
    { P.a = P.A.In_place; b = P.B.Copied; out = P.Out.Core }
  in
  T.check
    ~route:(C.route ~transfer:forced (config "mpact-m3"))
    ~contractions:true
    ~pipeline:(planned Ssa_ir.Ssa_target.zve32f128)
    ~oracle:T.Fp32 ~constant:T.positive ~calls:1 "forced DMA"
    (List.assoc "linear 9x32 k64" T.contraction_fixtures);
  [%expect
    {| forced DMA (1 invocations): invocation 0 (n0): unsupported(guest trap: mcause 7 mepc 0x0000066c mtval 0x40050008) |}]
