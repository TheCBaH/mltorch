module C = Machine_rivet_riscv32_coral.Rv32_mir_coral
module S = Machine_rivet_riscv32_coral_vme_test.Sdpa_composition

let config =
  {
    C.Config.dir = Sys.getenv "CORALNPU_DATA";
    configuration = "vme-core-mini-axi";
    deadline = 3600.;
    flags = [ "--cycles=500000000" ];
  }

let%expect_test "the per-op composition of Sdpa at 64x64" =
  S.run config ~tq:64 ~tk:64 ~d:64;
  [%expect
    {|
    sdpa composition 64x64 d64 (5 invocations): bitwise
      cycles 4211318 = 30616 + 13334 + 4092795 + 16267 + 58306 |}]

let%expect_test "generated exp and softmax cycles per element at 64x64" =
  Machine_rivet_riscv32_coral_vme_test.Exp_softmax.run config
    ~sizes:[ (64, 64) ];
  [%expect
    {|
    exp 64x64, planned (1 invocations): bitwise
      cycles 607246, 148.3 per element
    softmax over C 64x64, planned (1 invocations): bitwise
      cycles 4190117, 1023.0 per element
    softmax over W 64x64, planned (1 invocations): bitwise
      cycles 4190117, 1023.0 per element
    exp 64x64, forced vector (1 invocations): bitwise
      cycles 607246, 148.3 per element
    softmax over C 64x64, forced vector (1 invocations): bitwise
      cycles 4190117, 1023.0 per element
    softmax over W 64x64, forced vector (1 invocations): bitwise
      cycles 4190117, 1023.0 per element |}]
