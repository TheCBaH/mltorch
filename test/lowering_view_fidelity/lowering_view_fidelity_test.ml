let%expect_test "the inspector shows the kernel the builders used" =
  Fidelity.run ();
  [%expect
    {|
    C loop reference: 415 invocations, 415 equal
    C loop simd_fp32_relaxed: 415 invocations, 415 equal
    C ssa planned: 415 invocations, 415 equal
    JS loop: 415 invocations, 415 equal
    JS ssa planned: 415 invocations, 415 equal
    WAT loop reference (0 with constant tables, rebased: skipped): 415 invocations, 415 equal
    WAT loop simd_fp32_relaxed (0 with constant tables, rebased: skipped): 415 invocations, 415 equal
    WAT ssa planned (0 with constant tables, rebased: skipped): 415 invocations, 415 equal |}]
