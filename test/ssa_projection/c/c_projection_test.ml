let%expect_test "SSA programs through generated C agree with the reference" =
  Projection.run ~exec:Loop_c_exec.executor;
  [%expect
    {|
    pointwise specials, direct: agree
    pointwise specials, bridge: agree
    pointwise f32 boundary, direct: agree
    pointwise f32 boundary, bridge: agree
    pointwise sub and div, direct: agree
    pointwise sub and div, bridge: agree
    matmul 1x1x1, direct: agree
    matmul 1x1x1, bridge: agree
    matmul 1x3x2, direct: agree
    matmul 1x3x2, bridge: agree
    matmul 5x7x3, direct: agree
    matmul 5x7x3, bridge: agree
    matmul 4x4x4, direct: agree
    matmul 4x4x4, bridge: agree
    sum [0,0), direct: agree
    sum [0,0), bridge: agree
    sum [2,1), direct: agree
    sum [2,1), bridge: agree
    sum [0,3), direct: agree
    sum [0,3), bridge: agree
    chain, inlined producer, direct: agree
    chain, inlined producer, bridge: agree |}]
