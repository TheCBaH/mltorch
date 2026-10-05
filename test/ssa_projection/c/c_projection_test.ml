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
    pointwise exp, direct: agree
    pointwise exp, bridge: agree
    pointwise sqrt, direct: agree
    pointwise sqrt, bridge: agree
    lazy select, direct: agree
    lazy select, bridge: agree
    f16 decode, direct: agree
    f16 decode, bridge: agree
    i8 per channel, direct: agree
    i8 per channel, bridge: agree
    i64 read as a float, direct: agree
    i64 read as a float, bridge: agree
    i64 division, direct: agree
    i64 division, bridge: agree
    i64 division by zero, direct: agree on failure: i64_division_by_zero
    i64 division by zero, bridge: agree on failure: i64_division_by_zero
    i64 modular multiply, direct: agree
    i64 modular multiply, bridge: agree
    float to i64, direct: agree on failure: i64_from_float_nan
    float to i64, bridge: agree on failure: i64_from_float_nan
    max pool padded window, direct: agree
    max pool padded window, bridge: agree
    max pool index with NaN, direct: agree
    max pool index with NaN, bridge: agree
    gather in range and negative, direct: agree
    gather in range and negative, bridge: agree
    gather out of range, direct: agree on failure: gather_index_out_of_range
    gather out of range, bridge: agree on failure: gather_index_out_of_range
    region: scalar local, direct: agree
    region: scalar local, bridge: agree
    region: vector local, direct: agree
    region: vector local, bridge: agree
    region: vector read past its extent, direct: agree on failure: unbound_local
    region: vector read past its extent, bridge: agree on failure: unbound_local
    region: trace local, direct: agree
    region: trace local, bridge: agree
    region: inline scan, direct: agree
    region: inline scan, bridge: agree
    group: bool member, direct: agree
    group: bool member, bridge: agree
    max with NaN, direct: agree
    max with NaN, bridge: agree
    argmax index ties, direct: agree
    argmax index ties, bridge: agree
    argmax value NaN, direct: agree
    argmax value NaN, bridge: agree
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
