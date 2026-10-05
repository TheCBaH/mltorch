let%expect_test
    "SSA programs through generated JavaScript agree with the reference" =
  Projection.run ~exec:(Loop_js_exec.exec ?outputs:None);
  [%expect
    {|
    pointwise specials, direct: agree
    pointwise specials, bridge: agree
    pointwise specials, optimized: agree
    pointwise f32 boundary, direct: agree
    pointwise f32 boundary, bridge: agree
    pointwise f32 boundary, optimized: agree
    pointwise sub and div, direct: agree
    pointwise sub and div, bridge: agree
    pointwise sub and div, optimized: agree
    pointwise exp, direct: agree
    pointwise exp, bridge: agree
    pointwise exp, optimized: agree
    pointwise sqrt, direct: agree
    pointwise sqrt, bridge: agree
    pointwise sqrt, optimized: agree
    lazy select, direct: agree
    lazy select, bridge: agree
    lazy select, optimized: agree
    f16 decode, direct: agree
    f16 decode, bridge: agree
    f16 decode, optimized: agree
    i8 per channel, direct: agree
    i8 per channel, bridge: agree
    i8 per channel, optimized: agree
    i64 read as a float, direct: agree
    i64 read as a float, bridge: agree
    i64 read as a float, optimized: agree
    i64 division, direct: agree
    i64 division, bridge: agree
    i64 division, optimized: agree
    i64 division by zero, direct: agree on failure: i64_division_by_zero
    i64 division by zero, bridge: agree on failure: i64_division_by_zero
    i64 division by zero, optimized: agree on failure: i64_division_by_zero
    i64 modular multiply, direct: agree
    i64 modular multiply, bridge: agree
    i64 modular multiply, optimized: agree
    float to i64, direct: agree on failure: i64_from_float_nan
    float to i64, bridge: agree on failure: i64_from_float_nan
    float to i64, optimized: agree on failure: i64_from_float_nan
    max pool padded window, direct: agree
    max pool padded window, bridge: agree
    max pool padded window, optimized: agree
    max pool index with NaN, direct: agree
    max pool index with NaN, bridge: agree
    max pool index with NaN, optimized: agree
    gather in range and negative, direct: agree
    gather in range and negative, bridge: agree
    gather in range and negative, optimized: agree
    gather out of range, direct: agree on failure: gather_index_out_of_range
    gather out of range, bridge: agree on failure: gather_index_out_of_range
    gather out of range, optimized: agree on failure: gather_index_out_of_range
    region: scalar local, direct: agree
    region: scalar local, bridge: agree
    region: scalar local, optimized: agree
    region: vector local, direct: agree
    region: vector local, bridge: agree
    region: vector local, optimized: agree
    region: vector read past its extent, direct: agree on failure: unbound_local
    region: vector read past its extent, bridge: agree on failure: unbound_local
    region: vector read past its extent, optimized: agree on failure: unbound_local
    region: trace local, direct: agree
    region: trace local, bridge: agree
    region: trace local, optimized: agree
    region: inline scan, direct: agree
    region: inline scan, bridge: agree
    region: inline scan, optimized: agree
    group: bool member, direct: agree
    group: bool member, bridge: agree
    group: bool member, optimized: agree
    max with NaN, direct: agree
    max with NaN, bridge: agree
    max with NaN, optimized: agree
    argmax index ties, direct: agree
    argmax index ties, bridge: agree
    argmax index ties, optimized: agree
    argmax value NaN, direct: agree
    argmax value NaN, bridge: agree
    argmax value NaN, optimized: agree
    matmul 1x1x1, direct: agree
    matmul 1x1x1, bridge: agree
    matmul 1x1x1, optimized: agree
    matmul 1x3x2, direct: agree
    matmul 1x3x2, bridge: agree
    matmul 1x3x2, optimized: agree
    matmul 5x7x3, direct: agree
    matmul 5x7x3, bridge: agree
    matmul 5x7x3, optimized: agree
    matmul 4x4x4, direct: agree
    matmul 4x4x4, bridge: agree
    matmul 4x4x4, optimized: agree
    sum [0,0), direct: agree
    sum [0,0), bridge: agree
    sum [0,0), optimized: agree
    sum [2,1), direct: agree
    sum [2,1), bridge: agree
    sum [2,1), optimized: agree
    sum [0,3), direct: agree
    sum [0,3), bridge: agree
    sum [0,3), optimized: agree
    chain, inlined producer, direct: agree
    chain, inlined producer, bridge: agree
    chain, inlined producer, optimized: agree |}]
