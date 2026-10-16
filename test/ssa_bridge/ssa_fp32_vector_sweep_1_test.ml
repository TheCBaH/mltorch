let%expect_test "binary32 vector sweep, slice 1 of 4" =
  Ssa_fp32_vector_sweep_common.sweep ~shard:1;
  Ssa_fp32_vector_sweep_common.report ();
  [%expect
    {|
    disagreements: 0
    add                          agree=12 failed-alike=0
    arange_i64                   agree=12 failed-alike=0
    avg_pool2d                   agree=12 failed-alike=0
    bmm                          agree=12 failed-alike=0
    conv2d_padding               agree=12 failed-alike=0
    div_scalar                   agree=12 failed-alike=0
    eye                          agree=12 failed-alike=0
    gelu                         agree=12 failed-alike=0
    hardtanh                     agree=12 failed-alike=0
    lstm                         agree=12 failed-alike=0
    mean                         agree=12 failed-alike=0
    mul_scalar_i64               agree=12 failed-alike=0
    permute                      agree=12 failed-alike=0
    reshape                      agree=12 failed-alike=0
    sigmoid                      agree=12 failed-alike=0
    sqrt                         agree=12 failed-alike=0
    to_copy_bool                 agree=12 failed-alike=0
    upsample_bilinear2d          agree=12 failed-alike=0
      plans=864 vector-loops=48 forced-vector-loops=48 wide-vectors=0 inherited-from-scalar=0 |}]
