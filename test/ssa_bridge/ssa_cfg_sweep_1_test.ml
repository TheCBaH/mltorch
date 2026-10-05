let%expect_test "graph sweep, slice 1 of 4" =
  Ssa_cfg_sweep_common.run ~shard:1;
  Ssa_cfg_sweep_common.report ();
  [%expect
    {|
    cfg-lowered
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
    cfg-optimized
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
    cfg-vectorized
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
    upsample_bilinear2d          agree=12 failed-alike=0 |}]
