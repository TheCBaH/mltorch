let%expect_test "op sweep, slice 0 of 4" =
  Loop_sweep_common.sweep ~shard:0 ~shards:4;
  Loop_sweep_common.report ();
  [%expect
    {|
    disagreements: 0
    adaptive_avg_pool2d          agree=12 failed-alike=0
    amax                         agree=12 failed-alike=0
    arange                       agree=12 failed-alike=0
    batch_norm_no_stats          agree=12 failed-alike=0
    bitwise_not                  agree=12 failed-alike=0
    conv2d                       agree=12 failed-alike=0
    div                          agree=12 failed-alike=0
    expand                       agree=12 failed-alike=0
    hardswish                    agree=12 failed-alike=0
    linear                       agree=12 failed-alike=0
    max_pool2d_with_indices      agree=12 failed-alike=0
    mul_scalar                   agree=12 failed-alike=0
    pad                          agree=12 failed-alike=0
    relu                         agree=12 failed-alike=0
    sdpa                         agree=12 failed-alike=0
    softmax                      agree=12 failed-alike=0
    split_with_sizes_i64         agree=12 failed-alike=0
    sum                          agree=12 failed-alike=0
    upsample_bicubic2d           agree=12 failed-alike=0 |}]
