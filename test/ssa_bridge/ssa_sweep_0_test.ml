let%expect_test "op sweep, slice 0 of 4" =
  Ssa_sweep_common.sweep ~shard:0 ~shards:4;
  Ssa_sweep_common.report ();
  [%expect {|
    disagreements: 0
    adaptive_avg_pool2d          agree=0 failed-alike=0 refused: min, max, clamp or division on an index
    amax                         agree=0 failed-alike=0 refused: max reduction
    arange                       agree=12 failed-alike=0
    batch_norm_no_stats          agree=0 failed-alike=0 refused: unary operation
    bitwise_not                  agree=0 failed-alike=0 refused: select
    conv2d                       agree=0 failed-alike=0 refused: min, max, clamp or division on an index
    div                          agree=12 failed-alike=0
    expand                       agree=12 failed-alike=0
    hardswish                    agree=0 failed-alike=0 refused: select
    linear                       agree=10 failed-alike=0 refused: filled input
    max_pool2d_with_indices      agree=0 failed-alike=0 refused: int64 value
    mul_scalar                   agree=12 failed-alike=0
    pad                          agree=0 failed-alike=0 refused: select, min, max, clamp or division on an index
    relu                         agree=0 failed-alike=0 refused: select
    sdpa                         agree=0 failed-alike=0 refused: region program
    softmax                      agree=0 failed-alike=0 refused: region program
    split_with_sizes_i64         agree=0 failed-alike=0 refused: int64 value
    sum                          agree=12 failed-alike=0
    upsample_bicubic2d           agree=0 failed-alike=0 refused: min, max, clamp or division on an index |}]
