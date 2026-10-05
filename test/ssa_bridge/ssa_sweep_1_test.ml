let%expect_test "op sweep, slice 1 of 4" =
  Ssa_sweep_common.sweep ~shard:1 ~shards:4;
  Ssa_sweep_common.report ();
  [%expect {|
    disagreements: 0
    add                          agree=12 failed-alike=0
    arange_i64                   agree=0 failed-alike=0 refused: int64 value
    avg_pool2d                   agree=0 failed-alike=0 refused: min, max, clamp or division on an index
    bmm                          agree=12 failed-alike=0
    conv2d_padding               agree=0 failed-alike=0 refused: min, max, clamp or division on an index
    div_scalar                   agree=12 failed-alike=0
    eye                          agree=0 failed-alike=0 refused: select
    gelu                         agree=0 failed-alike=0 refused: unary operation
    hardtanh                     agree=0 failed-alike=0 refused: select
    lstm                         agree=0 failed-alike=0 refused: region program
    mean                         agree=12 failed-alike=0
    mul_scalar_i64               agree=0 failed-alike=0 refused: int64 value
    permute                      agree=12 failed-alike=0
    reshape                      agree=0 failed-alike=0 refused: min, max, clamp or division on an index
    sigmoid                      agree=0 failed-alike=0 refused: unary operation
    sqrt                         agree=0 failed-alike=0 refused: unary operation
    to_copy_bool                 agree=0 failed-alike=0 refused: select
    upsample_bilinear2d          agree=0 failed-alike=0 refused: min, max, clamp or division on an index |}]
