let%expect_test "op sweep, slice 3 of 4" =
  Ssa_sweep_common.sweep ~shard:3 ~shards:4;
  Ssa_sweep_common.report ();
  [%expect {|
    disagreements: 0
    add_scalar                   agree=12 failed-alike=0
    batched_matmul               agree=12 failed-alike=0
    clone                        agree=12 failed-alike=0
    cumsum                       agree=12 failed-alike=0
    eq_tensor                    agree=0 failed-alike=0 refused: select
    hardsigmoid                  agree=0 failed-alike=0 refused: select
    layer_norm                   agree=0 failed-alike=0 refused: region program
    max_pool2d                   agree=0 failed-alike=0 refused: intrinsic
    mul_i64                      agree=0 failed-alike=0 refused: int64 value
    ne_tensor                    agree=0 failed-alike=0 refused: select
    pow                          agree=4 failed-alike=0 refused: unary operation
    rms_norm                     agree=0 failed-alike=0 refused: region program
    slice                        agree=0 failed-alike=0 refused: min, max, clamp or division on an index
    split_with_sizes             agree=0 failed-alike=0 refused: min, max, clamp or division on an index
    sub_i64                      agree=0 failed-alike=0 refused: int64 value
    to_copy_long                 agree=0 failed-alike=0 refused: int64 value
    unbind                       agree=0 failed-alike=0 refused: min, max, clamp or division on an index
    vector_norm                  agree=0 failed-alike=0 refused: unary operation |}]
