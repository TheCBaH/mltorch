let%expect_test "op sweep, slice 3 of 4" =
  Ssa_sweep_common.sweep ~shard:3 ~shards:4;
  Ssa_sweep_common.report ();
  [%expect
    {|
    disagreements: 0
    add_scalar                   agree=12 failed-alike=0
    batched_matmul               agree=12 failed-alike=0
    clone                        agree=12 failed-alike=0
    cumsum                       agree=12 failed-alike=0
    eq_tensor                    agree=12 failed-alike=0
    hardsigmoid                  agree=12 failed-alike=0
    layer_norm                   agree=0 failed-alike=0 refused: region program
    max_pool2d                   agree=12 failed-alike=0
    mul_i64                      agree=12 failed-alike=0
    ne_tensor                    agree=12 failed-alike=0
    pow                          agree=12 failed-alike=0
    rms_norm                     agree=0 failed-alike=0 refused: region program
    slice                        agree=12 failed-alike=0
    split_with_sizes             agree=12 failed-alike=0
    sub_i64                      agree=12 failed-alike=0
    to_copy_long                 agree=12 failed-alike=0
    unbind                       agree=12 failed-alike=0
    vector_norm                  agree=12 failed-alike=0 |}]
