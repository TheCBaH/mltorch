let%expect_test "op sweep, slice 2 of 4" =
  Ssa_sweep_common.sweep ~shard:2 ~shards:4;
  Ssa_sweep_common.report ();
  [%expect
    {|
    disagreements: 0
    add_i64                      agree=12 failed-alike=0
    batch_norm                   agree=12 failed-alike=0
    clamp                        agree=12 failed-alike=0
    convolution                  agree=12 failed-alike=0
    eq_scalar                    agree=12 failed-alike=0
    gt_scalar                    agree=12 failed-alike=0
    index_tensor                 agree=12 failed-alike=0
    max_dim                      agree=12 failed-alike=0
    mul                          agree=12 failed-alike=0
    ne_scalar                    agree=12 failed-alike=0
    permute_i64                  agree=12 failed-alike=0
    reshape_i64                  agree=12 failed-alike=0
    silu                         agree=12 failed-alike=0
    sub                          agree=12 failed-alike=0
    to_copy_float_i64            agree=12 failed-alike=0
    unbind_i64                   agree=12 failed-alike=0
    upsample_nearest2d           agree=12 failed-alike=0
    zeros                        agree=12 failed-alike=0 |}]
