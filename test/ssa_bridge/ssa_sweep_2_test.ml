let%expect_test "op sweep, slice 2 of 4" =
  Ssa_sweep_common.sweep ~shard:2 ~shards:4;
  Ssa_sweep_common.report ();
  [%expect {|
    disagreements: 0
    add_i64                      agree=0 failed-alike=0 refused: int64 value
    batch_norm                   agree=0 failed-alike=0 refused: unary operation
    clamp                        agree=4 failed-alike=0 refused: select
    convolution                  agree=0 failed-alike=0 refused: min, max, clamp or division on an index
    eq_scalar                    agree=0 failed-alike=0 refused: select
    gt_scalar                    agree=0 failed-alike=0 refused: select
    index_tensor                 agree=0 failed-alike=0 refused: gather index
    max_dim                      agree=0 failed-alike=0 refused: int64 value
    mul                          agree=12 failed-alike=0
    ne_scalar                    agree=0 failed-alike=0 refused: select
    permute_i64                  agree=0 failed-alike=0 refused: int64 value
    reshape_i64                  agree=0 failed-alike=0 refused: int64 value
    silu                         agree=0 failed-alike=0 refused: unary operation
    sub                          agree=12 failed-alike=0
    to_copy_float_i64            agree=0 failed-alike=0 refused: int64 value
    unbind_i64                   agree=0 failed-alike=0 refused: int64 value
    upsample_nearest2d           agree=0 failed-alike=0 refused: min, max, clamp or division on an index
    zeros                        agree=12 failed-alike=0 |}]
