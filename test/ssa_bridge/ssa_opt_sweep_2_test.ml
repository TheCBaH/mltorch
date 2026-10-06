let%expect_test "optimizer sweep, slice 2 of 4" =
  Ssa_opt_sweep_common.sweep ~shard:2 ~shards:4;
  Ssa_opt_sweep_common.report ();
  [%expect
    {|
    disagreements: 0, work differs: 0, more loads: 0
    add_i64                  agree=12 failed-alike=0 instrs 264 -> 84, checked 24 -> 0, loops 72 -> 36
    batch_norm               agree=12 failed-alike=0 instrs 588 -> 256, checked 60 -> 0, loops 72 -> 48
    clamp                    agree=12 failed-alike=0 instrs 288 -> 114, checked 32 -> 0, loops 72 -> 36
    convolution              agree=12 failed-alike=0 instrs 840 -> 460, checked 238 -> 0, loops 108 -> 78
    eq_scalar                agree=12 failed-alike=0 instrs 264 -> 140, checked 12 -> 0, loops 72 -> 38
    gt_scalar                agree=12 failed-alike=0 instrs 264 -> 138, checked 12 -> 0, loops 72 -> 42
    index_tensor             agree=12 failed-alike=0 instrs 348 -> 174, checked 48 -> 36, loops 72 -> 28
    max_dim                  agree=12 failed-alike=0 instrs 552 -> 238, checked 36 -> 12, loops 168 -> 96
    mul                      agree=12 failed-alike=0 instrs 288 -> 116, checked 24 -> 0, loops 72 -> 36
    ne_scalar                agree=12 failed-alike=0 instrs 264 -> 134, checked 12 -> 0, loops 72 -> 36
    permute_i64              agree=12 failed-alike=0 instrs 168 -> 62, checked 12 -> 0, loops 72 -> 36
    reshape_i64              agree=12 failed-alike=0 instrs 1340 -> 300, checked 1020 -> 12, loops 72 -> 12
    silu                     agree=12 failed-alike=0 instrs 276 -> 164, checked 24 -> 0, loops 72 -> 36
    sub                      agree=12 failed-alike=0 instrs 280 -> 122, checked 24 -> 0, loops 72 -> 40
    to_copy_float_i64        agree=12 failed-alike=0 instrs 204 -> 110, checked 12 -> 0, loops 72 -> 42
    unbind_i64               agree=12 failed-alike=0 instrs 576 -> 126, checked 36 -> 0, loops 216 -> 108
    upsample_nearest2d       agree=12 failed-alike=0 instrs 240 -> 132, checked 36 -> 0, loops 72 -> 36
    zeros                    agree=12 failed-alike=0 instrs 192 -> 76, checked 0 -> 0, loops 72 -> 44 |}]
