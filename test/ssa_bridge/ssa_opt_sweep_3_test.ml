let%expect_test "optimizer sweep, slice 3 of 4" =
  Ssa_opt_sweep_common.sweep ~shard:3 ~shards:4;
  Ssa_opt_sweep_common.report ();
  [%expect
    {|
    disagreements: 0, work differs: 0, more loads: 0
    add_scalar               agree=12 failed-alike=0 instrs 216 -> 114, checked 12 -> 0, loops 72 -> 36
    batched_matmul           agree=12 failed-alike=0 instrs 320 -> 144, checked 24 -> 0, loops 84 -> 56
    clone                    agree=12 failed-alike=0 instrs 192 -> 94, checked 12 -> 0, loops 72 -> 36
    cumsum                   agree=12 failed-alike=0 instrs 252 -> 138, checked 24 -> 0, loops 84 -> 60
    eq_tensor                agree=12 failed-alike=0 instrs 320 -> 146, checked 24 -> 0, loops 72 -> 44
    hardsigmoid              agree=12 failed-alike=0 instrs 456 -> 176, checked 48 -> 0, loops 72 -> 40
    layer_norm               agree=12 failed-alike=0 instrs 1034 -> 604, checked 72 -> 0, loops 96 -> 68
    max_pool2d               agree=12 failed-alike=0 instrs 492 -> 302, checked 108 -> 0, loops 96 -> 60
    mul_i64                  agree=12 failed-alike=0 instrs 264 -> 94, checked 24 -> 0, loops 72 -> 36
    ne_tensor                agree=12 failed-alike=0 instrs 336 -> 142, checked 24 -> 0, loops 72 -> 36
    pow                      agree=12 failed-alike=0 instrs 240 -> 124, checked 20 -> 0, loops 72 -> 36
    rms_norm                 agree=12 failed-alike=0 instrs 722 -> 394, checked 48 -> 0, loops 104 -> 68
    slice                    agree=12 failed-alike=0 instrs 210 -> 102, checked 18 -> 0, loops 72 -> 48
    split_with_sizes         agree=12 failed-alike=0 instrs 964 -> 360, checked 92 -> 0, loops 312 -> 174
    sub_i64                  agree=12 failed-alike=0 instrs 264 -> 90, checked 24 -> 0, loops 72 -> 36
    to_copy_long             agree=12 failed-alike=0 instrs 180 -> 76, checked 24 -> 12, loops 72 -> 36
    unbind                   agree=12 failed-alike=0 instrs 576 -> 182, checked 32 -> 0, loops 192 -> 96
    vector_norm              agree=12 failed-alike=0 instrs 324 -> 148, checked 24 -> 0, loops 96 -> 48 |}]
