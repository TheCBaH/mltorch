let%expect_test "optimizer sweep, slice 0 of 4" =
  Ssa_opt_sweep_common.sweep ~shard:0 ~shards:4;
  Ssa_opt_sweep_common.report ();
  [%expect
    {|
    disagreements: 0, work differs: 0, more loads: 0
    adaptive_avg_pool2d      agree=12 failed-alike=0 instrs 624 -> 354, checked 204 -> 0, loops 96 -> 70
    amax                     agree=12 failed-alike=0 instrs 312 -> 156, checked 12 -> 0, loops 96 -> 42
    arange                   agree=12 failed-alike=0 instrs 240 -> 120, checked 0 -> 0, loops 72 -> 12
    batch_norm_no_stats      agree=12 failed-alike=0 instrs 3168 -> 1120, checked 156 -> 0, loops 696 -> 552
    bitwise_not              agree=12 failed-alike=0 instrs 264 -> 128, checked 12 -> 0, loops 72 -> 40
    conv2d                   agree=12 failed-alike=0 instrs 810 -> 416, checked 230 -> 0, loops 108 -> 72
    div                      agree=12 failed-alike=0 instrs 292 -> 114, checked 24 -> 0, loops 72 -> 34
    expand                   agree=12 failed-alike=0 instrs 240 -> 88, checked 12 -> 0, loops 72 -> 36
    hardswish                agree=12 failed-alike=0 instrs 480 -> 182, checked 60 -> 0, loops 72 -> 42
    linear                   agree=12 failed-alike=0 instrs 398 -> 606, checked 36 -> 0, loops 84 -> 42
    max_pool2d_with_indices  agree=12 failed-alike=0 instrs 984 -> 534, checked 228 -> 12, loops 192 -> 120
    mul_scalar               agree=12 failed-alike=0 instrs 216 -> 108, checked 12 -> 0, loops 72 -> 36
    pad                      agree=12 failed-alike=0 instrs 720 -> 316, checked 252 -> 0, loops 72 -> 36
    relu                     agree=12 failed-alike=0 instrs 240 -> 118, checked 24 -> 0, loops 72 -> 36
    sdpa                     agree=12 failed-alike=0 instrs 1362 -> 690, checked 96 -> 0, loops 144 -> 78
    softmax                  agree=12 failed-alike=0 instrs 588 -> 372, checked 36 -> 0, loops 96 -> 66
    split_with_sizes_i64     agree=12 failed-alike=0 instrs 384 -> 122, checked 36 -> 0, loops 144 -> 84
    sum                      agree=12 failed-alike=0 instrs 296 -> 122, checked 12 -> 0, loops 98 -> 48
    upsample_bicubic2d       agree=12 failed-alike=0 instrs 15336 -> 1946, checked 4464 -> 0, loops 72 -> 42 |}]
