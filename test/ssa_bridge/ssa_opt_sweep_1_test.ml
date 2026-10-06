let%expect_test "optimizer sweep, slice 1 of 4" =
  Ssa_opt_sweep_common.sweep ~shard:1 ~shards:4;
  Ssa_opt_sweep_common.report ();
  [%expect
    {|
    disagreements: 0, work differs: 0, more loads: 0
    add                      agree=12 failed-alike=0 instrs 280 -> 120, checked 24 -> 0, loops 72 -> 40
    arange_i64               agree=12 failed-alike=0 instrs 216 -> 96, checked 0 -> 0, loops 72 -> 12
    avg_pool2d               agree=12 failed-alike=0 instrs 670 -> 374, checked 234 -> 0, loops 96 -> 60
    bmm                      agree=12 failed-alike=0 instrs 300 -> 450, checked 24 -> 0, loops 84 -> 20
    conv2d_padding           agree=12 failed-alike=0 instrs 900 -> 454, checked 268 -> 0, loops 108 -> 72
    div_scalar               agree=12 failed-alike=0 instrs 216 -> 116, checked 12 -> 0, loops 72 -> 36
    eye                      agree=12 failed-alike=0 instrs 252 -> 138, checked 24 -> 0, loops 72 -> 36
    gelu                     agree=12 failed-alike=0 instrs 388 -> 238, checked 36 -> 0, loops 72 -> 36
    hardtanh                 agree=12 failed-alike=0 instrs 336 -> 132, checked 48 -> 0, loops 72 -> 36
    lstm                     agree=12 failed-alike=0 instrs 16376 -> 5724, checked 3972 -> 192, loops 492 -> 484
    mean                     agree=12 failed-alike=0 instrs 328 -> 156, checked 12 -> 0, loops 100 -> 48
    mul_scalar_i64           agree=12 failed-alike=0 instrs 228 -> 126, checked 12 -> 0, loops 72 -> 36
    permute                  agree=12 failed-alike=0 instrs 192 -> 78, checked 12 -> 0, loops 72 -> 28
    reshape                  agree=12 failed-alike=0 instrs 1348 -> 284, checked 1020 -> 12, loops 72 -> 12
    sigmoid                  agree=12 failed-alike=0 instrs 276 -> 162, checked 12 -> 0, loops 72 -> 32
    sqrt                     agree=12 failed-alike=0 instrs 204 -> 104, checked 12 -> 0, loops 72 -> 38
    to_copy_bool             agree=12 failed-alike=0 instrs 264 -> 130, checked 12 -> 0, loops 72 -> 36
    upsample_bilinear2d      agree=12 failed-alike=0 instrs 1656 -> 576, checked 552 -> 0, loops 72 -> 42 |}]
