let%expect_test "vector sweep (forced), slice 0 of 4" =
  Ssa_vector_sweep_common.sweep ~config:"vector-forced"
    ~target:(Ssa_ir.Ssa_target.forced Ssa_ir.Ssa_target.wasm128)
    ~shard:0 ();
  Ssa_vector_sweep_common.report ~config:"vector-forced";
  [%expect
    {|
    disagreements: 0
    adaptive_avg_pool2d          agree=12 failed-alike=0
    amax                         agree=12 failed-alike=0
    arange                       agree=12 failed-alike=0
    batch_norm_no_stats          agree=12 failed-alike=0
    bitwise_not                  agree=12 failed-alike=0
    conv2d                       agree=12 failed-alike=0
    div                          agree=12 failed-alike=0
    expand                       agree=12 failed-alike=0
    hardswish                    agree=12 failed-alike=0
    linear                       agree=12 failed-alike=0
    max_pool2d_with_indices      agree=12 failed-alike=0
    mul_scalar                   agree=12 failed-alike=0
    pad                          agree=12 failed-alike=0
    relu                         agree=12 failed-alike=0
    sdpa                         agree=12 failed-alike=0
    softmax                      agree=12 failed-alike=0
    split_with_sizes_i64         agree=12 failed-alike=0
    sum                          agree=12 failed-alike=0
    upsample_bicubic2d           agree=12 failed-alike=0
      branch                             loops=86 work=11602
      carries_values                     loops=74 work=0
      no_vector_form:load.in_bounds      loops=60 work=2464
      no_vector_form:local.alloc         loops=32 work=9770
      no_vector_form:local.read          loops=22 work=2782
      no_vector_form:select              loops=24 work=1072
      non_affine_access                  loops=58 work=6068
      too_short                          loops=168 work=8592
      vectorized                         loops=154 work=16134 |}]
