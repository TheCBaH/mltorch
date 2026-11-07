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
      branch                             loops=98 work=12112
      carries_values                     loops=74 work=0
      no_vector_form:load.in_bounds      loops=60 work=2464
      no_vector_form:local.alloc         loops=20 work=9090
      no_vector_form:local.read          loops=2 work=48
      no_vector_form:local.write         loops=2 work=48
      no_vector_form:select              loops=24 work=1072
      non_affine_access                  loops=58 work=6068
      too_short                          loops=162 work=8580
      vectorized                         loops=160 work=18310 |}]
