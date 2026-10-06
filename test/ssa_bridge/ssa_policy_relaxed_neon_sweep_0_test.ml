let%expect_test "policy sweep (relaxed_neon), slice 0 of 4" =
  Ssa_policy_sweep_common.sweep ~config:"policy-relaxed_neon"
    ~numerics:Ssa_ir.Ssa_numerics.Simd_fp32_relaxed
    ~target:Ssa_ir.Ssa_target.neon128 ~shard:0;
  Ssa_policy_sweep_common.report ~config:"policy-relaxed_neon";
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
      plans=228 binary32=24 vector-loops=56 scheduled-sums=0 fused-multiply-adds=0
      against the Loop plan: same-precision=162 same-vector-loops=164 more-vector-loops=0 fewer-vector-loops=64 same-blocked-rows=214 |}]
