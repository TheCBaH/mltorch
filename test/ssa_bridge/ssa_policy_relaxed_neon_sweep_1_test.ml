let%expect_test "policy sweep (relaxed_neon), slice 1 of 4" =
  Ssa_policy_sweep_common.sweep ~config:"policy-relaxed_neon"
    ~numerics:Ssa_ir.Ssa_numerics.Simd_fp32_relaxed
    ~target:Ssa_ir.Ssa_target.neon128 ~shard:1;
  Ssa_policy_sweep_common.report ~config:"policy-relaxed_neon";
  [%expect
    {|
    disagreements: 0
    add                          agree=12 failed-alike=0
    arange_i64                   agree=12 failed-alike=0
    avg_pool2d                   agree=12 failed-alike=0
    bmm                          agree=12 failed-alike=0
    conv2d_padding               agree=12 failed-alike=0
    div_scalar                   agree=12 failed-alike=0
    eye                          agree=12 failed-alike=0
    gelu                         agree=12 failed-alike=0
    hardtanh                     agree=12 failed-alike=0
    lstm                         agree=12 failed-alike=0
    mean                         agree=12 failed-alike=0
    mul_scalar_i64               agree=12 failed-alike=0
    permute                      agree=12 failed-alike=0
    reshape                      agree=12 failed-alike=0
    sigmoid                      agree=12 failed-alike=0
    sqrt                         agree=12 failed-alike=0
    to_copy_bool                 agree=12 failed-alike=0
    upsample_bilinear2d          agree=12 failed-alike=0
      plans=216 binary32=24 vector-loops=46 scheduled-sums=0 fused-multiply-adds=8
      against the Loop plan: same-precision=132 same-vector-loops=140 more-vector-loops=0 fewer-vector-loops=76 same-blocked-rows=212 |}]
