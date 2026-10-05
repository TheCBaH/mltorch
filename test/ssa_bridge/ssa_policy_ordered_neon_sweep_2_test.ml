let%expect_test "policy sweep (ordered_neon), slice 2 of 4" =
  Ssa_policy_sweep_common.sweep ~config:"policy-ordered_neon"
    ~numerics:Ssa_ir.Ssa_numerics.Simd_fp32_ordered
    ~target:Ssa_ir.Ssa_target.neon128 ~shard:2;
  Ssa_policy_sweep_common.report ~config:"policy-ordered_neon";
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
    zeros                        agree=12 failed-alike=0
      plans=216 binary32=12 vector-loops=34 scheduled-sums=0 fused-multiply-adds=0 |}]
