let%expect_test "policy sweep (ordered_neon), slice 3 of 4" =
  Ssa_policy_sweep_common.sweep ~config:"policy-ordered_neon"
    ~numerics:Ssa_ir.Ssa_numerics.Simd_fp32_ordered
    ~target:Ssa_ir.Ssa_target.neon128 ~shard:3;
  Ssa_policy_sweep_common.report ~config:"policy-ordered_neon";
  [%expect
    {|
    disagreements: 0
    add_scalar                   agree=12 failed-alike=0
    batched_matmul               agree=12 failed-alike=0
    clone                        agree=12 failed-alike=0
    cumsum                       agree=12 failed-alike=0
    eq_tensor                    agree=12 failed-alike=0
    hardsigmoid                  agree=12 failed-alike=0
    layer_norm                   agree=12 failed-alike=0
    max_pool2d                   agree=12 failed-alike=0
    mul_i64                      agree=12 failed-alike=0
    ne_tensor                    agree=12 failed-alike=0
    pow                          agree=12 failed-alike=0
    rms_norm                     agree=12 failed-alike=0
    slice                        agree=12 failed-alike=0
    split_with_sizes             agree=12 failed-alike=0
    sub_i64                      agree=12 failed-alike=0
    to_copy_long                 agree=12 failed-alike=0
    unbind                       agree=12 failed-alike=0
    vector_norm                  agree=12 failed-alike=0
      plans=216 binary32=6 vector-loops=86 scheduled-sums=0 fused-multiply-adds=0
      against the Loop plan: same-precision=106 same-vector-loops=114 more-vector-loops=0 fewer-vector-loops=102 same-blocked-rows=216 |}]
