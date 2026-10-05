let%expect_test "vector sweep (forced), slice 2 of 4" =
  Ssa_vector_sweep_common.sweep ~config:"vector-forced"
    ~target:(Ssa_ir.Ssa_target.forced Ssa_ir.Ssa_target.wasm128)
    ~shard:2 ();
  Ssa_vector_sweep_common.report ~config:"vector-forced";
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
      branch                             loops=80 work=2512
      carries_values                     loops=12 work=0
      no_vector_form:check_gather        loops=4 work=64
      no_vector_form:load                loops=12 work=7744
      no_vector_form:load.in_bounds      loops=178 work=11598
      no_vector_form:select              loops=42 work=5072
      non_affine_access                  loops=40 work=7788
      too_short                          loops=206 work=14214
      vectorized                         loops=80 work=18652 |}]
