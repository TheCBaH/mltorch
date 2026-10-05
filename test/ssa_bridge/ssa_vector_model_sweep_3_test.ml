let%expect_test "vector sweep (model), slice 3 of 4" =
  Ssa_vector_sweep_common.sweep ~config:"vector-model"
    ~target:Ssa_ir.Ssa_target.wasm128 ~shard:3 ();
  Ssa_vector_sweep_common.report ~config:"vector-model";
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
      branch                             loops=76 work=5920
      carries_values                     loops=24 work=0
      no_vector_form:float.to_i64        loops=32 work=9208
      no_vector_form:load.in_bounds      loops=60 work=6564
      no_vector_form:local.alloc         loops=12 work=72
      no_vector_form:local.read          loops=34 work=1900
      no_vector_form:select              loops=12 work=576
      non_affine_access                  loops=14 work=144
      too_short                          loops=160 work=9472
      unprofitable                       loops=122 work=4392
      varying_bounds                     loops=6 work=1088
      vectorized                         loops=110 work=6400 |}]
