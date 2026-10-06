let%expect_test "vector sweep (forced), slice 1 of 4" =
  Ssa_vector_sweep_common.sweep ~config:"vector-forced"
    ~target:(Ssa_ir.Ssa_target.forced Ssa_ir.Ssa_target.wasm128)
    ~shard:1 ();
  Ssa_vector_sweep_common.report ~config:"vector-forced";
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
      branch                             loops=104 work=2384
      no_vector_form:convert.index_to_i64 loops=12 work=156
      no_vector_form:load.in_bounds      loops=24 work=440
      no_vector_form:meter.charge        loops=48 work=2016
      non_affine_access                  loops=46 work=7266
      too_short                          loops=128 work=7334
      vectorized                         loops=128 work=22060 |}]
