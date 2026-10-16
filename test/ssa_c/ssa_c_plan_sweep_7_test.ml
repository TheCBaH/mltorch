let%expect_test "C sweep of vector and numerical plans, slice 7 of 8" =
  let open Ssa_c_sweep_common in
  let neon = Ssa_ir.Ssa_target.forced Ssa_ir.Ssa_target.neon128 in
  List.iter
    (fun (config, numerics) ->
      sweep ~engine:Interpreter ~config
        ~prepare:(planned ~numerics ~target:neon)
        ~shard:7 ~shards:8 ();
      report ~config)
    [
      ("strict-vectors", Ssa_ir.Ssa_numerics.Reference_f64);
      ("ordered-binary32", Ssa_ir.Ssa_numerics.Simd_fp32_ordered);
      ("relaxed-binary32", Ssa_ir.Ssa_numerics.Simd_fp32_relaxed);
      ("fp32-vectors", Ssa_ir.Ssa_numerics.Fp32_ordered);
    ];
  [%expect
    {|
    strict-vectors: disagreements 0, agree 108, failed alike 0, refused 0; with binary32 72, vectors 16, fused 0
    ordered-binary32: disagreements 0, agree 108, failed alike 0, refused 0; with binary32 72, vectors 16, fused 0
    relaxed-binary32: disagreements 0, agree 108, failed alike 0, refused 0; with binary32 72, vectors 16, fused 0
    fp32-vectors: disagreements 0, agree 108, failed alike 0, refused 0; with binary32 96, vectors 0, fused 0 |}]
