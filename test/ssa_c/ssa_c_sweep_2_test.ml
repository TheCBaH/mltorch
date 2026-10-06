let%expect_test "C sweep, slice 2 of 8" =
  let open Ssa_c_sweep_common in
  sweep ~config:"lowered" ~prepare:Fun.id ~shard:2 ~shards:8 ();
  sweep ~config:"optimized" ~prepare:optimize ~shard:2 ~shards:8 ();
  report ~config:"lowered";
  report ~config:"optimized";
  [%expect
    {|
    lowered: disagreements 0, agree 108, failed alike 0, refused 0; with binary32 72, vectors 0, fused 0
    optimized: disagreements 0, agree 108, failed alike 0, refused 0; with binary32 56, vectors 0, fused 0 |}]

let%expect_test "C sweep of vector and numerical plans, slice 2 of 8" =
  let open Ssa_c_sweep_common in
  let neon = Ssa_ir.Ssa_target.forced Ssa_ir.Ssa_target.neon128 in
  List.iter
    (fun (config, numerics) ->
      sweep ~engine:Interpreter ~config
        ~prepare:(planned ~numerics ~target:neon)
        ~shard:2 ~shards:8 ();
      report ~config)
    [
      ("strict-vectors", Ssa_ir.Ssa_numerics.Reference_f64);
      ("ordered-binary32", Ssa_ir.Ssa_numerics.Simd_fp32_ordered);
      ("relaxed-binary32", Ssa_ir.Ssa_numerics.Simd_fp32_relaxed);
    ];
  [%expect
    {|
    strict-vectors: disagreements 0, agree 108, failed alike 0, refused 0; with binary32 56, vectors 4, fused 0
    ordered-binary32: disagreements 0, agree 108, failed alike 0, refused 0; with binary32 60, vectors 8, fused 0
    relaxed-binary32: disagreements 0, agree 108, failed alike 0, refused 0; with binary32 60, vectors 8, fused 0 |}]
