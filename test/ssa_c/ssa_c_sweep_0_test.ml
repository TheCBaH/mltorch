let%expect_test "C sweep, slice 0 of 8" =
  let open Ssa_c_sweep_common in
  sweep ~config:"lowered" ~prepare:Fun.id ~shard:0 ~shards:8 ();
  sweep ~config:"optimized" ~prepare:optimize ~shard:0 ~shards:8 ();
  report ~config:"lowered";
  report ~config:"optimized";
  [%expect
    {|
    lowered: disagreements 0, agree 120, failed alike 0, refused 0; with binary32 96, vectors 0, fused 0
    optimized: disagreements 0, agree 120, failed alike 0, refused 0; with binary32 96, vectors 0, fused 0 |}]

let%expect_test "C sweep of vector and numerical plans, slice 0 of 8" =
  let open Ssa_c_sweep_common in
  let neon = Ssa_ir.Ssa_target.forced Ssa_ir.Ssa_target.neon128 in
  List.iter
    (fun (config, numerics) ->
      sweep ~engine:Interpreter ~config
        ~prepare:(planned ~numerics ~target:neon)
        ~shard:0 ~shards:8 ();
      report ~config)
    [
      ("strict-vectors", Ssa_ir.Ssa_numerics.Reference_f64);
      ("ordered-binary32", Ssa_ir.Ssa_numerics.Simd_fp32_ordered);
      ("relaxed-binary32", Ssa_ir.Ssa_numerics.Simd_fp32_relaxed);
    ];
  [%expect
    {|
    strict-vectors: disagreements 0, agree 120, failed alike 0, refused 0; with binary32 88, vectors 10, fused 0
    ordered-binary32: disagreements 0, agree 120, failed alike 0, refused 0; with binary32 90, vectors 10, fused 0
    relaxed-binary32: disagreements 0, agree 120, failed alike 0, refused 0; with binary32 90, vectors 10, fused 0 |}]
