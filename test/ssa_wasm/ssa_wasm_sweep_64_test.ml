let%expect_test "Wasm sweep, slice 64 of 73" =
  let open Ssa_wasm_sweep_common in
  sweep ~config:"lowered" ~prepare:Fun.id ~shard:64 ~shards:73 ();
  sweep ~config:"optimized" ~prepare:optimize ~shard:64 ~shards:73 ();
  report ~config:"lowered";
  report ~config:"optimized";
  [%expect
    {|
    lowered: disagreements 0, agree 12, failed alike 0, refused 0; with vectors 0, relaxed 0
    optimized: disagreements 0, agree 12, failed alike 0, refused 0; with vectors 0, relaxed 0 |}]

let%expect_test "Wasm sweep of vector and numerical plans, slice 64 of 73" =
  let open Ssa_wasm_sweep_common in
  let wasm128 = Ssa_ir.Ssa_target.forced Ssa_ir.Ssa_target.wasm128 in
  let relaxed = Ssa_ir.Ssa_target.forced Ssa_ir.Ssa_target.wasm128_relaxed in
  List.iter
    (fun (config, numerics, target, relaxed) ->
      sweep ~engine:Interpreter ~relaxed ~config
        ~prepare:(planned ~numerics ~target)
        ~shard:64 ~shards:73 ();
      report ~config)
    [
      ("strict-vectors", Ssa_ir.Ssa_numerics.Reference_f64, wasm128, false);
      ("ordered-binary32", Ssa_ir.Ssa_numerics.Simd_fp32_ordered, wasm128, false);
      ("relaxed-binary32", Ssa_ir.Ssa_numerics.Simd_fp32_relaxed, relaxed, true);
    ];
  [%expect
    {|
    strict-vectors: disagreements 0, agree 12, failed alike 0, refused 0; with vectors 12, relaxed 0
    ordered-binary32: disagreements 0, agree 12, failed alike 0, refused 0; with vectors 12, relaxed 0
    relaxed-binary32: disagreements 0, agree 12, failed alike 0, refused 0; with vectors 12, relaxed 0 |}]
