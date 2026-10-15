let%expect_test "C sweep, slice 5 of 8" =
  let open Ssa_c_sweep_common in
  sweep ~config:"lowered" ~prepare:Fun.id ~shard:5 ~shards:8 ();
  sweep ~config:"optimized" ~prepare:optimize ~shard:5 ~shards:8 ();
  report ~config:"lowered";
  report ~config:"optimized";
  [%expect
    {|
    lowered: disagreements 0, agree 108, failed alike 0, refused 0; with binary32 88, vectors 0, fused 0
    optimized: disagreements 0, agree 108, failed alike 0, refused 0; with binary32 88, vectors 0, fused 0 |}]
