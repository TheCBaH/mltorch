let%expect_test "JavaScript sweep, slice 3 of 4" =
  let open Ssa_js_sweep_common in
  sweep ~config:"lowered" ~prepare:Fun.id ~shard:3 ~shards:4;
  sweep ~config:"optimized" ~prepare:optimize ~shard:3 ~shards:4;
  report ~config:"lowered";
  report ~config:"optimized";
  [%expect
    {|
    lowered: disagreements 0, agree 216, failed alike 0, refused 0
    optimized: disagreements 0, agree 216, failed alike 0, refused 0 |}]
