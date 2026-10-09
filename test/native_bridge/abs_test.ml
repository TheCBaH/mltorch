(* [torch.ops.aten.abs.default] against real ATen: float32 with signed zero,
   infinities and NaN; int64 exactly, including values past 2^53 and
   [min_int], where ATen's two's-complement [abs] wraps. *)

open Helpers

let%expect_test "verify: abs.default on float32" =
  let x =
    float_tensor [ 2; 4 ] [ -1.5; -0.; 0.; 0.5; 3.25; -2e-38; -8.; 1e20 ]
  in
  verify_print ~target:"torch.ops.aten.abs.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self" ];
  [%expect {| aten and native agree |}]

let%expect_test "verify: abs.default on int64, past 2^53 and at min_int" =
  let x =
    i64_tensor [ 6 ]
      [ -3L; 0L; 7L; -9_007_199_254_740_993L; Int64.max_int; Int64.min_int ]
  in
  verify_print ~target:"torch.ops.aten.abs.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self" ];
  [%expect {| aten and native agree |}]

let%expect_test "dispatch: abs.default builds one Abs node and keeps the dtype"
    =
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.abs.default"
    ~bindings:[ ("self", i64_tensor [ 3 ] [ -4L; 0L; 5L ]) ]
    ~inputs:[ in_tensor "self" ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 i64 [C=3] ->[n0]]
    nodes:
      n0: [t1 i64 [C=3]] = abs x=t0
    outputs: [t1 i64 [C=3] <-n0]
    tensor i64 [C=3] {4, 0, 5} |}]
