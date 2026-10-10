(* Independent real C++ ATen oracle, including the exact integer prepend
   used by Whisper. No Python torch/reference generation is involved. *)
open Helpers

let target = "torch.ops.aten.diff.default"

let%expect_test "diff: defaults, negative/positive axes and prepend/append" =
  let x = float_tensor [ 2; 3 ] [ 1.; 4.; 9.; 2.; 5.; 11. ] in
  let p = float_tensor [ 2; 2 ] [ -3.; 0.; -4.; 1. ] in
  let a = float_tensor [ 2; 1 ] [ 20.; 30. ] in
  verify_print ~target ~bindings:[ ("self", x) ] ~inputs:[ in_tensor "self" ];
  verify_print ~target
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim" 0 ];
  verify_print ~target
    ~bindings:[ ("self", x); ("prepend", p); ("append", a) ]
    ~inputs:
      [
        in_tensor "self";
        in_int "dim" 1;
        in_tensor "prepend";
        in_tensor "append";
      ];
  verify_print ~target
    ~bindings:[ ("self", x); ("prepend", p); ("append", a) ]
    ~inputs:
      [
        in_tensor "self"; in_int "n" 0; in_tensor "prepend"; in_tensor "append";
      ];
  verify_print ~target
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_none "prepend"; in_none "append" ];
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree
    aten and native agree
    aten and native agree |}]

let%expect_test "diff: int64 above 2^53, wrapping and prepend" =
  let x =
    i64_tensor [ 1; 4 ]
      [
        9_007_199_254_740_993L;
        9_007_199_254_740_994L;
        Int64.max_int;
        Int64.min_int;
      ]
  in
  let p = i64_tensor [ 1; 1 ] [ 9_007_199_254_740_992L ] in
  let a = i64_tensor [ 1; 1 ] [ 0L ] in
  verify_print ~target ~bindings:[ ("self", x) ] ~inputs:[ in_tensor "self" ];
  verify_print ~target
    ~bindings:[ ("self", x); ("prepend", p); ("append", a) ]
    ~inputs:
      [
        in_tensor "self";
        in_int "n" 1;
        in_int "dim" (-1);
        in_tensor "prepend";
        in_tensor "append";
      ];
  verify_print ~target
    ~bindings:[ ("self", x); ("prepend", p) ]
    ~inputs:[ in_tensor "self"; in_tensor "prepend" ];
  dispatch_print_with_graph ~print_graph:true ~target
    ~bindings:[ ("self", x); ("prepend", p) ]
    ~inputs:[ in_tensor "self"; in_tensor "prepend" ]
    ~noutputs:1;
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree
    graph
    inputs: [t0 i64 [C=1] ->[n0], t1 i64 [C=4] ->[n0]]
    nodes:
      n0: [t2 i64 [C=4]] = diff x=t1 axis=C order=1 prepend=t0 append=
    outputs: [t2 i64 [C=4] <-n0]
    tensor i64 [C=4] {1, 1, 9214364837600034813, 1} |}]

let%expect_test "diff: each axis of a rank-six tensor" =
  let x = float_tensor [ 2; 2; 2; 2; 2; 2 ] (ramp 64) in
  List.iter
    (fun dim ->
      verify_print ~target
        ~bindings:[ ("self", x) ]
        ~inputs:[ in_tensor "self"; in_int "dim" dim ])
    [ 0; 1; 2; 3; 4; 5 ];
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree
    aten and native agree
    aten and native agree
    aten and native agree |}]

let%expect_test "diff: zero order ignores mixed-dtype optional values" =
  verify_print ~target
    ~bindings:
      [
        ("self", float_tensor [ 3 ] [ 1.; 4.; 9. ]);
        ("prepend", i64_tensor [ 2 ] [ Int64.min_int; Int64.max_int ]);
      ]
    ~inputs:[ in_tensor "self"; in_int "n" 0; in_tensor "prepend" ];
  [%expect {| aten and native agree |}]

let%expect_test "diff: IEEE edge cases and append on a singleton" =
  verify_print ~target
    ~bindings:
      [
        ( "self",
          float_tensor [ 6 ] [ neg_infinity; infinity; nan; -0.; 0.; 1e-45 ] );
      ]
    ~inputs:[ in_tensor "self" ];
  verify_print ~target
    ~bindings:
      [
        ("self", float_tensor [ 1 ] [ 1. ]);
        ("append", float_tensor [ 1 ] [ 3. ]);
      ]
    ~inputs:[ in_tensor "self"; in_tensor "append" ];
  [%expect {|
    aten and native agree
    aten and native agree |}]

let%expect_test "diff: explicit refusals and malformed optional operands" =
  let x = float_tensor [ 2; 3 ] [ 1.; 4.; 9.; 2.; 5.; 11. ] in
  let run bindings inputs =
    dispatch_print ~target ~bindings ~inputs ~noutputs:1
  in
  List.iter
    (fun n -> run [ ("self", x) ] [ in_tensor "self"; in_int "n" n ])
    [ -1; 2 ];
  run [ ("self", x) ] [ in_tensor "self"; in_int "dim" 2 ];
  run [ ("self", float_tensor [ 1 ] [ 1. ]) ] [ in_tensor "self" ];
  run [ ("self", x) ] [ in_tensor "self"; in_int "prepend" 3 ];
  run
    [ ("self", x); ("prepend", float_tensor [ 3 ] [ 0.; 0.; 0. ]) ]
    [ in_tensor "self"; in_tensor "prepend" ];
  run
    [ ("self", x); ("prepend", i64_tensor [ 2; 1 ] [ 0L; 0L ]) ]
    [ in_tensor "self"; in_tensor "prepend" ];
  run [ ("self", double_tensor [ 3 ] [ 1.; 4.; 9. ]) ] [ in_tensor "self" ];
  [%expect
    {|
    error: diff: supported orders are 0 and 1, got -1
    error: diff: supported orders are 0 and 1, got 2
    error: diff.default: invalid dimension 2 for rank 2
    error: diff: empty results are not represented
    error: argument "prepend": expected tensor?, got Int
    error: prepend must be rank-2, got rank-1
    error: diff: mixed dtype i64
    error: diff: unsupported dtype f64 |}]
