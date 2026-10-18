(* Mapping mutations are caught. *)

open Conformance_mutation

let%expect_test "mapping mutations are caught" =
  List.iter report_mapping [ commuted_sub; select_arms; branch_sense ];
  [%expect
    {|
    map commuted sub: 1 instances disagree
    map select arms: 7 instances disagree
    map branch sense: 8 instances disagree |}]
