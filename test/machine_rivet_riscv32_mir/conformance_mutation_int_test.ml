(* Integer-form semantic mutations are caught. *)

open Conformance_mutation

let%expect_test "integer semantic mutations are caught" =
  List.iter report_semantic [ div_zero; sltiu_zero_extended ];
  [%expect
    {|
    div by zero as 0: 1 instances disagree
    sltiu immediate zero-extended: 2 instances disagree |}]
