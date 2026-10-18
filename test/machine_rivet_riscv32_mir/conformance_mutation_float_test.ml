(* Float-form semantic mutations are caught. *)

open Conformance_mutation

let%expect_test "float semantic mutations are caught" =
  List.iter report_semantic [ fmax_ieee; fcvt_nan_zero; nan_not_canonical ];
  [%expect
    {|
    fmax.s as IEEE maximum: 1 instances disagree
    fcvt.w.s of a NaN as 0: 1 instances disagree
    a NaN result not canonical: 1 instances disagree |}]
