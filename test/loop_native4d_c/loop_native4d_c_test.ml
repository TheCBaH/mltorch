let%expect_test
    "every Native4D operation: compiled C equals Eval_direct4 bitwise" =
  List.iter (C_case.run_case ~label:"") (C_case.Common.per_op ());
  [%expect {| |}]
