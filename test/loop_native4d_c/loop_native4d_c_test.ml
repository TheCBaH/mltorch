let%expect_test "every Native4D case: compiled C equals Eval_direct4 bitwise" =
  List.iter (C_case.run_case ~label:"") (C_case.Common.all ());
  [%expect {| |}]
