let%expect_test
    "lowered Native4D graphs: compiled C equals Eval_direct4 bitwise" =
  List.iter
    (C_case.run_case ~label:"")
    (C_case.Common.lowered () @ [ C_case.Common.deferred_constant () ]);
  [%expect {| |}]
