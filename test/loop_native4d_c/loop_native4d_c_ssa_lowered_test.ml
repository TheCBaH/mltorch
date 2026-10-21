let%expect_test
    "lowered Native4D graphs: SSA-made C equals Eval_direct4 bitwise" =
  List.iter
    (C_case.run_case
       ~kernel:(Ssa_backends.c ~pipeline:Ssa_backends.Pipeline.Exact)
       ~label:" (ssa exact)")
    (C_case.Common.lowered () @ [ C_case.Common.deferred_constant () ]);
  [%expect {| |}]
