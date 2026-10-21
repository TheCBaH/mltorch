let%expect_test
    "every Native4D operation: SSA-made C equals Eval_direct4 bitwise" =
  List.iter
    (C_case.run_case
       ~kernel:(Ssa_backends.c ~pipeline:Ssa_backends.Pipeline.Exact)
       ~label:" (ssa exact)")
    (C_case.Common.per_op ());
  [%expect {| |}]
