(* The fast erf against the reference one, over the fast exp's inputs
   ([F32_math_fast_test.compare]). *)

let%expect_test "the fast erf is the reference erf" =
  F32_math_fast_test.compare "erf" F32_math.Value_fast.erf F32_math.Value.erf;
  [%expect {| erf: 5321954 inputs; differing: 0 |}]
