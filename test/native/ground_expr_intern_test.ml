(* The intern table keys a constant by its exact bits: equal bits share a node,
   and values that compare equal as floats but differ in bits do not. A NaN
   payload is not tested: js_of_ocaml does not preserve it through
   [Int64.float_of_bits], so the outcome would depend on the backend. *)

let%expect_test "a constant interns by bits" =
  let arena = Ground_expr.Arena.create () in
  let c x = Ground_expr.const arena x in
  let same a b = Ground_expr.equal a b in
  Fmt.pr "1 and 1: %b@." (same (c 1.) (c 1.));
  Fmt.pr "+0 and -0: %b@." (same (c 0.) (c (-0.)));
  Fmt.pr "nan and nan: %b@." (same (c Float.nan) (c Float.nan));
  Fmt.pr "high bits differ only: %b@."
    (same
       (c (Int64.float_of_bits 0x3FF0_0000_0000_0000L))
       (c (Int64.float_of_bits 0x3FF0_0000_0000_0001L)));
  [%expect
    {|
    1 and 1: true
    +0 and -0: false
    nan and nan: true
    high bits differ only: false |}]
