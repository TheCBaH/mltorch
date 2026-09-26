open Loop_ir

(* Stage 1 of the Loop IR optimization plan. An empty loop whose bounds are 2^32 - 1 apart
   the wrong way round. Its [hi - lo] is [1] in a 32-bit [int] (js_of_ocaml),
   so it must be compared in [int64] or the loop is "unrolled" into one
   iteration that never ran. Meaningful under node; natively it is a plain
   regression check. *)
let%expect_test "an empty loop is never taken for a unit loop" =
  let program =
    Loop_fixtures.program
      [
        Loop_stmt.For
          {
            var = Loop_fixtures.v 0;
            lo = Loop_index.Const 2147483647;
            hi = Loop_index.Const (-2147483648);
            body = [ Loop_stmt.Mark Loop_mark.Reduction ];
          };
      ]
  in
  Fmt.pr "%a" Loop_pp.program (Loop_opt_unit_loops.run program);
  [%expect {|
    for i0 in [2147483647, -2147483648):
      mark reduction |}]
