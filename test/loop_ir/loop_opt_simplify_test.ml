open Loop_ir

(* [Loop_opt_simplify]: range-aware index simplification. The golden diffs in
   loop_js_{dense,pointwise}_test.ml are the positive cases
   (adaptive_avg_pool2d's divisions, the pools' clamps, reshape's cancelling
   terms); this file pins the overflow gate. *)

let widen (e : Loop_interp.error) = (e :> Kernel_eval.error)

(* [floor (4 * i / 4)] is [i] on the integers, but [4 * i] leaves the domain
   for [i = 2^30], and the guard over it must still fire: the rewrite has to
   be refused because the node is not proven, not because the values differ. *)
let overflow_program =
  let i0 = Loop_fixtures.v 0 in
  let idx =
    Loop_index.Floor_div_pos (Loop_index.Scale (4, Loop_index.Var i0), 4)
  in
  let out =
    Loop_fixtures.buffer 0 (Loop_fixtures.shape_w 1) Loop_fixtures.f32
      Loop_buffer.Output
  in
  Loop_fixtures.program ~buffers:[ out ]
    [
      Loop_stmt.For
        {
          var = i0;
          lo = Loop_index.Const 1_073_741_824;
          hi = Loop_index.Const 1_073_741_826;
          body =
            [
              Loop_stmt.Fail_if
                ( Loop_bool.Index_overflows idx,
                  Loop_failure.Index_overflow { index = idx } );
            ];
        };
    ]

let verdict pass =
  let bind = Loop_fixtures.bind_none in
  let reference =
    Err.map_error widen (Loop_interp.run overflow_program ~bind)
  in
  let opt = Loop_interp.run (pass overflow_program) ~bind in
  Fmt.pr "%a@." Loop_check.pp_verdict (Loop_check.compare ~reference ~loop:opt)

let%expect_test "an unproven division is not simplified" =
  verdict Loop_opt_simplify.run;
  [%expect {| agree on failure: index_overflow |}]

let%expect_test "mutation proof: simplifying without the proof turns it red" =
  verdict (Loop_opt_simplify.with_proof (fun _ _ -> true));
  [%expect {| DISAGREE: only the reference failed: index_overflow |}]
