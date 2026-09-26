open Loop_ir

(* Stage 3 of the Loop IR optimization plan: [Loop_opt_guards] drops a
   [Fail_if (Index_overflows _ | Out_of_range _)] (or an [Or] of only those
   two shapes) that [Loop_range] proves can never fire, given the enclosing
   loop variables' ranges rebuilt top-down over the already-lowered program. *)

let widen (e : Loop_interp.error) = (e :> Kernel_eval.error)

let%expect_test "a provably in-range Out_of_range guard is dropped" =
  let i0 = Loop_fixtures.v 0 in
  let program =
    Loop_fixtures.program
      [
        Loop_stmt.For
          {
            var = i0;
            lo = Loop_index.Const 0;
            hi = Loop_index.Const 4;
            body =
              [
                Loop_stmt.Fail_if
                  ( Loop_bool.Out_of_range (Loop_index.Var i0, 4),
                    Loop_failure.Index_overflow { index = Loop_index.Var i0 } );
              ];
          };
      ]
  in
  Fmt.pr "%a" Loop_pp.program (Loop_opt_guards.run program);
  [%expect {|
    for i0 in [0, 4): |}]

let%expect_test
    "a genuinely out-of-range guard (shifted_loop) is left untouched" =
  let raw = Loop_programs.shifted_loop ~extent:4 in
  let same =
    String.equal
      (Fmt.str "%a" Loop_pp.program raw)
      (Fmt.str "%a" Loop_pp.program (Loop_opt_guards.run raw))
  in
  Fmt.pr "unchanged: %b@." same;
  [%expect {| unchanged: true |}]

(* ---- mutation proof: dropping an unproven guard must turn it red -------- *)

let rec drop_out_of_range_shaped (pred : Loop_expr.pred) =
  match pred with
  | Loop_bool.Out_of_range _ | Loop_bool.Index_overflows _ -> true
  | Loop_bool.Or (a, b) ->
      drop_out_of_range_shaped a && drop_out_of_range_shaped b
  | _ -> false

(* [Loop_opt_guards.run], but without the [Loop_range] proof: every guard of
   the shape it targets is dropped unconditionally, regardless of whether it
   can actually fire. *)
let rec drop_unconditionally (stmts : Loop_stmt.t list) =
  List.concat_map
    (fun (stmt : Loop_stmt.t) ->
      match stmt with
      | Loop_stmt.Fail_if (pred, _) when drop_out_of_range_shaped pred -> []
      | Loop_stmt.For f ->
          [ Loop_stmt.For { f with body = drop_unconditionally f.body } ]
      | Loop_stmt.If (p, a, b) ->
          [ Loop_stmt.If (p, drop_unconditionally a, drop_unconditionally b) ]
      | s -> [ s ])
    stmts

let drop_all_out_of_range : Loop_opt.pass =
 fun program ->
  {
    program with
    Loop_program.body = drop_unconditionally program.Loop_program.body;
  }

let%expect_test
    "stage 3 mutation proof: dropping shifted_loop's guard unconditionally \
     turns it red" =
  let raw = Loop_programs.shifted_loop ~extent:4 in
  let bind = Loop_programs.bind in
  let raw_result = Loop_interp.run raw ~bind in
  let broken = Loop_opt.run ~passes:[ drop_all_out_of_range ] raw in
  let broken_result = Loop_interp.run broken ~bind in
  let reference = Err.map_error widen raw_result in
  Fmt.pr "%a@." Loop_check.pp_verdict
    (Loop_check.compare ~reference ~loop:broken_result);
  [%expect {| DISAGREE: only the reference failed: coord_out_of_range |}]

(* Review regression: an index temporary is NOT always assigned once -- an
   argmax's [best_i] and a max-pool's [best_ix] are seeded before their loop
   and reassigned inside it. Recording only the range of whichever assignment
   the walk saw last proves this guard safe from the seed [0] alone, though
   from the third iteration on [t] holds 1 and the guard fires. *)
let%expect_test "a reassigned index temporary does not prove its guard" =
  let i0 = Loop_fixtures.v 0 and t = Loop_fixtures.temp 0 in
  let out =
    Loop_fixtures.buffer 0 (Loop_fixtures.shape_w 1) Loop_fixtures.f32
      Loop_buffer.Output
  in
  let program =
    Loop_fixtures.program ~buffers:[ out ]
      [
        Loop_stmt.Assign_index (t, Loop_index.Const 0);
        Loop_stmt.For
          {
            var = i0;
            lo = Loop_index.Const 0;
            hi = Loop_index.Const 4;
            body =
              [
                Loop_stmt.Fail_if
                  ( Loop_bool.Out_of_range (Loop_index.Temp t, 1),
                    Loop_failure.Load_out_of_range
                      {
                        buffer = out;
                        coord = Loop_fixtures.at_w (Loop_index.Temp t);
                      } );
                Loop_stmt.Assign_index (t, Loop_index.Var i0);
              ];
          };
      ]
  in
  let bind = Loop_fixtures.bind_none in
  let reference = Err.map_error widen (Loop_interp.run program ~bind) in
  let opt = Loop_interp.run (Loop_opt_guards.run program) ~bind in
  Fmt.pr "%a@." Loop_check.pp_verdict (Loop_check.compare ~reference ~loop:opt);
  [%expect {| agree on failure: coord_out_of_range |}]

(* ---- stage 4: relational window bounds ---------------------------------- *)

(* [for i0 in [0, 8): for k in [max (0, 1 - i0), min (3, hi)): check
   i0 - 1 + k in [0, 8)] -- conv2d's padded window. With [hi = 9 - i0] the
   clamp makes the check dead; with [hi = 10 - i0] (one wider) it fires at
   [i0 = 7, k = 2]. *)
let window ~hi =
  let i0 = Loop_fixtures.v 0 and k = Loop_fixtures.v 1 in
  let buf =
    Loop_fixtures.buffer 0 (Loop_fixtures.shape_w 8) Loop_fixtures.f32
      Loop_buffer.Output
  in
  let coord =
    Loop_index.Add
      ( Loop_index.Add (Loop_index.Var i0, Loop_index.Const (-1)),
        Loop_index.Var k )
  in
  let neg_i0 = Loop_index.Scale (-1, Loop_index.Var i0) in
  Loop_fixtures.program ~buffers:[ buf ]
    [
      Loop_stmt.For
        {
          var = i0;
          lo = Loop_index.Const 0;
          hi = Loop_index.Const 8;
          body =
            [
              Loop_stmt.For
                {
                  var = k;
                  lo =
                    Loop_index.Clamp_low
                      (Loop_index.Add (Loop_index.Const 1, neg_i0));
                  hi =
                    Loop_index.Min
                      ( Loop_index.Const 3,
                        Loop_index.Add (Loop_index.Const hi, neg_i0) );
                  body =
                    [
                      Loop_stmt.Fail_if
                        ( Loop_bool.Out_of_range (coord, 8),
                          Loop_failure.Load_out_of_range
                            { buffer = buf; coord = Loop_fixtures.at_w coord }
                        );
                    ];
                };
            ];
        };
    ]

let guards_left program =
  let rec count (stmts : Loop_stmt.t list) =
    List.fold_left
      (fun n (s : Loop_stmt.t) ->
        match s with
        | Loop_stmt.Fail_if _ -> n + 1
        | Loop_stmt.For { body; _ } -> n + count body
        | Loop_stmt.If (_, a, b) -> n + count a + count b
        | _ -> n)
      0 stmts
  in
  count program.Loop_program.body

let%expect_test "a clamped window's bounds check is proven dead" =
  Fmt.pr "exact window: %d guard(s) left@."
    (guards_left (Loop_opt_guards.run (window ~hi:9)));
  Fmt.pr "one wider: %d guard(s) left@."
    (guards_left (Loop_opt_guards.run (window ~hi:10)));
  [%expect
    {|
    exact window: 0 guard(s) left
    one wider: 1 guard(s) left |}]

let%expect_test
    "stage 4 mutation proof: a loop fact one tighter than the loop turns it red"
    =
  let program = window ~hi:10 in
  let bind = Loop_fixtures.bind_none in
  let reference = Err.map_error widen (Loop_interp.run program ~bind) in
  let run pass = Loop_interp.run (pass program) ~bind in
  Fmt.pr "sound: %a@." Loop_check.pp_verdict
    (Loop_check.compare ~reference ~loop:(run Loop_opt_guards.run));
  Fmt.pr "mutated: %a@." Loop_check.pp_verdict
    (Loop_check.compare ~reference
       ~loop:(run (Loop_opt_guards.with_upper_slack 2)));
  [%expect
    {|
    sound: agree on failure: coord_out_of_range
    mutated: DISAGREE: only the reference failed: coord_out_of_range |}]

(* Dilation 2, padding 2: the access is [i0 - 2 + 2 * k], and the bounds are
   [ceil ((2 - i0) / 2)] and [floor ((9 - i0) / 2) + 1], so the facts are on
   [2 * k]. *)
let%expect_test "a dilated window's bounds check is proven dead" =
  let i0 = Loop_fixtures.v 0 and k = Loop_fixtures.v 1 in
  let buf =
    Loop_fixtures.buffer 0 (Loop_fixtures.shape_w 8) Loop_fixtures.f32
      Loop_buffer.Output
  in
  let open Loop_index in
  let coord = Add (Add (Var i0, Const (-2)), Scale (2, Var k)) in
  let program =
    Loop_fixtures.program ~buffers:[ buf ]
      [
        Loop_stmt.For
          {
            var = i0;
            lo = Const 0;
            hi = Const 8;
            body =
              [
                Loop_stmt.For
                  {
                    var = k;
                    lo =
                      Clamp_low
                        (Ceil_div_pos (Add (Const 2, Scale (-1, Var i0)), 2));
                    hi =
                      Min
                        ( Const 3,
                          Add
                            ( Floor_div_pos
                                (Add (Const 9, Scale (-1, Var i0)), 2),
                              Const 1 ) );
                    body =
                      [
                        Loop_stmt.Fail_if
                          ( Loop_bool.Out_of_range (coord, 8),
                            Loop_failure.Load_out_of_range
                              { buffer = buf; coord = Loop_fixtures.at_w coord }
                          );
                      ];
                  };
              ];
          };
      ]
  in
  Fmt.pr "%d guard(s) left@." (guards_left (Loop_opt_guards.run program));
  [%expect {| 0 guard(s) left |}]
