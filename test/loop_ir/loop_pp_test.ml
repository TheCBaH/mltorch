open Loop_ir
open Loop_fixtures

(* Every statement and expression form, in one program, so the printer's whole
   vocabulary is pinned. *)
let everything =
  let input = buffer 0 (shape_w 4) f32 Loop_buffer.Input in
  let output = buffer 1 (shape_w 4) f32 Loop_buffer.Output in
  let i = Loop_index.Var (v 7) in
  let loaded = Loop_expr.Load (input, at_w i) in
  let acc = Loop_expr.Temp (Loop_carrier.Float, temp 5) in
  program ~buffers:[ input; output ]
    [
      Loop_stmt.Alloc (Loop_array.of_int 9, Slot.(count_of_extent (extent 3)));
      Loop_stmt.Assign (Loop_carrier.Float, temp 5, Loop_expr.Const (-0.));
      Loop_stmt.For
        {
          var = v 7;
          lo = Loop_index.Const 0;
          hi = Loop_index.Const 4;
          body =
            [
              Loop_stmt.Mark Loop_mark.Key;
              Loop_stmt.Fail_if
                ( Loop_bool.Out_of_range (i, 4),
                  Loop_failure.Load_out_of_range
                    { buffer = input; coord = at_w i } );
              Loop_stmt.Assign
                ( Loop_carrier.Float,
                  temp 5,
                  Loop_expr.Binary (Expr.Value.Add, acc, loaded) );
              Loop_stmt.If
                ( Loop_bool.Pool_better (acc, loaded),
                  [ Loop_stmt.Assign_index (temp 6, i) ],
                  [
                    Loop_stmt.Array_set
                      ( Loop_array.of_int 9,
                        Loop_index.Scale (2, i),
                        Loop_expr.Float_max (acc, loaded) );
                  ] );
              Loop_stmt.Store
                {
                  buffer = output;
                  coord = at_w i;
                  value =
                    Loop_stored.F32
                      (Loop_expr.Select
                         ( Loop_bool.Value_lt (loaded, Loop_expr.Const nan),
                           Loop_expr.Round_f32 acc,
                           Loop_expr.Unary (Expr.Value.Exp, loaded) ));
                };
            ];
        };
    ]

let%expect_test "the printer covers every form" =
  Fmt.pr "%a@." Loop_pp.program everything;
  [%expect
    {|
    in t0 f32 [W=4 C=1]
    out t1 f32 [W=4 C=1]
    alloc a0 : float64[3]
    x0 = -0.
    for i0 in [0, 4):
      mark key
      fail_if out_of_range(i0, 4) -> load_out_of_range(t0[0,0,0,0,i0,0])
      x0 = (x0 + load t0[0,0,0,0,i0,0])
      if pool_better(x0, load t0[0,0,0,0,i0,0]):
        x1 = i0
      else:
        a0[(2 * i0)] = float_max(x0, load t0[0,0,0,0,i0,0])
      store t1[0,0,0,0,i0,0] = f32((load t0[0,0,0,0,i0,0] < nan ? round_f32(x0) : exp(load t0[0,0,0,0,i0,0]))) |}]

(* Names come from position of first appearance, never from allocation ids: the
   same program built from different id numbers prints identically. *)
let%expect_test "printing is independent of allocation ids" =
  let renumber =
    let i = Loop_index.Var (v 100) in
    program
      [
        Loop_stmt.For
          {
            var = v 100;
            lo = Loop_index.Const 0;
            hi = Loop_index.Const 2;
            body =
              [
                Loop_stmt.Assign
                  (Loop_carrier.Float, temp 42, Loop_expr.Value_of_index i);
              ];
          };
      ]
  in
  let plain =
    let i = Loop_index.Var (v 0) in
    program
      [
        Loop_stmt.For
          {
            var = v 0;
            lo = Loop_index.Const 0;
            hi = Loop_index.Const 2;
            body =
              [
                Loop_stmt.Assign
                  (Loop_carrier.Float, temp 0, Loop_expr.Value_of_index i);
              ];
          };
      ]
  in
  let render p = Fmt.str "%a" Loop_pp.program p in
  Fmt.pr "%b@." (String.equal (render renumber) (render plain));
  Fmt.pr "%b@." (String.equal (render everything) (render everything));
  [%expect {|
    true
    true |}]
