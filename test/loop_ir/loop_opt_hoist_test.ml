open Loop_ir
open Loop_fixtures

(* Stage 7: loop-invariant hoisting. [sdpa]'s per-row scale is the golden
   diff; this file replays the bug the first attempt had -- hoisting an
   accumulator's reset -- as the mutation proof of the "no other write"
   condition. *)

let widen (e : Loop_interp.error) = (e :> Kernel_eval.error)

(* for i in [0, 2): x = 0; for k in [0, 2): x = x + in[k]; out[i] = x.
   [x = 0] is closed, but [x] is written again in the loop: hoisted, the
   second row accumulates on top of the first. *)
let accumulate =
  let x = temp 0 and i = v 0 and k = v 1 in
  let xe = Loop_expr.Temp (Loop_carrier.Float, x) in
  program
    ~buffers:[ Loop_programs.input; Loop_programs.output ]
    [
      Loop_stmt.For
        {
          var = i;
          lo = Loop_index.Const 0;
          hi = Loop_index.Const 2;
          body =
            [
              Loop_stmt.Assign (Loop_carrier.Float, x, Loop_expr.Const 0.);
              Loop_stmt.For
                {
                  var = k;
                  lo = Loop_index.Const 0;
                  hi = Loop_index.Const 2;
                  body =
                    [
                      Loop_stmt.Assign
                        ( Loop_carrier.Float,
                          x,
                          Loop_expr.Binary
                            ( Expr.Value.Add,
                              xe,
                              Loop_expr.Load
                                (Loop_programs.input, at_w (Loop_index.Var k))
                            ) );
                    ];
                };
              Loop_stmt.Store
                {
                  buffer = Loop_programs.output;
                  coord = at_w (Loop_index.Var i);
                  value = Loop_stored.F32 (Loop_expr.Round_f32 xe);
                };
            ];
        };
    ]

let verdict pass =
  let bind = Loop_programs.bind in
  let reference = Err.map_error widen (Loop_interp.run accumulate ~bind) in
  let opt = Loop_interp.run (pass accumulate) ~bind in
  Fmt.pr "%a@." Loop_check.pp_verdict (Loop_check.compare ~reference ~loop:opt)

let%expect_test "an accumulator's reset stays in its loop" =
  let same =
    String.equal
      (Fmt.str "%a" Loop_pp.program accumulate)
      (Fmt.str "%a" Loop_pp.program (Loop_opt_hoist.run accumulate))
  in
  Fmt.pr "unchanged: %b@." same;
  verdict Loop_opt_hoist.run;
  [%expect {|
    unchanged: true
    agree |}]

let%expect_test
    "stage 7 mutation proof: hoisting past another write turns it red" =
  verdict (Loop_opt_hoist.with_write_check false);
  [%expect {| DISAGREE: t1 differs bitwise |}]

(* A read before the statement sees, on the first iteration, what the loop
   was entered with: [out[i] = a[0]; a[0] = 1] must not hoist. *)
let%expect_test "a read before the statement keeps it in the loop" =
  let a = Loop_array.of_int 0 and i = v 0 in
  let p =
    program ~buffers:[ Loop_programs.output ]
      [
        Loop_stmt.Alloc (a, Slot.(count_of_extent (extent 1)));
        Loop_stmt.For
          {
            var = i;
            lo = Loop_index.Const 0;
            hi = Loop_index.Const 2;
            body =
              [
                Loop_stmt.Store
                  {
                    buffer = Loop_programs.output;
                    coord = at_w (Loop_index.Var i);
                    value =
                      Loop_stored.F32
                        (Loop_expr.Array_get (a, Loop_index.Const 0));
                  };
                Loop_stmt.Array_set (a, Loop_index.Const 0, Loop_expr.Const 1.);
              ];
          };
      ]
  in
  Fmt.pr "%a@." Loop_pp.program (Loop_opt_hoist.run p);
  [%expect
    {|
    out t1 f32 [W=4 C=1]
    alloc a0 : float64[1]
    for i0 in [0, 2):
      store t1[0,0,0,0,i0,0] = f32(a0[0])
      a0[0] = 1 |}]
