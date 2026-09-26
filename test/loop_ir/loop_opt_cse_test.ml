open Loop_ir

(* Stage 8 of the Loop IR optimization plan: [Loop_opt_cse] shares an
   identical [Load (b, coord)] of a never-stored buffer occurring twice
   within one [Store]/[Assign]'s own value expression, through one float
   temp computed once immediately before the statement. *)

let widen (e : Loop_interp.error) = (e :> Kernel_eval.error)

(* [relu]'s own double load is covered by the golden diff in
   loop_js_pointwise_test.ml (`x0 = b0[...]; ... x0 < 0 ? 0 : x0`); this file
   is the mutation proof. *)

(* Two DIFFERENT positions of the same input buffer, added together:
   [output[0] = input[0] + input[1]]. With [Loop_programs.bind]'s data
   ([-0.; 1.5; nan; 3.]), input[0] <> input[1], so merging them into one
   shared load (as an implementation that dropped the coordinate half of
   [same_load]'s equality check would) is detectably wrong. *)
let two_loads_program =
  Loop_fixtures.program
    ~buffers:[ Loop_programs.input; Loop_programs.output ]
    [
      Loop_stmt.Store
        {
          buffer = Loop_programs.output;
          coord = Loop_fixtures.at_w (Loop_index.Const 0);
          value =
            Loop_stored.F32
              (Loop_expr.Binary
                 ( Expr.Value.Add,
                   Loop_expr.Load
                     ( Loop_programs.input,
                       Loop_fixtures.at_w (Loop_index.Const 0) ),
                   Loop_expr.Load
                     ( Loop_programs.input,
                       Loop_fixtures.at_w (Loop_index.Const 1) ) ));
        };
    ]

(* [Loop_opt_cse], but with the coordinate dropped from its own [same_load]
   equality: any two loads of the SAME buffer are treated as one, whatever
   position each actually reads. *)
let merge_any_load_of_buffer : Loop_opt.pass =
 fun program ->
  match program.Loop_program.body with
  | [
   Loop_stmt.Store
     {
       buffer;
       coord;
       value =
         Loop_stored.F32
           (Loop_expr.Binary
              (op, Loop_expr.Load (b1, _), Loop_expr.Load (b2, c2)));
     };
  ]
    when Tensor_id.equal b1.Loop_buffer.id b2.Loop_buffer.id ->
      let t = Loop_temp.of_int 1000 in
      {
        program with
        Loop_program.body =
          [
            Loop_stmt.Assign (Loop_carrier.Float, t, Loop_expr.Load (b1, c2));
            Loop_stmt.Store
              {
                buffer;
                coord;
                value =
                  Loop_stored.F32
                    (Loop_expr.Binary
                       ( op,
                         Loop_expr.Temp (Loop_carrier.Float, t),
                         Loop_expr.Temp (Loop_carrier.Float, t) ));
              };
          ];
      }
  | _ -> program

let%expect_test
    "stage 8 mutation proof: merging two loads at different coordinates turns \
     it red" =
  let bind = Loop_programs.bind in
  let raw_result = Loop_interp.run two_loads_program ~bind in
  let broken =
    Loop_opt.run ~passes:[ merge_any_load_of_buffer ] two_loads_program
  in
  let broken_result = Loop_interp.run broken ~bind in
  let reference = Err.map_error widen raw_result in
  Fmt.pr "%a@." Loop_check.pp_verdict
    (Loop_check.compare ~reference ~loop:broken_result);
  [%expect {| DISAGREE: t1 differs bitwise |}]

(* Review regression: a load that only a [Select] branch (or the right of an
   [Or]) reaches is evaluated only when that branch is taken. Sharing it
   through a temp computed before the statement makes it unconditional, so a
   read the program guarded by its own predicate runs out of range (design
   invariant 4). The lowering keeps a guarded load out of a pure [Select]
   today, but the pass must not depend on that. *)
let%expect_test "a load reached only under a Select branch is not shared" =
  let i0 = Loop_fixtures.v 0 in
  let out8 =
    Loop_fixtures.buffer 1 (Loop_fixtures.shape_w 8) Loop_fixtures.f32
      Loop_buffer.Output
  in
  let load =
    Loop_expr.Load (Loop_programs.input, Loop_fixtures.at_w (Loop_index.Var i0))
  in
  let program =
    Loop_fixtures.program
      ~buffers:[ Loop_programs.input; out8 ]
      [
        Loop_stmt.For
          {
            var = i0;
            lo = Loop_index.Const 0;
            hi = Loop_index.Const 8;
            body =
              [
                Loop_stmt.Store
                  {
                    buffer = out8;
                    coord = Loop_fixtures.at_w (Loop_index.Var i0);
                    value =
                      Loop_stored.F32
                        (Loop_expr.Select
                           ( Loop_bool.Index_lt
                               (Loop_index.Var i0, Loop_index.Const 4),
                             Loop_expr.Binary (Expr.Value.Add, load, load),
                             Loop_expr.Const 0. ));
                  };
              ];
          };
      ]
  in
  let bind = Loop_programs.bind in
  let reference = Err.map_error widen (Loop_interp.run program ~bind) in
  (match Loop_interp.run (Loop_opt_cse.run program) ~bind with
  | opt ->
      Fmt.pr "%a@." Loop_check.pp_verdict
        (Loop_check.compare ~reference ~loop:opt)
  | exception Invalid_argument m -> Fmt.pr "red: raised Invalid_argument %S@." m);
  [%expect {| agree |}]
