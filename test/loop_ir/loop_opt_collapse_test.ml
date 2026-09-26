open Loop_ir
open Loop_fixtures

(* Stage 6: loop collapsing. The per-op goldens are the positive cases (the
   elementwise ops become one loop); this file pins the broadcast refusal and
   is its mutation proof. *)

let widen (e : Loop_interp.error) = (e :> Kernel_eval.error)
let shape_hw = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:1
let at_hw h w = Expr.Coord.set (at_w w) Expr.Axis.H h

(* out[h, w] = in[h, w] + bias[w]: [bias] is broadcast along H, so in its
   offset [h] has coefficient 0 beside [w]'s 1, which is not [3 * 1]. *)
let broadcast_add =
  let input = buffer 0 shape_hw f32 Loop_buffer.Input
  and bias = buffer 1 (shape_w 3) f32 Loop_buffer.Input
  and out = buffer 2 shape_hw f32 Loop_buffer.Output in
  let h = Loop_index.Var (v 0) and w = Loop_index.Var (v 1) in
  program ~buffers:[ input; bias; out ]
    [
      Loop_stmt.For
        {
          var = v 0;
          lo = Loop_index.Const 0;
          hi = Loop_index.Const 2;
          body =
            [
              Loop_stmt.For
                {
                  var = v 1;
                  lo = Loop_index.Const 0;
                  hi = Loop_index.Const 3;
                  body =
                    [
                      Loop_stmt.Store
                        {
                          buffer = out;
                          coord = at_hw h w;
                          value =
                            Loop_stored.F32
                              (Loop_expr.Round_f32
                                 (Loop_expr.Binary
                                    ( Expr.Value.Add,
                                      Loop_expr.Load (input, at_hw h w),
                                      Loop_expr.Load (bias, at_w w) )));
                        };
                    ];
                };
            ];
        };
    ]

let bind id =
  if Tensor_id.equal id (tid 0) then
    Some
      (f32_tensor shape_hw (fun c ->
           float_of_int
             ((3 * Dim.to_int (Vec6.get c Axis.H))
             + Dim.to_int (Vec6.get c Axis.W))))
  else if Tensor_id.equal id (tid 1) then
    Some
      (f32_tensor (shape_w 3) (fun c ->
           100. *. float_of_int (Dim.to_int (Vec6.get c Axis.W))))
  else None

let verdict pass =
  let reference = Err.map_error widen (Loop_interp.run broadcast_add ~bind) in
  match Loop_interp.run (pass broadcast_add) ~bind with
  | opt ->
      Fmt.pr "%a@." Loop_check.pp_verdict
        (Loop_check.compare ~reference ~loop:opt)
  | exception Invalid_argument m -> Fmt.pr "red: raised Invalid_argument %S@." m

let%expect_test "a broadcast axis stays nested" =
  Fmt.pr "%a@." Loop_pp.program (Loop_opt_collapse.run broadcast_add);
  verdict Loop_opt_collapse.run;
  [%expect
    {|
    in t0 f32 [H=2 W=3 C=1]
    in t1 f32 [W=3 C=1]
    out t2 f32 [H=2 W=3 C=1]
    for i0 in [0, 2):
      for i1 in [0, 3):
        store t2[0,0,0,i0,i1,0] = f32(round_f32((load t0[0,0,0,i0,i1,0] + load t1[0,0,0,0,i1,0])))
    agree |}]

let%expect_test
    "stage 6 mutation proof: collapsing across a broadcast axis turns it red" =
  let anything ~c1:_ ~c2:_ ~n_inner:_ = true in
  Fmt.pr "%a@." Loop_pp.program
    (Loop_opt_collapse.with_ratio anything broadcast_add);
  verdict (Loop_opt_collapse.with_ratio anything);
  [%expect
    {|
    in t0 f32 [H=2 W=3 C=1]
    in t1 f32 [W=3 C=1]
    out t2 f32 [H=2 W=3 C=1]
    for i0 in [0, 6):
      store t2[@i0] = f32(round_f32((load t0[@i0] + load t1[@i0])))
    red: raised Invalid_argument "Loop_interp: unchecked access out of range" |}]
