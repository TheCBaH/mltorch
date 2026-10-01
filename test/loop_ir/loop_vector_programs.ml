open Loop_ir
open Loop_fixtures

(* A corpus of loops for the vectorizer's backends: every one is checked by each
   backend that lowers vector loops, bitwise against the interpreter running the
   scalar program. The inputs are chosen so a lane that skipped a rounding, took
   the wrong lane of a load, or lost a NaN's rule shows: values that need more
   than 24 bits after one multiply, signed zeros, NaN, infinities and
   subnormals. *)

let specials =
  [|
    0.;
    -0.;
    1.5;
    -2.25;
    nan;
    infinity;
    neg_infinity;
    5e-324;
    3e38;
    -3e38;
    0.1;
    16777217.;
    1.1;
    3.14159274;
    -0.3333333;
    1e-30;
    123456.789;
  |]

let input_of shape =
  f32_tensor shape (fun c ->
      specials.((Vec6.offset shape c :> int) mod Array.length specials))

let big = 96
let i0 = Loop_index.Var (v 0)
let input = buffer 0 (shape_w big) f32 Loop_buffer.Input
let aux = buffer 2 (shape_w big) f32 Loop_buffer.Input
let output = buffer 1 (shape_w big) f32 Loop_buffer.Output
let ibuf = buffer 4 (shape_w big) (Payload.Fmt Payload.I32) Loop_buffer.Input
let bout = buffer 5 (shape_w big) (Payload.Fmt Payload.Bool) Loop_buffer.Output

let bind id =
  if Tensor_id.equal id (tid 0) || Tensor_id.equal id (tid 2) then
    Some (input_of (shape_w big))
  else if Tensor_id.equal id (tid 4) then (
    let t =
      match Err.payload (Tensor.create_of_sig ibuf.Loop_buffer.sg) with
      | Ok t -> t
      | Error _ -> failwith "i32 tensor"
    in
    Vec6.iter (shape_w big) (fun c ->
        Tensor.set_float t c
          (float_of_int
             (((Vec6.offset (shape_w big) c :> int) * 7919) - 100000)));
    Some t)
  else None

let loop ?(buffers = [ input; aux; output ]) ~n body =
  program ~buffers
    [
      Loop_stmt.For
        { var = v 0; lo = Loop_index.Const 0; hi = Loop_index.Const n; body };
    ]

let store_f32 buffer offset e =
  Loop_stmt.Store_flat
    { buffer; offset; value = Loop_stored.F32 (Loop_expr.Round_f32 e) }

let ld buffer offset = Loop_expr.Load_flat (buffer, offset)
let binary op a b = Loop_expr.Binary (op, a, b)
let add_i a k = Loop_index.Add (a, Loop_index.Const k)
let c = Loop_expr.Const 1.0000001
let rnd e = Loop_expr.Round_f32 e
let t k = Loop_expr.Temp (Loop_carrier.Float, temp k)

let all =
  [
    ( "double rounding",
      loop ~n:37
        [
          store_f32 output i0
            (rnd
               (binary Expr.Value.Mul
                  (rnd (binary Expr.Value.Mul (ld input i0) c))
                  c));
        ] );
    ( "arith chain",
      loop ~n:41
        [
          store_f32 output i0
            (binary Expr.Value.Div
               (binary Expr.Value.Sub
                  (binary Expr.Value.Mul (ld input i0) (ld aux i0))
                  (ld aux i0))
               (binary Expr.Value.Add (ld input i0) (Loop_expr.Const 7.)));
        ] );
    ( "offset view",
      loop ~n:29 [ store_f32 output (add_i i0 5) (ld input (add_i i0 3)) ] );
    ( "broadcast and invariant",
      loop ~n:33
        [
          store_f32 output i0
            (binary Expr.Value.Add
               (binary Expr.Value.Mul
                  (ld input (Loop_index.Const 7))
                  (Loop_expr.Const 3.))
               (ld input i0));
        ] );
    ( "strided",
      loop ~n:20
        [
          store_f32 output
            (Loop_index.Scale (3, i0))
            (binary Expr.Value.Mul
               (ld input (Loop_index.Scale (3, i0)))
               (Loop_expr.Const 2.));
        ] );
    ( "maximum",
      loop ~n:45
        [
          store_f32 output i0
            (Loop_expr.Float_max (ld input i0, ld aux (add_i i0 1)));
        ] );
    ( "select and compare",
      loop ~n:43
        [
          store_f32 output i0
            (Loop_expr.Select
               ( Loop_bool.Value_lt (ld input i0, Loop_expr.Const 0.),
                 Loop_expr.Unary (Expr.Value.Trunc, ld input i0),
                 binary Expr.Value.Div (ld input i0) (Loop_expr.Const 3.) ));
        ] );
    ( "pool_better",
      loop ~n:39
        [
          store_f32 output i0
            (Loop_expr.Select
               ( Loop_bool.Pool_better (ld input i0, ld aux i0),
                 ld input i0,
                 ld aux i0 ));
        ] );
    ( "not and or",
      loop ~n:35
        [
          store_f32 output i0
            (Loop_expr.Select
               ( Loop_bool.Or
                   ( Loop_bool.Not (Loop_bool.Value_eq (ld input i0, ld aux i0)),
                     Loop_bool.Value_lt (ld aux i0, Loop_expr.Const 1.) ),
                 Loop_expr.Const 5.,
                 Loop_expr.Const (-5.) ));
        ] );
    ( "index value",
      loop ~n:27
        [
          store_f32 output i0
            (binary Expr.Value.Mul (Loop_expr.Value_of_index i0) (ld input i0));
        ] );
    ( "temporaries",
      loop ~n:31
        [
          Loop_stmt.Assign
            ( Loop_carrier.Float,
              temp 0,
              binary Expr.Value.Add (ld input i0) (Loop_expr.Const 1.) );
          Loop_stmt.Assign
            (Loop_carrier.Float, temp 1, binary Expr.Value.Mul (t 0) (t 0));
          store_f32 output i0 (binary Expr.Value.Sub (t 1) (t 0));
        ] );
    ( "sqrt and trunc",
      loop ~n:25
        [
          store_f32 output i0
            (binary Expr.Value.Add
               (Loop_expr.Unary
                  ( Expr.Value.Sqrt,
                    Loop_expr.Unary (Expr.Value.Trunc, ld input i0) ))
               (Loop_expr.Const 0.5));
        ] );
    ( "transcendentals",
      loop ~n:21
        [
          store_f32 output i0
            (binary Expr.Value.Add
               (Loop_expr.Unary
                  ( Expr.Value.Exp,
                    binary Expr.Value.Mul (ld input i0) (Loop_expr.Const 0.01)
                  ))
               (Loop_expr.Unary (Expr.Value.Sin, ld aux i0)));
        ] );
    ( "int32 source",
      loop ~buffers:[ input; ibuf; output ] ~n:26
        [
          store_f32 output i0 (binary Expr.Value.Add (ld ibuf i0) (ld input i0));
        ] );
    ( "bool store",
      loop ~buffers:[ input; aux; bout ] ~n:30
        [
          Loop_stmt.Store_flat
            {
              buffer = bout;
              offset = i0;
              value =
                Loop_stored.Bool
                  (binary Expr.Value.Sub (ld input i0) (ld aux i0));
            };
        ] );
    ( "nested loops, vector inner",
      program ~buffers:[ input; aux; output ]
        [
          Loop_stmt.For
            {
              var = v 1;
              lo = Loop_index.Const 0;
              hi = Loop_index.Const 3;
              body =
                [
                  Loop_stmt.For
                    {
                      var = v 0;
                      lo = Loop_index.Const 0;
                      hi = Loop_index.Const 19;
                      body =
                        [
                          store_f32 output
                            (Loop_index.Add
                               (Loop_index.Scale (24, Loop_index.Var (v 1)), i0))
                            (binary Expr.Value.Mul
                               (ld input
                                  (Loop_index.Add
                                     ( Loop_index.Scale
                                         (24, Loop_index.Var (v 1)),
                                       i0 )))
                               (ld aux (Loop_index.Var (v 1))));
                        ];
                    };
                ];
            };
        ] );
    ( "extents around the width: 3",
      loop ~n:3 [ store_f32 output i0 (binary Expr.Value.Mul (ld input i0) c) ]
    );
    ( "extents around the width: 4",
      loop ~n:4 [ store_f32 output i0 (binary Expr.Value.Mul (ld input i0) c) ]
    );
    ( "extents around the width: 5",
      loop ~n:5 [ store_f32 output i0 (binary Expr.Value.Mul (ld input i0) c) ]
    );
  ]
