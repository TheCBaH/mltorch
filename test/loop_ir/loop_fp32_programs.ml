open Loop_ir

(* Programs that exist for binary32: the helpers with a float transcription, the
   packed float scratch, and reductions per output at widths that fill the
   binary32 vector (sixteen lanes) and leave a remainder. Shared by the native C
   and the Wasm suites, so both run the same corpus against the same oracle. *)

module P = Loop_vector_programs

let un op e = Loop_expr.Unary (op, e)

(* The shared corpus was written for four binary64 lanes; binary32 plans
   sixteen, so reductions per output are repeated here at widths that fill a
   vector and leave a remainder. [row_stride] is the distance between output
   rows (the row width by default); a smaller one makes the rows overlap. *)
let matmul ?row_stride ~m ~k ~n () =
  let row_stride = Option.value row_stride ~default:n in
  let acc = Loop_fixtures.temp 0 in
  let acc_e = Loop_expr.Temp (Loop_carrier.Float, acc) in
  let mi = Loop_index.Var (Loop_fixtures.v 0)
  and ni = Loop_index.Var (Loop_fixtures.v 1)
  and ki = Loop_index.Var (Loop_fixtures.v 2) in
  let ( +: ) a b = Loop_index.Add (a, b) in
  let term =
    P.binary Expr.Value.Mul
      (P.ld P.input (Loop_index.Scale (k, mi) +: ki))
      (P.ld P.aux (Loop_index.Scale (n, ki) +: ni))
  in
  let loop var hi body =
    Loop_stmt.For
      {
        var = Loop_fixtures.v var;
        lo = Loop_index.Const 0;
        hi = Loop_index.Const hi;
        body;
      }
  in
  Loop_fixtures.program
    ~buffers:[ P.input; P.aux; P.output ]
    [
      loop 0 m
        [
          loop 1 n
            [
              Loop_stmt.Assign (Loop_carrier.Float, acc, Loop_expr.Const 0.);
              loop 2 k
                [
                  Loop_stmt.Assign
                    (Loop_carrier.Float, acc, P.binary Expr.Value.Add acc_e term);
                ];
              P.store_f32 P.output
                (Loop_index.Scale (row_stride, mi) +: ni)
                (Loop_expr.Round_f32 acc_e);
            ];
        ];
    ]

(* Programs that exist for binary32 alone: the helpers with a float transcription
   and the packed float scratch. *)
let extra =
  let x = P.ld P.input P.i0 and y = P.ld P.aux P.i0 in
  let arrays =
    let a1 = Loop_array.of_int 0 and a2 = Loop_array.of_int 1 in
    let fill a n src =
      Loop_stmt.For
        {
          var = Loop_fixtures.v 0;
          lo = Loop_index.Const 0;
          hi = Loop_index.Const n;
          body =
            [
              Loop_stmt.Array_set
                (a, P.i0, P.binary Expr.Value.Mul (P.ld src P.i0) P.c);
            ];
        }
    in
    Loop_fixtures.program
      ~buffers:[ P.input; P.aux; P.output ]
      [
        Loop_stmt.Alloc (a1, Slot.(count_of_extent (extent 5)));
        Loop_stmt.Alloc (a2, Slot.(count_of_extent (extent 3)));
        fill a1 5 P.input;
        fill a2 3 P.aux;
        Loop_stmt.For
          {
            var = Loop_fixtures.v 0;
            lo = Loop_index.Const 0;
            hi = Loop_index.Const 5;
            body =
              [
                P.store_f32 P.output P.i0
                  (P.binary Expr.Value.Add
                     (Loop_expr.Array_get (a1, P.i0))
                     (Loop_expr.Array_get (a1, P.i0)));
              ];
          };
      ]
  in
  [
    ("matvec: 21 outputs, k 4", matmul ~m:1 ~k:4 ~n:21 ());
    ("matmul: 3 rows, 19 outputs, k 4", matmul ~m:3 ~k:4 ~n:19 ());
    ("matvec: exactly one vector", matmul ~m:1 ~k:4 ~n:16 ());
    ( "int64 to float, once rounded",
      (* 2^60 + 2^36 + 1 sits just above the midpoint of two binary32 values:
         through binary64 first it would round the wrong way *)
      P.loop ~n:20
        [
          P.store_f32 P.output P.i0
            (P.binary Expr.Value.Add
               (Loop_expr.I64_to_float
                  (Loop_expr.I64_const
                     Int64.(add (shift_left 1L 60) (add (shift_left 1L 36) 1L))))
               (P.ld P.input P.i0));
        ] );
    ( "index to float",
      P.loop ~n:20
        [
          P.store_f32 P.output P.i0
            (P.binary Expr.Value.Mul
               (Loop_expr.I64_to_float (Loop_expr.I64_of_index P.i0))
               (P.ld P.input P.i0));
        ] );
    ("erf", P.loop ~n:17 [ P.store_f32 P.output P.i0 (un Expr.Value.Erf x) ]);
    ( "log, cos, sqrt",
      P.loop ~n:19
        [
          P.store_f32 P.output P.i0
            (P.binary Expr.Value.Add (un Expr.Value.Log x)
               (P.binary Expr.Value.Mul (un Expr.Value.Cos y)
                  (un Expr.Value.Sqrt (P.binary Expr.Value.Mul x x))));
        ] );
    ("packed float scratch", arrays);
  ]

(* A dot product of the first [k] cells of the two inputs, as the lowering shapes
   a sum: the seed, then a loop with its mark and the accumulate. Sums of
   [k >= 64] terms (four sixteen-lane vectors) are the ones the relaxed policy
   schedules along their own axis; the lengths below cover every branch of the
   schedule: rounds only (96 = 6 vectors over 3 accumulators, no tail), rounds
   and a leftover vector (83), rounds and a tail without a leftover (70), and
   the shortest (64). *)
let dot ~k =
  let acc = Loop_fixtures.temp 0 in
  let acc_e = Loop_expr.Temp (Loop_carrier.Float, acc) in
  let i = Loop_index.Var (Loop_fixtures.v 0) in
  Loop_fixtures.program
    ~buffers:[ P.input; P.aux; P.output ]
    [
      Loop_stmt.Assign (Loop_carrier.Float, acc, Loop_expr.Const 0.);
      Loop_stmt.For
        {
          var = Loop_fixtures.v 0;
          lo = Loop_index.Const 0;
          hi = Loop_index.Const k;
          body =
            [
              Loop_stmt.Mark Loop_mark.Reduction;
              Loop_stmt.Assign
                ( Loop_carrier.Float,
                  acc,
                  P.binary Expr.Value.Add acc_e
                    (P.binary Expr.Value.Mul (P.ld P.input i) (P.ld P.aux i)) );
            ];
        };
      P.store_f32 P.output (Loop_index.Const 0) (Loop_expr.Round_f32 acc_e);
    ]

let dots =
  List.map
    (fun k -> (Printf.sprintf "dot, %d terms" k, dot ~k))
    [ 96; 83; 70; 64 ]

(* Moderate values, a fixed generator in [-1, 1]: long sums stay finite, so a
   different association of the terms shows in the low bits instead of being
   hidden by a NaN or an infinity. *)
let moderate_bind id =
  if
    Tensor_id.equal id (Loop_fixtures.tid 0)
    || Tensor_id.equal id (Loop_fixtures.tid 2)
  then
    let seed = if Tensor_id.equal id (Loop_fixtures.tid 0) then 17 else 29 in
    Some
      (Loop_fixtures.f32_tensor (Loop_fixtures.shape_w P.big) (fun c ->
           let k = (Vec6.offset (Loop_fixtures.shape_w P.big) c :> int) in
           Float.sin (float_of_int ((k * 7919) + seed)) *. 1.37))
  else None

(* [x * y + z] per cell: with moderate values a fused multiply-add and a rounded
   product then sum differ in the last bit of many cells, so this is the program
   that shows a contraction (or an oracle that does not fuse) at once. *)
let multiply_add =
  ( "multiply-add, 40 cells",
    P.loop ~n:40
      [
        P.store_f32 P.output P.i0
          (P.binary Expr.Value.Add
             (P.binary Expr.Value.Mul (P.ld P.input P.i0) (P.ld P.aux P.i0))
             (P.ld P.input (P.add_i P.i0 3)));
      ] )
