open Loop_ir
open Loop_fixtures
module V = Loop_vector

(* The vector layer against its oracle: a program is vectorized, expanded lane by
   lane into the scalar program its semantics define, and both are run by the
   interpreter; outputs must be bitwise equal. The scalar original is the
   reference throughout. *)

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
  |]

let input_of shape =
  f32_tensor shape (fun c ->
      specials.((Vec6.offset shape c :> int) mod Array.length specials))

let outputs_equal a b =
  Tensor_id.Map.equal (fun x y -> Loop_check.tensors_equal x y) a b

(* An output the program does not write is whatever the allocator left, so each
   run is given its own zeroed outputs: the comparison is of what is written. *)
let run (p : Loop_program.t) ~bind =
  let outputs id =
    List.find_map
      (fun (b : Loop_buffer.t) ->
        if
          b.Loop_buffer.role = Loop_buffer.Output
          && Tensor_id.equal b.Loop_buffer.id id
        then Some (Loop_interp.allocate b)
        else None)
      p.Loop_program.buffers
  in
  match Err.payload (Loop_interp.run ~outputs p ~bind) with
  | Ok m -> Ok m
  | Error e -> Error (Fmt.str "%a" Loop_interp.pp_error e)

let verdict name (p : Loop_program.t) ~bind =
  let vp, report = Loop_vectorize.program p in
  let tally =
    String.concat ", "
      (List.map
         (fun (k, (n, _)) -> Printf.sprintf "%s=%d" k n)
         (Loop_vectorize.tally report))
  in
  let checked =
    match Err.payload (Loop_vector_check.program vp) with
    | Ok () -> "verified"
    | Error e -> Fmt.str "INVALID %a" Loop_vector_check.pp_error e
  in
  let expanded = Loop_vector_expand.expand vp in
  let same =
    match (run p ~bind, run expanded ~bind) with
    | Ok a, Ok b ->
        if outputs_equal a b then "values equal" else "VALUES DIFFER"
    | Error a, Error b -> if a = b then "same failure" else "FAILURES DIFFER"
    | Ok _, Error _ | Error _, Ok _ -> "ONE FAILED"
  in
  Fmt.pr "%-28s %s; %s; %s@." name
    (if tally = "" then "no loops" else tally)
    checked same

let inp = buffer 0 (shape_w 64) f32 Loop_buffer.Input
let inp_n n = buffer 0 (shape_w (max n 1)) f32 Loop_buffer.Input
let out_n n = buffer 1 (shape_w (max n 1)) f32 Loop_buffer.Output

let bind_for shape id =
  if Tensor_id.equal id (tid 0) then Some (input_of shape) else None

(* out[i] = round_f32 (max (in[i] * 1.5 + 0.25, 0.)) over n elements. *)
let pointwise n =
  let i = Loop_index.Var (v 0) in
  let input = inp_n n and output = out_n n in
  program ~buffers:[ input; output ]
    [
      Loop_stmt.For
        {
          var = v 0;
          lo = Loop_index.Const 0;
          hi = Loop_index.Const n;
          body =
            [
              Loop_stmt.Store_flat
                {
                  buffer = output;
                  offset = i;
                  value =
                    Loop_stored.F32
                      (Loop_expr.Round_f32
                         (Loop_expr.Float_max
                            ( Loop_expr.Binary
                                ( Expr.Value.Add,
                                  Loop_expr.Binary
                                    ( Expr.Value.Mul,
                                      Loop_expr.Load_flat (input, i),
                                      Loop_expr.Const 1.5 ),
                                  Loop_expr.Const 0.25 ),
                              Loop_expr.Const 0. )));
                };
            ];
        };
    ]

let%expect_test "pointwise loops: every extent around the vector width" =
  List.iter
    (fun n ->
      verdict
        (Printf.sprintf "pointwise n=%d" n)
        (pointwise n)
        ~bind:(bind_for (shape_w (max n 1))))
    [ 0; 1; 3; 4; 5; 7; 8; 9; 17; 63 ];
  [%expect
    {|
    pointwise n=0                too_short=1; verified; values equal
    pointwise n=1                too_short=1; verified; values equal
    pointwise n=3                too_short=1; verified; values equal
    pointwise n=4                vectorized=1; verified; values equal
    pointwise n=5                vectorized=1; verified; values equal
    pointwise n=7                vectorized=1; verified; values equal
    pointwise n=8                vectorized=1; verified; values equal
    pointwise n=9                vectorized=1; verified; values equal
    pointwise n=17               vectorized=1; verified; values equal
    pointwise n=63               vectorized=1; verified; values equal |}]

let i0 = Loop_index.Var (v 0)

(* One loop over [n] iterations of [body], with the given buffers. *)
let loop_program ~buffers ~n body =
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

let%expect_test "layouts: offset views, broadcasts, strides, branches" =
  let n = 13 in
  let big = 64 in
  let input = buffer 0 (shape_w big) f32 Loop_buffer.Input in
  let aux = buffer 2 (shape_w big) f32 Loop_buffer.Input in
  let output = buffer 1 (shape_w big) f32 Loop_buffer.Output in
  let bind id =
    if Tensor_id.equal id (tid 0) || Tensor_id.equal id (tid 2) then
      Some (input_of (shape_w big))
    else None
  in
  let show name body =
    verdict name (loop_program ~buffers:[ input; aux; output ] ~n body) ~bind
  in
  show "offset view (in[i+3])"
    [ store_f32 output (add_i i0 5) (ld input (add_i i0 3)) ];
  show "broadcast (in[7] + in[i])"
    [
      store_f32 output i0
        (binary Expr.Value.Add (ld input (Loop_index.Const 7)) (ld input i0));
    ];
  show "strided (stride 3)"
    [
      store_f32 output
        (Loop_index.Scale (3, i0))
        (binary Expr.Value.Mul
           (ld input (Loop_index.Scale (3, i0)))
           (Loop_expr.Const 2.));
    ];
  show "two inputs"
    [ store_f32 output i0 (binary Expr.Value.Sub (ld input i0) (ld aux i0)) ];
  show "select on a compare"
    [
      store_f32 output i0
        (Loop_expr.Select
           ( Loop_bool.Value_lt (ld input i0, Loop_expr.Const 0.),
             Loop_expr.Unary (Expr.Value.Trunc, ld input i0),
             binary Expr.Value.Div (ld input i0) (Loop_expr.Const 3.) ));
    ];
  show "pool_better predicate"
    [
      store_f32 output i0
        (Loop_expr.Select
           ( Loop_bool.Pool_better (ld input i0, ld aux i0),
             ld input i0,
             ld aux i0 ));
    ];
  show "index value"
    [
      store_f32 output i0
        (binary Expr.Value.Mul (Loop_expr.Value_of_index i0) (ld input i0));
    ];
  show "temp reused in the iteration"
    [
      Loop_stmt.Assign
        ( Loop_carrier.Float,
          temp 0,
          binary Expr.Value.Add (ld input i0) (Loop_expr.Const 1.) );
      store_f32 output i0
        (binary Expr.Value.Mul
           (Loop_expr.Temp (Loop_carrier.Float, temp 0))
           (Loop_expr.Temp (Loop_carrier.Float, temp 0)));
    ];
  [%expect
    {|
    offset view (in[i+3])        vectorized=1; verified; values equal
    broadcast (in[7] + in[i])    vectorized=1; verified; values equal
    strided (stride 3)           unprofitable=1; verified; values equal
    two inputs                   vectorized=1; verified; values equal
    select on a compare          vectorized=1; verified; values equal
    pool_better predicate        vectorized=1; verified; values equal
    index value                  vectorized=1; verified; values equal
    temp reused in the iteration vectorized=1; verified; values equal |}]

let%expect_test "refusals carry a structured reason and leave the loop scalar" =
  let n = 12 in
  let input = buffer 0 (shape_w 64) f32 Loop_buffer.Input in
  let scratch = buffer 3 (shape_w 64) f32 Loop_buffer.Scratch in
  let output = buffer 1 (shape_w 64) f32 Loop_buffer.Output in
  let bind id =
    if Tensor_id.equal id (tid 0) then Some (input_of (shape_w 64)) else None
  in
  let show name ?(extra = []) body =
    verdict name
      (program ~buffers:[ input; scratch; output ]
         (extra
         @ [
             Loop_stmt.For
               {
                 var = v 0;
                 lo = Loop_index.Const 0;
                 hi = Loop_index.Const n;
                 body;
               };
           ]))
      ~bind
  in
  let acc = Loop_expr.Temp (Loop_carrier.Float, temp 0) in
  show "reduction (acc += in[i])"
    ~extra:[ Loop_stmt.Assign (Loop_carrier.Float, temp 0, Loop_expr.Const 0.) ]
    [
      Loop_stmt.Assign
        (Loop_carrier.Float, temp 0, binary Expr.Value.Add acc (ld input i0));
    ];
  show "scratch carried (s[i+1] = s[i])"
    [ store_f32 scratch (add_i i0 1) (ld scratch i0) ];
  show "failure site in the body"
    [
      Loop_stmt.Fail_if
        ( Loop_bool.Out_of_range (i0, 64),
          Loop_failure.Local_out_of_range
            {
              local = Expr.Builder.run Expr.Builder.fresh_local;
              index = i0;
              extent = 64;
            } );
      store_f32 output i0 (ld input i0);
    ];
  show "non-affine (in[i*i])"
    [
      store_f32 output i0 (ld input (Loop_index.Min (i0, Loop_index.Const 20)));
    ];
  show "store through a broadcast"
    [ store_f32 output (Loop_index.Const 0) (ld input i0) ];
  show "transcendental"
    [ store_f32 output i0 (Loop_expr.Unary (Expr.Value.Exp, ld input i0)) ];
  [%expect
    {|
    reduction (acc += in[i])     loop_carried=1; verified; values equal
    scratch carried (s[i+1] = s[i]) loop_carried=1; verified; values equal
    failure site in the body     statement:failure check=1; verified; values equal
    non-affine (in[i*i])         non_affine_access=1; verified; values equal
    store through a broadcast    store_through_broadcast=1; verified; values equal
    transcendental               unprofitable=1; verified; values equal |}]

let%expect_test "the oracle catches a dropped rounding and a wrong stride" =
  let n = 16 in
  let input = buffer 0 (shape_w 64) f32 Loop_buffer.Input in
  let output = buffer 1 (shape_w 64) f32 Loop_buffer.Output in
  let bind id =
    if Tensor_id.equal id (tid 0) then Some (input_of (shape_w 64)) else None
  in
  (* Two roundings: the inner one is observable, the one directly under the
     store is not (the store narrows to binary32 itself). *)
  let c = Loop_expr.Const 1.0000001 in
  let p =
    loop_program ~buffers:[ input; output ] ~n
      [
        Loop_stmt.Store_flat
          {
            buffer = output;
            offset = i0;
            value =
              Loop_stored.F32
                (Loop_expr.Round_f32
                   (binary Expr.Value.Mul
                      (Loop_expr.Round_f32
                         (binary Expr.Value.Mul (ld input i0) c))
                      c));
          };
      ]
  in
  let vp, _ = Loop_vectorize.program p in
  let rec drop_inner_round ~top : V.t -> V.t = function
    | V.Round_f32 a when top -> V.Round_f32 (drop_inner_round ~top:false a)
    | V.Round_f32 a -> drop_inner_round ~top:false a
    | V.Binary (op, a, b) ->
        V.Binary
          (op, drop_inner_round ~top:false a, drop_inner_round ~top:false b)
    | e -> e
  in
  let map_body f (vp : V.program) =
    {
      vp with
      V.body =
        List.map
          (function
            | V.Vector l -> V.Vector { l with V.body = List.map f l.V.body }
            | n -> n)
          vp.V.body;
    }
  in
  let report label vp =
    let a = run p ~bind and b = run (Loop_vector_expand.expand vp) ~bind in
    let same =
      match (a, b) with Ok a, Ok b -> outputs_equal a b | _ -> false
    in
    Fmt.pr "%s: expansion %s the scalar program@." label
      (if same then "equals" else "DIFFERS FROM")
  in
  report "as vectorized" vp;
  report "an inner rounding dropped"
    (map_body
       (function
         | V.Store { access; value = V.F32 e } ->
             V.Store { access; value = V.F32 (drop_inner_round ~top:true e) }
         | s -> s)
       vp);
  let wrong_stride =
    map_body
      (function
        | V.Store { access; value } ->
            V.Store { access = { access with V.Access.stride = 2 }; value }
        | s -> s)
      vp
  in
  (match Err.payload (Loop_vector_check.program wrong_stride) with
  | Ok () -> Fmt.pr "wrong stride: ACCEPTED@."
  | Error e ->
      Fmt.pr "wrong stride: rejected: %a@." Loop_vector_check.pp_error e);
  report "a wrong stride, unchecked" wrong_stride;
  [%expect
    {|
    as vectorized: expansion equals the scalar program
    an inner rounding dropped: expansion DIFFERS FROM the scalar program
    wrong stride: rejected: invalid vector program: stride 2, but the loop variable's coefficient is 1
    a wrong stride, unchecked: expansion DIFFERS FROM the scalar program |}]

let%expect_test "every verifier refusal" =
  let input = buffer 0 (shape_w 64) f32 Loop_buffer.Input in
  let output = buffer 1 (shape_w 64) f32 Loop_buffer.Output in
  let access ?(stride = 1) buffer offset =
    { V.Access.buffer; offset; stride }
  in
  let loop ?(lanes = 4) ?(lo = Loop_index.Const 0) ?(hi = Loop_index.Const 16)
      body =
    {
      V.var = v 0;
      lo;
      hi;
      lanes;
      body;
      scalar = Loop_stmt.For { var = v 0; lo; hi; body = [] };
    }
  in
  let show name l =
    let vp =
      {
        V.scalar = program ~buffers:[ input; output ] [];
        body = [ V.Vector l ];
      }
    in
    match Err.payload (Loop_vector_check.program vp) with
    | Ok () -> Fmt.pr "%-30s ok@." name
    | Error e -> Fmt.pr "%-30s %a@." name Loop_vector_check.pp_error e
  in
  let st ?(stride = 1) e =
    V.Store { access = access ~stride output i0; value = V.F32 e }
  in
  show "well formed" (loop [ st (V.Load (access input i0)) ]);
  show "one lane" (loop ~lanes:1 [ st (V.Load (access input i0)) ]);
  show "non-constant bounds" (loop ~hi:i0 [ st (V.Load (access input i0)) ]);
  show "stride mismatch" (loop [ st (V.Load (access ~stride:2 input i0)) ]);
  show "non-affine offset"
    (loop
       [ st (V.Load (access input (Loop_index.Min (i0, Loop_index.Const 5)))) ]);
  show "temp read before assigned" (loop [ st (V.Temp (V.Temp.of_int 0)) ]);
  show "temp assigned twice"
    (loop
       [
         V.Assign (V.Temp.of_int 0, V.Const 1.);
         V.Assign (V.Temp.of_int 0, V.Const 2.);
         st (V.Temp (V.Temp.of_int 0));
       ]);
  show "store through a broadcast"
    (loop
       [
         V.Store
           {
             access = access ~stride:0 output (Loop_index.Const 0);
             value = V.F32 (V.Const 1.);
           };
       ]);
  show "stored and loaded elsewhere"
    (loop
       [
         V.Store
           {
             access = access output i0;
             value = V.F32 (V.Load (access output (add_i i0 1)));
           };
       ]);
  show "splat reads the loop variable"
    (loop [ st (V.Splat (Loop_expr.Value_of_index i0)) ]);
  show "splat loads a stored buffer"
    (loop [ st (V.Splat (Loop_expr.Load_flat (output, Loop_index.Const 0))) ]);
  show "index value step" (loop [ st (V.Index_value { base = i0; step = 2 }) ]);
  [%expect
    {|
    well formed                    ok
    one lane                       invalid vector program: 1 lanes (at least two are needed)
    non-constant bounds            invalid vector program: the loop bounds are not constants
    stride mismatch                invalid vector program: stride 2, but the loop variable's coefficient is 1
    non-affine offset              invalid vector program: an access offset is not affine in the loop variable
    temp read before assigned      invalid vector program: vector temporary v0 is read before it is assigned
    temp assigned twice            ok
    store through a broadcast      invalid vector program: a store with stride zero
    stored and loaded elsewhere    invalid vector program: buffer t1 is stored and loaded at different accesses
    splat reads the loop variable  invalid vector program: a splat reads the loop variable or a loop temporary
    splat loads a stored buffer    invalid vector program: a splat loads from buffer t1, which the loop stores
    index value step               invalid vector program: index value steps by 2, the loop variable's coefficient is 1 |}]

let%expect_test "the oracle over the whole backend corpus, nests included" =
  List.iter
    (fun (name, p) -> verdict name p ~bind:Loop_vector_programs.bind)
    Loop_vector_programs.all;
  [%expect
    {|
    double rounding              vectorized=1; verified; values equal
    arith chain                  vectorized=1; verified; values equal
    offset view                  vectorized=1; verified; values equal
    broadcast and invariant      vectorized=1; verified; values equal
    strided                      unprofitable=1; verified; values equal
    maximum                      vectorized=1; verified; values equal
    select and compare           vectorized=1; verified; values equal
    pool_better                  vectorized=1; verified; values equal
    not and or                   vectorized=1; verified; values equal
    index value                  vectorized=1; verified; values equal
    temporaries                  vectorized=1; verified; values equal
    sqrt and trunc               vectorized=1; verified; values equal
    transcendentals              vectorized=1; verified; values equal
    int32 source                 vectorized=1; verified; values equal
    bool store                   unprofitable=1; verified; values equal
    nested loops, vector inner   vectorized=1; verified; values equal
    matmul: reduction per output vectorized=1; verified; values equal
    reduction with a triangular inner bound vectorized=1; verified; values equal
    uniform index temporary      vectorized=1; verified; values equal
    extents around the width: 3  too_short=1; verified; values equal
    extents around the width: 4  vectorized=1; verified; values equal
    extents around the width: 5  vectorized=1; verified; values equal |}]
