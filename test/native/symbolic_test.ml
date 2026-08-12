let s1c n = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:n
let row c = Dim.to_int (Vec6.get c Axis.H)
let col c = Dim.to_int (Vec6.get c Axis.W)
let chan c = Dim.to_int (Vec6.get c Axis.C)
let f32 = Payload.Fmt Payload.F32
let pp_result pp_ok = Core.Pretty.err_result ~ok:pp_ok ~error:Shape_error.pp

let pp_eval_result ppf (vals, ok) =
  Format.fprintf ppf "eval=%s direct==symbolic=%b" vals ok

let pp_ground_result ppf (tensor, ok) =
  Format.fprintf ppf "grounded=%a match=%b" Tensor.pp tensor ok

(* [Symbolic] produces a construction COMPUTATION, so every expression is run
   here, once, at the point it is named -- not at each consumer. Running twice
   would build the expression twice and mint two sets of reducer identities:
   alpha-equivalent and harmless, but it is the trap the adapter's header warns
   about, and a test file should not model it. *)
let build = Expr.Builder.run

(* The symbolic side now evaluates through [Expr.Eval] and the bridge. The
   binding is id-keyed and option-valued to match; [Err.or_raise ~pp_error:] converts at
   this one seam, exactly as [Schedule.ground] does. *)
let eval_expr ~binding e c =
  Err.or_raise ~pp_error:Expr.Eval.pp_error
    (Expr.Eval.value (Expr_bridge.env ~binding)
       ~output:(Expr_bridge.coord_of_vec6 (Vec6.map Dim.to_int c))
       e)

let compare_symbolic out_shape_result ~iter_shape ~eval_direct ~eval_symbolic =
  let open Err.Syntax in
  let* out_shape = out_shape_result in
  let outs = ref [] and ok = ref true in
  (* Bitwise, not [Float.equal]: the latter equates -0. with +0. and any NaN
     with any other, which is precisely what the max-pool tests below have to
     tell apart. See [Tensor.equal_bits]. *)
  let same a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b) in
  Vec6.iter iter_shape (fun c ->
      let d = eval_direct c in
      let s = eval_symbolic c in
      outs := s :: !outs;
      if not (same d s) then ok := false);
  Err.return
    ( String.concat "," (List.map (Printf.sprintf "%g") (List.rev !outs)),
      !ok,
      out_shape )

let conv_axis ?(pad_before = 0) ?pad_after ?(dilation = 1) ~kernel ~stride () :
    Conv.Conv2d.axis_window =
  {
    kernel = Dim.extent kernel;
    stride = Op_config.Pos.of_int stride;
    pad_before = Op_config.Nonneg.of_int pad_before;
    pad_after =
      Op_config.Nonneg.of_int (Option.value pad_after ~default:pad_before);
    dilation = Op_config.Pos.of_int dilation;
  }

let conv_params ?(groups = 1) ?(pad = (0, 0)) ~kernel ~stride ~in_channels () =
  let kh, kw = kernel and sh, sw = stride and ph, pw = pad in
  {
    Conv.Conv2d.h = conv_axis ~pad_before:ph ~kernel:kh ~stride:sh ();
    w = conv_axis ~pad_before:pw ~kernel:kw ~stride:sw ();
    in_channels = Dim.extent in_channels;
    groups = Op_config.Pos.of_int groups;
  }

(* [~binding] is id-keyed and option-valued, matching [Schedule.ground]: a
   missing binding is a value the evaluator can report rather than a [Not_found]
   raised out of a map lookup. *)
let compare_grounded out_shape_result ~binding ~expr ~eval_direct =
  let open Err.Syntax in
  let* out_shape = out_shape_result in
  let direct = Schedule.evaluate out_shape eval_direct in
  let grounded = Schedule.ground out_shape ~binding expr in
  Err.return (grounded, Tensor.equal_bits direct grounded)

let%expect_test "Symbolic: pointwise expr pp" =
  let module S = Symbolic in
  let module R = Pointwise.Relu.Compute (S) in
  let module A = Pointwise.Add.Compute (S) in
  let xs =
    Tensor_sig.create ~id:(Tensor_id.of_int 0) ~name:"x" ~shape:(s1c 3) ~fmt:f32
      ()
  in
  let ys =
    Tensor_sig.create ~id:(Tensor_id.of_int 1) ~name:"y" ~shape:(s1c 3) ~fmt:f32
      ()
  in
  Format.printf "%a@." Expr.Pp.value (build (R.pixel xs Symbolic.out_vec));
  [%expect {| select((t0[N,T,D,H,W,C] < 0), 0, t0[N,T,D,H,W,C]) |}];
  Format.printf "%a@." Expr.Pp.value
    (build (A.pixel ~a_shape:(s1c 3) ~b_shape:(s1c 3) xs ys Symbolic.out_vec));
  (* the broadcast (extent-1) axes are read at index 0 — [load] is strict *)
  [%expect {| (t0[0,0,0,0,0,C] + t1[0,0,0,0,0,C]) |}]

let%expect_test "Symbolic conv: expr structure + eval matches Direct" =
  let module S = Symbolic in
  let module Cd = Conv.Conv2d.Compute (Direct) in
  let module Cs = Conv.Conv2d.Compute (S) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:3 ~c:1 in
  let w_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int ((row c * 3) + col c))
  in
  let weight = Tensor.materialize w_shape (fun _ -> 1.) in
  let bias = Tensor.materialize (s1c 1) (fun _ -> 0.) in
  let xs =
    Tensor_sig.create ~id:(Tensor_id.of_int 0) ~name:"x" ~shape:x_shape ~fmt:f32
      ()
  in
  let ws =
    Tensor_sig.create ~id:(Tensor_id.of_int 1) ~name:"w" ~shape:w_shape ~fmt:f32
      ()
  in
  let bs =
    Tensor_sig.create ~id:(Tensor_id.of_int 2) ~name:"b" ~shape:(s1c 1) ~fmt:f32
      ()
  in
  let p = conv_params ~kernel:(2, 2) ~stride:(1, 1) ~in_channels:1 () in
  let e =
    build
      (Cs.pixel p ~x_shape ~weight_shape:w_shape ~x:xs ~weight:ws ~bias:bs
         Symbolic.out_vec)
  in
  Format.printf "%a@." Expr.Pp.value e;
  [%expect
    {| (sum(r1=0..1: sum(r2=max(0,-1*H)..min(2,3+-1+-1*H+1): sum(r3=max(0,-1*W)..min(2,3+-1+-1*W+1): (t0[N,T,D,H+r2,W+r3,r1] * t1[C,0,0,r2,r3,r1])))) + t2[0,0,0,0,0,C]) |}];
  let binding id =
    if id = xs.id then Some x else if id = ws.id then Some weight else Some bias
  in
  Format.printf "%a@." (pp_result pp_eval_result)
    (compare_symbolic
       (Conv.Conv2d.output_shape ~x_shape ~weight_shape:w_shape p)
       ~iter_shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1)
       ~eval_direct:(Cd.pixel p ~x_shape ~weight_shape:w_shape ~x ~weight ~bias)
       ~eval_symbolic:(fun c -> eval_expr ~binding e c)
    |> Result.map (fun (vals, ok, _) -> (vals, ok)));
  [%expect {| eval=8,12,20,24 direct==symbolic=true |}]

let%expect_test
    "Symbolic max_pool2d: eval matches Direct (negative inputs, padded)" =
  let module S = Symbolic in
  let module Pd = Pool.MaxPool2d.Compute (Direct) in
  let module Ps = Pool.MaxPool2d.Compute (S) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:3 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c ->
        float_of_int (-((row c * 3) + col c) - 1))
  in
  let xs =
    Tensor_sig.create ~id:(Tensor_id.of_int 0) ~name:"x" ~shape:x_shape ~fmt:f32
      ()
  in
  let p =
    {
      Pool.MaxPool2d.kernel =
        Op_config.Hw.{ h = Dim.extent 2; w = Dim.extent 2 };
      stride =
        Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
      pad =
        Op_config.Hw.
          { h = Op_config.Nonneg.of_int 1; w = Op_config.Nonneg.of_int 1 };
    }
  in
  let e = build (Ps.pixel p ~x_shape ~x:xs Symbolic.out_vec) in
  Format.printf "%a@." Expr.Pp.value e;
  [%expect {| max_pool2d_value(t0; k=2x2 s=1x1 p=1x1; out=[N,T,D,H,W,C]) |}];
  let binding id = if id = xs.id then Some x else None in
  Format.printf "%a@." (pp_result pp_eval_result)
    (compare_symbolic (Pool.MaxPool2d.output_shape ~x_shape p)
       ~iter_shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:4 ~w:4 ~c:1)
       ~eval_direct:(Pd.pixel p ~x_shape ~x) ~eval_symbolic:(fun c ->
         eval_expr ~binding e c)
    |> Result.map (fun (vals, ok, _) -> (vals, ok)));
  [%expect
    {| eval=-1,-1,-2,-3,-1,-1,-2,-3,-4,-4,-5,-6,-7,-7,-8,-9 direct==symbolic=true |}]

let%expect_test
    "Symbolic avg_pool2d: eval matches Direct (padded, count_include_pad)" =
  let module S = Symbolic in
  let module Pd = Pool.AvgPool2d.Compute (Direct) in
  let module Ps = Pool.AvgPool2d.Compute (S) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let x = Tensor.materialize x_shape (fun _ -> 1.) in
  let xs =
    Tensor_sig.create ~id:(Tensor_id.of_int 0) ~name:"x" ~shape:x_shape ~fmt:f32
      ()
  in
  let p =
    {
      Pool.AvgPool2d.kernel =
        Op_config.Hw.{ h = Dim.extent 2; w = Dim.extent 2 };
      stride =
        Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
      pad =
        Op_config.Hw.
          { h = Op_config.Nonneg.of_int 1; w = Op_config.Nonneg.of_int 1 };
    }
  in
  let e = build (Ps.pixel p ~x_shape ~x:xs Symbolic.out_vec) in
  let binding id = if id = xs.id then Some x else None in
  Format.printf "%a@." (pp_result pp_eval_result)
    (compare_symbolic (Pool.AvgPool2d.output_shape ~x_shape p)
       ~iter_shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:3 ~c:1)
       ~eval_direct:(Pd.pixel p ~x_shape ~x) ~eval_symbolic:(fun c ->
         eval_expr ~binding e c)
    |> Result.map (fun (vals, ok, _) -> (vals, ok)));
  [%expect
    {| eval=0.25,0.5,0.25,0.5,1,0.5,0.25,0.5,0.25 direct==symbolic=true |}]

let%expect_test "Symbolic linear: eval matches Direct" =
  let module S = Symbolic in
  let module Ld = Linear.Linear.Compute (Direct) in
  let module Ls = Linear.Linear.Compute (S) in
  let x_shape = s1c 3 in
  let weight_shape = Vec6.shape ~n:2 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3 in
  let bias_shape = s1c 2 in
  let x = Tensor.materialize x_shape (fun c -> [| 1.; 2.; 3. |].(chan c)) in
  let weight =
    Tensor.materialize weight_shape (fun c ->
        match (Dim.to_int (Vec6.get c Axis.N), chan c) with
        | 0, 0 -> 1.
        | 1, 1 | 1, 2 -> 1.
        | _ -> 0.)
  in
  let bias =
    Tensor.materialize bias_shape (fun c -> [| 10.; 100. |].(chan c))
  in
  let xs =
    Tensor_sig.create ~id:(Tensor_id.of_int 0) ~name:"x" ~shape:x_shape ~fmt:f32
      ()
  in
  let ws =
    Tensor_sig.create ~id:(Tensor_id.of_int 1) ~name:"w" ~shape:weight_shape
      ~fmt:f32 ()
  in
  let bs =
    Tensor_sig.create ~id:(Tensor_id.of_int 2) ~name:"b" ~shape:bias_shape
      ~fmt:f32 ()
  in
  let p = { Linear.Linear.in_features = Dim.extent 3 } in
  let e = build (Ls.pixel p ~x:xs ~weight:ws ~bias:bs Symbolic.out_vec) in
  let binding id =
    if id = xs.id then Some x else if id = ws.id then Some weight else Some bias
  in
  Format.printf "%a@." (pp_result pp_eval_result)
    (compare_symbolic (Linear.Linear.output_shape p ~x_shape ~weight_shape)
       ~iter_shape:(s1c 2) ~eval_direct:(Ld.pixel p ~x ~weight ~bias)
       ~eval_symbolic:(fun c -> eval_expr ~binding e c)
    |> Result.map (fun (vals, ok, _) -> (vals, ok)));
  [%expect {| eval=11,105 direct==symbolic=true |}]

(* Grounding: the [pixel] expression is built exactly once per op below, then
   [Schedule.ground] is called once per concrete input, against several
   genuinely different inputs — the point being that the expression itself is
   reusable data, not a one-shot evaluation. *)

let%expect_test "Symbolic ground: relu over several different inputs" =
  let module S = Symbolic in
  let module R = Pointwise.Relu.Compute (S) in
  let module Rd = Pointwise.Relu.Compute (Direct) in
  let x_shape = s1c 4 in
  let xs =
    Tensor_sig.create ~id:(Tensor_id.of_int 0) ~name:"x" ~shape:x_shape ~fmt:f32
      ()
  in
  let e = build (R.pixel xs Symbolic.out_vec) in
  let inputs =
    [ [| -2.; -0.5; 1.; 3. |]; [| 0.; 0.; 0.; 0. |]; [| 5.; -5.; 2.; -2. |] ]
  in
  List.iter
    (fun vals ->
      let x = Tensor.materialize x_shape (fun c -> vals.(chan c)) in
      let binding id = if id = xs.id then Some x else None in
      Format.printf "%a@."
        (pp_result pp_ground_result)
        (compare_grounded
           (Pointwise.Relu.output_shape x_shape)
           ~binding ~expr:e ~eval_direct:(Rd.pixel x)))
    inputs;
  [%expect
    {|
    grounded=tensor f32 [C=4] {0, 0, 1, 3} match=true
    grounded=tensor f32 [C=4] {0, 0, 0, 0} match=true
    grounded=tensor f32 [C=4] {5, 0, 2, 0} match=true
    |}]

let%expect_test "Symbolic ground: add over several different input pairs" =
  let module S = Symbolic in
  let module A = Pointwise.Add.Compute (S) in
  let module Ad = Pointwise.Add.Compute (Direct) in
  let x_shape = s1c 3 and y_shape = s1c 3 in
  let xs =
    Tensor_sig.create ~id:(Tensor_id.of_int 0) ~name:"x" ~shape:x_shape ~fmt:f32
      ()
  in
  let ys =
    Tensor_sig.create ~id:(Tensor_id.of_int 1) ~name:"y" ~shape:y_shape ~fmt:f32
      ()
  in
  let e =
    build (A.pixel ~a_shape:x_shape ~b_shape:y_shape xs ys Symbolic.out_vec)
  in
  let cases =
    [
      ([| 1.; 2.; 3. |], [| 10.; 10.; 10. |]);
      ([| 0.; 0.; 0. |], [| -1.; -2.; -3. |]);
      ([| -5.; 5.; -5. |], [| 5.; -5.; 5. |]);
    ]
  in
  List.iter
    (fun (xv, yv) ->
      let x = Tensor.materialize x_shape (fun c -> xv.(chan c)) in
      let y = Tensor.materialize y_shape (fun c -> yv.(chan c)) in
      let binding id =
        if id = xs.id then Some x else if id = ys.id then Some y else None
      in
      Format.printf "%a@."
        (pp_result pp_ground_result)
        (compare_grounded
           (Pointwise.Add.output_shape x_shape y_shape)
           ~binding ~expr:e
           ~eval_direct:(Ad.pixel ~a_shape:x_shape ~b_shape:y_shape x y)))
    cases;
  [%expect
    {|
    grounded=tensor f32 [C=3] {11, 12, 13} match=true
    grounded=tensor f32 [C=3] {-1, -2, -3} match=true
    grounded=tensor f32 [C=3] {0, 0, 0} match=true
    |}]

let%expect_test
    "Symbolic ground: conv2d over several different (x, weight, bias)" =
  let module S = Symbolic in
  let module Cs = Conv.Conv2d.Compute (S) in
  let module Cd = Conv.Conv2d.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:3 ~c:1 in
  let weight_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let xs =
    Tensor_sig.create ~id:(Tensor_id.of_int 0) ~name:"x" ~shape:x_shape ~fmt:f32
      ()
  in
  let ws =
    Tensor_sig.create ~id:(Tensor_id.of_int 1) ~name:"w" ~shape:weight_shape
      ~fmt:f32 ()
  in
  let bs =
    Tensor_sig.create ~id:(Tensor_id.of_int 2) ~name:"b" ~shape:(s1c 1) ~fmt:f32
      ()
  in
  let p = conv_params ~kernel:(2, 2) ~stride:(1, 1) ~in_channels:1 () in
  let e =
    build
      (Cs.pixel p ~x_shape ~weight_shape ~x:xs ~weight:ws ~bias:bs
         Symbolic.out_vec)
  in
  let cases =
    [
      ((fun c -> float_of_int ((row c * 3) + col c)), (fun _ -> 1.), fun _ -> 0.);
      ( (fun c -> float_of_int (-((row c * 3) + col c))),
        (fun _ -> 2.),
        fun _ -> 1. );
      ((fun _ -> 1.), (fun c -> if row c = col c then 1. else 0.), fun _ -> 0.);
    ]
  in
  List.iter
    (fun (fx, fw, fb) ->
      let x = Tensor.materialize x_shape fx in
      let weight = Tensor.materialize weight_shape fw in
      let bias = Tensor.materialize (s1c 1) fb in
      let binding id =
        if id = xs.id then Some x
        else if id = ws.id then Some weight
        else Some bias
      in
      Format.printf "%a@."
        (pp_result pp_ground_result)
        (compare_grounded
           (Conv.Conv2d.output_shape ~x_shape ~weight_shape p)
           ~binding ~expr:e
           ~eval_direct:(Cd.pixel p ~x_shape ~weight_shape ~x ~weight ~bias)))
    cases;
  [%expect
    {|
    grounded=tensor f32 [H=2 W=2 C=1] {8, 12, 20, 24} match=true
    grounded=tensor f32 [H=2 W=2 C=1] {-15, -23, -39, -47} match=true
    grounded=tensor f32 [H=2 W=2 C=1] {2, 2, 2, 2} match=true
    |}]

let%expect_test "Symbolic ground: max_pool2d over several different inputs" =
  let module S = Symbolic in
  let module Ps = Pool.MaxPool2d.Compute (S) in
  let module Pd = Pool.MaxPool2d.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:3 ~c:1 in
  let xs =
    Tensor_sig.create ~id:(Tensor_id.of_int 0) ~name:"x" ~shape:x_shape ~fmt:f32
      ()
  in
  let p =
    {
      Pool.MaxPool2d.kernel =
        Op_config.Hw.{ h = Dim.extent 2; w = Dim.extent 2 };
      stride =
        Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
      pad =
        Op_config.Hw.
          { h = Op_config.Nonneg.of_int 1; w = Op_config.Nonneg.of_int 1 };
    }
  in
  let e = build (Ps.pixel p ~x_shape ~x:xs Symbolic.out_vec) in
  let inputs =
    [
      (fun c -> float_of_int ((row c * 3) + col c));
      (fun c -> float_of_int (-((row c * 3) + col c) - 1));
      (fun c -> if row c = col c then 9. else 0.);
    ]
  in
  List.iter
    (fun fx ->
      let x = Tensor.materialize x_shape fx in
      let binding id = if id = xs.id then Some x else None in
      Format.printf "%a@."
        (pp_result pp_ground_result)
        (compare_grounded
           (Pool.MaxPool2d.output_shape ~x_shape p)
           ~binding ~expr:e ~eval_direct:(Pd.pixel p ~x_shape ~x)))
    inputs;
  [%expect
    {|
    grounded=tensor f32 [H=4 W=4 C=1] {0, 1, 2, 2, 3, 4, 5, 5, ...} match=true
    grounded=tensor f32 [H=4 W=4 C=1] {-1, -1, -2, -3, -1, -1, -2, -3, ...} match=true
    grounded=tensor f32 [H=4 W=4 C=1] {9, 9, 0, 0, 9, 9, 9, 0, ...} match=true
    |}]

let%expect_test "Symbolic ground: avg_pool2d over several different inputs" =
  let module S = Symbolic in
  let module Ps = Pool.AvgPool2d.Compute (S) in
  let module Pd = Pool.AvgPool2d.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:3 ~c:1 in
  let xs =
    Tensor_sig.create ~id:(Tensor_id.of_int 0) ~name:"x" ~shape:x_shape ~fmt:f32
      ()
  in
  let p =
    {
      Pool.AvgPool2d.kernel =
        Op_config.Hw.{ h = Dim.extent 2; w = Dim.extent 2 };
      stride =
        Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
      pad =
        Op_config.Hw.
          { h = Op_config.Nonneg.of_int 1; w = Op_config.Nonneg.of_int 1 };
    }
  in
  let e = build (Ps.pixel p ~x_shape ~x:xs Symbolic.out_vec) in
  let inputs =
    [
      (fun _ -> 1.);
      (fun c -> float_of_int ((row c * 3) + col c));
      (fun c -> if row c = col c then 9. else 0.);
    ]
  in
  List.iter
    (fun fx ->
      let x = Tensor.materialize x_shape fx in
      let binding id = if id = xs.id then Some x else None in
      Format.printf "%a@."
        (pp_result pp_ground_result)
        (compare_grounded
           (Pool.AvgPool2d.output_shape ~x_shape p)
           ~binding ~expr:e ~eval_direct:(Pd.pixel p ~x_shape ~x)))
    inputs;
  [%expect
    {|
    grounded=tensor f32 [H=4 W=4 C=1] {0.25, 0.5, 0.5, 0.25, 0.5, 1, 1, 0.5, ...} match=true
    grounded=tensor f32 [H=4 W=4 C=1] {0, 0.25, 0.75, 0.5, 0.75, 2, 3, 1.75, ...} match=true
    grounded=tensor f32 [H=4 W=4 C=1] {2.25, 2.25, 0, 0, 2.25, 4.5, 2.25, 0, ...} match=true
    |}]

let%expect_test
    "Symbolic ground: linear over several different (x, weight, bias)" =
  let module S = Symbolic in
  let module Ls = Linear.Linear.Compute (S) in
  let module Ld = Linear.Linear.Compute (Direct) in
  let x_shape = s1c 3 in
  let weight_shape = Vec6.shape ~n:2 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3 in
  let bias_shape = s1c 2 in
  let xs =
    Tensor_sig.create ~id:(Tensor_id.of_int 0) ~name:"x" ~shape:x_shape ~fmt:f32
      ()
  in
  let ws =
    Tensor_sig.create ~id:(Tensor_id.of_int 1) ~name:"w" ~shape:weight_shape
      ~fmt:f32 ()
  in
  let bs =
    Tensor_sig.create ~id:(Tensor_id.of_int 2) ~name:"b" ~shape:bias_shape
      ~fmt:f32 ()
  in
  let p = { Linear.Linear.in_features = Dim.extent 3 } in
  let e = build (Ls.pixel p ~x:xs ~weight:ws ~bias:bs Symbolic.out_vec) in
  let select_w c =
    match (Dim.to_int (Vec6.get c Axis.N), chan c) with
    | 0, 0 -> 1.
    | 1, 1 | 1, 2 -> 1.
    | _ -> 0.
  in
  let cases =
    [
      ([| 1.; 2.; 3. |], select_w, [| 10.; 100. |]);
      ([| -1.; -2.; -3. |], select_w, [| 0.; 0. |]);
      ([| 4.; 4.; 4. |], (fun _ -> 0.5), [| 1.; -1. |]);
    ]
  in
  List.iter
    (fun (xv, fw, bv) ->
      let x = Tensor.materialize x_shape (fun c -> xv.(chan c)) in
      let weight = Tensor.materialize weight_shape fw in
      let bias = Tensor.materialize bias_shape (fun c -> bv.(chan c)) in
      let binding id =
        if id = xs.id then Some x
        else if id = ws.id then Some weight
        else Some bias
      in
      Format.printf "%a@."
        (pp_result pp_ground_result)
        (compare_grounded
           (Linear.Linear.output_shape p ~x_shape ~weight_shape)
           ~binding ~expr:e
           ~eval_direct:(Ld.pixel p ~x ~weight ~bias)))
    cases;
  [%expect
    {|
    grounded=tensor f32 [C=2] {11, 105} match=true
    grounded=tensor f32 [C=2] {-1, -5} match=true
    grounded=tensor f32 [C=2] {7, 5} match=true
    |}]

let%expect_test "Symbolic ground: mean over spatial (H,W), several inputs" =
  let module S = Symbolic in
  let module Ms = Reduce.Mean.Compute (S) in
  let module Md = Reduce.Mean.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let xs =
    Tensor_sig.create ~id:(Tensor_id.of_int 0) ~name:"x" ~shape:x_shape ~fmt:f32
      ()
  in
  let p = { Reduce.Mean.dims = [ Axis.H; Axis.W ]; keepdim = true } in
  let e = build (Ms.pixel p ~x_shape ~x:xs Symbolic.out_vec) in
  let inputs =
    [
      (fun c -> [| 1.; 2.; 3.; 4. |].((row c * 2) + col c));
      (fun _ -> 5.);
      (fun c -> if row c = col c then 8. else 0.);
    ]
  in
  List.iter
    (fun fx ->
      let x = Tensor.materialize x_shape fx in
      let binding id = if id = xs.id then Some x else None in
      Format.printf "%a@."
        (pp_result pp_ground_result)
        (compare_grounded
           (Reduce.Mean.output_shape ~x_shape p)
           ~binding ~expr:e ~eval_direct:(Md.pixel p ~x_shape ~x)))
    inputs;
  [%expect
    {|
    grounded=tensor f32 [C=1] {2.5} match=true
    grounded=tensor f32 [C=1] {5} match=true
    grounded=tensor f32 [C=1] {4} match=true |}]

let%expect_test "Symbolic rms_norm over C: expr structure + eval matches Direct"
    =
  let module S = Symbolic in
  let module Rd = Norm.RmsNorm.Compute (Direct) in
  let module Rs = Norm.RmsNorm.Compute (S) in
  let x_shape = s1c 2 in
  let x = Tensor.materialize x_shape (fun c -> [| 1.; 7. |].(chan c)) in
  let weight = Tensor.materialize x_shape (fun _ -> 1.) in
  let xs =
    Tensor_sig.create ~id:(Tensor_id.of_int 0) ~name:"x" ~shape:x_shape ~fmt:f32
      ()
  in
  let ws =
    Tensor_sig.create ~id:(Tensor_id.of_int 1) ~name:"w" ~shape:x_shape ~fmt:f32
      ()
  in
  let p = { Norm.RmsNorm.dims = [ Axis.C ]; eps = 0. } in
  let e = build (Rs.pixel p ~x_shape ~x:xs ~weight:ws Symbolic.out_vec) in
  Format.printf "%a@." Expr.Pp.value e;
  [%expect
    {| ((t0[N,T,D,H,W,C] * (1 / sqrt(((sum(r1=0..2: (t0[N,T,D,H,W,r1] * t0[N,T,D,H,W,r1])) / 2) + 0)))) * t1[0,0,0,0,0,C]) |}];
  let binding id = if id = xs.id then Some x else Some weight in
  Format.printf "%a@." (pp_result pp_eval_result)
    (compare_symbolic (Norm.RmsNorm.output_shape ~x_shape) ~iter_shape:(s1c 2)
       ~eval_direct:(Rd.pixel p ~x_shape ~x ~weight) ~eval_symbolic:(fun c ->
         eval_expr ~binding e c)
    |> Result.map (fun (vals, ok, _) -> (vals, ok)));
  [%expect {| eval=0.2,1.4 direct==symbolic=true |}]

let%expect_test
    "Symbolic ground: bmm over several different (input, mat2) pairs" =
  let module S = Symbolic in
  let module Bs = Matmul.Bmm.Compute (S) in
  let module Bd = Matmul.Bmm.Compute (Direct) in
  (* B=2, 2×3 × 3×2 → 2×2 output per batch item *)
  let input_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:3 in
  let mat2_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:2 in
  let ins =
    Tensor_sig.create ~id:(Tensor_id.of_int 0) ~name:"input" ~shape:input_shape
      ~fmt:f32 ()
  in
  let m2s =
    Tensor_sig.create ~id:(Tensor_id.of_int 1) ~name:"mat2" ~shape:mat2_shape
      ~fmt:f32 ()
  in
  let e = build (Bs.pixel ~input_shape ~input:ins ~mat2:m2s Symbolic.out_vec) in
  let batch h = Dim.to_int (Vec6.get h Axis.H) in
  let cases =
    [
      (* identity-like mat2: output = input columns as two-column projection *)
      ( (fun c ->
          let b = batch c and r = Dim.to_int (Vec6.get c Axis.W) in
          float_of_int ((b * 10) + r + 1)),
        fun c ->
          let k = Dim.to_int (Vec6.get c Axis.W)
          and j = Dim.to_int (Vec6.get c Axis.C) in
          if k = j then 1. else 0. );
      (* zero mat2: output must be all-zero regardless of input *)
      ((fun c -> float_of_int (batch c + 1)), fun _ -> 0.);
      (* all-ones input and mat2: output[W=i,C=j] = inner_dim = 3 for every (b,i,j) *)
      ((fun _ -> 1.), fun _ -> 1.);
    ]
  in
  List.iter
    (fun (fi, fm) ->
      let input = Tensor.materialize input_shape fi in
      let mat2 = Tensor.materialize mat2_shape fm in
      let binding id =
        if id = ins.id then Some input
        else if id = m2s.id then Some mat2
        else None
      in
      Format.printf "%a@."
        (pp_result pp_ground_result)
        (compare_grounded
           (Matmul.Bmm.output_shape ~input_shape ~mat2_shape)
           ~binding ~expr:e
           ~eval_direct:(Bd.pixel ~input_shape ~input ~mat2)))
    cases;
  [%expect
    {|
    grounded=tensor f32 [H=2 W=2 C=2] {1, 1, 2, 2, 11, 11, 12, 12} match=true
    grounded=tensor f32 [H=2 W=2 C=2] {0, 0, 0, 0, 0, 0, 0, 0} match=true
    grounded=tensor f32 [H=2 W=2 C=2] {3, 3, 3, 3, 3, 3, 3, 3} match=true
    |}]

(* ---- max semantics: Direct and Symbolic must agree bit-for-bit -------------

   These are the agreement half of the ATen conformance tests in
   compute_test.ml. They exist because the two interpreters used to pool with
   DIFFERENT predicates ([Float.max] in Direct, a strict [>] in Expr.Eval.value), so
   they disagreed on -0. vs +0. and on NaN. Both now route through [Max_op];
   these lock that in, and would fail again if a site reached for [Float.max]
   directly. *)

let nan1 = Int32.float_of_bits 0x7FC00001l
let nan2 = Int32.float_of_bits 0x7FC00002l

let row_pool n =
  {
    Pool.MaxPool2d.kernel = Op_config.Hw.{ h = Dim.extent 1; w = Dim.extent n };
    stride =
      Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int n };
    pad =
      Op_config.Hw.
        { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
  }

let%expect_test "Symbolic max_pool2d: agrees with Direct on signed zero and NaN"
    =
  let module S = Symbolic in
  let module Pd = Pool.MaxPool2d.Compute (Direct) in
  let module Ps = Pool.MaxPool2d.Compute (S) in
  List.iter
    (fun (name, values) ->
      let n = Array.length values in
      let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:n ~c:1 in
      let x = Tensor.materialize x_shape (fun c -> values.(col c)) in
      let xs =
        Tensor_sig.create ~id:(Tensor_id.of_int 0) ~name:"x" ~shape:x_shape
          ~fmt:f32 ()
      in
      let p = row_pool n in
      let e = build (Ps.pixel p ~x_shape ~x:xs Symbolic.out_vec) in
      let binding id = if id = xs.id then Some x else None in
      Format.printf "%s: %a@." name
        (pp_result (fun ppf (_, ok) -> Format.fprintf ppf "match=%b" ok))
        (compare_grounded
           (Pool.MaxPool2d.output_shape ~x_shape p)
           ~binding ~expr:e ~eval_direct:(Pd.pixel p ~x_shape ~x)))
    [
      ("-0 then +0", [| -0.; 0. |]);
      ("+0 then -0", [| 0.; -0. |]);
      ("two NaN payloads", [| nan1; 5.; nan2; 7. |]);
    ];
  [%expect
    {|
    -0 then +0: match=true
    +0 then -0: match=true
    two NaN payloads: match=true |}]

let%expect_test "Symbolic max_reduce: agrees with Direct on signed zero and NaN"
    =
  (* The generic reduction, which no op reaches today, so it is exercised
     directly. Both sides go through [Expr.Max_op.apply Float_max]; the test asserts
     they AGREE, deliberately not what the answer is — [Float_max] does not
     match ATen's [amax], whose scalar and vector reducers disagree with each
     other, so there is no answer to pin. See max_op.mli. *)
  let module S = Symbolic in
  List.iter
    (fun (name, values) ->
      let n = Array.length values in
      let shape = s1c n in
      let x = Tensor.materialize shape (fun c -> values.(chan c)) in
      let xs =
        Tensor_sig.create ~id:(Tensor_id.of_int 0) ~name:"x" ~shape ~fmt:f32 ()
      in
      let at zero i = Vec6.of_fn (fun a -> if a = Axis.C then i else zero) in
      let direct =
        Direct.max_reduce ~lo:Direct.index_zero ~hi:(Direct.index_const n)
          (fun i -> Direct.load x (at Direct.index_zero i))
      in
      (* Built directly from the reduction combinator rather than through an
         op's [pixel], so it is the one site where the run boundary is not a
         [pixel] call -- the computation still has to be run exactly once. *)
      let e =
        build
          (S.max_reduce ~lo:S.index_zero ~hi:(S.index_const n) (fun i ->
               S.load xs (at S.index_zero i)))
      in
      let symbolic = eval_expr ~binding:(fun _ -> Some x) e Vec6.origin in
      Format.printf "%s: direct==symbolic=%b@." name
        (Int64.equal
           (Int64.bits_of_float direct)
           (Int64.bits_of_float symbolic)))
    [
      ("-0 then +0", [| -0.; 0. |]);
      ("+0 then -0", [| 0.; -0. |]);
      ("two NaN payloads", [| nan1; 5.; nan2; 7. |]);
    ];
  [%expect
    {|
    -0 then +0: direct==symbolic=true
    +0 then -0: direct==symbolic=true
    two NaN payloads: direct==symbolic=true |}]

(* A symbolic value is a construction COMPUTATION, which is what removed the
   supply ref. These two tests pin the consequences, neither of which the rest
   of the suite can see: no current op combines two reduction-bearing operands,
   and none reuses an effectful value. *)
let%expect_test "Symbolic: combining two computations threads the supply" =
  let module S = Symbolic in
  let xs =
    Tensor_sig.create ~id:(Tensor_id.of_int 0) ~name:"x" ~shape:(s1c 4) ~fmt:f32
      ()
  in
  let red () =
    S.sum ~lo:S.index_zero ~hi:(S.index_const 4) (fun i ->
        S.load xs (Vec6.of_fn (fun a -> if a = Axis.C then i else S.index_zero)))
  in
  (* Two independently written reductions under one [add]: [map2] runs the left
     operand, then the right FROM THE STATE IT RETURNED. Threading the incoming
     state into both instead would hand them the same ordinal and collapse them
     onto one variable -- a wrong answer, not a naming clash. *)
  let e = build (S.add (red ()) (red ())) in
  Format.printf "%a@." Expr.Pp.value e;
  [%expect
    {| (sum(r1=0..4: t0[0,0,0,0,0,r1]) + sum(r2=0..4: t0[0,0,0,0,0,r2])) |}];
  let binders = Expr.Fold.binders e in
  Format.printf "binders %d, distinct %d@." (List.length binders)
    (Expr.Reduce_var.Set.cardinal (Expr.Reduce_var.Set.of_list binders));
  [%expect {| binders 2, distinct 2 |}]

let%expect_test "Symbolic: reusing a computation builds it twice" =
  let module S = Symbolic in
  let x = Tensor.materialize (s1c 4) (fun c -> float_of_int (chan c + 1)) in
  let xs =
    Tensor_sig.create ~id:(Tensor_id.of_int 0) ~name:"x" ~shape:(s1c 4) ~fmt:f32
      ()
  in
  let at i = Vec6.of_fn (fun a -> if a = Axis.C then i else S.index_zero) in
  (* Bound ONCE in OCaml, used twice. Under a plain-value semantics this would
     share one node; here each use re-runs the computation, so the expression
     carries two structurally identical reductions with DISTINCT identities.
     Alpha-equivalent, same answer, duplicated work -- and no op may rely on
     physical sharing. *)
  let s =
    S.sum ~lo:S.index_zero ~hi:(S.index_const 4) (fun i -> S.load xs (at i))
  in
  let e = build (S.add s s) in
  let binders = Expr.Fold.binders e in
  Format.printf "reductions %d, distinct identities %d, well-scoped: %a@."
    (List.length binders)
    (Expr.Reduce_var.Set.cardinal (Expr.Reduce_var.Set.of_list binders))
    (Core.Pretty.err_result ~ok:(Fmt.any "yes") ~error:Expr.Check.pp_error)
    (Expr.Check.value e);
  [%expect {| reductions 2, distinct identities 2, well-scoped: yes |}];
  (* And it still denotes the same thing Direct computes. *)
  let direct =
    let d =
      Direct.sum ~lo:Direct.index_zero ~hi:(Direct.index_const 4) (fun i ->
          Direct.load x
            (Vec6.of_fn (fun a -> if a = Axis.C then i else Direct.index_zero)))
    in
    d +. d
  in
  let symbolic = eval_expr ~binding:(fun _ -> Some x) e Vec6.origin in
  Format.printf "direct==symbolic: %b@."
    (Int64.equal (Int64.bits_of_float direct) (Int64.bits_of_float symbolic));
  [%expect {| direct==symbolic: true |}]
