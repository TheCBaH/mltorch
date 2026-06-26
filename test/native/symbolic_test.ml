let s1c n = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:n
let row c = Dim.to_int (Vec6.get c Axis.H)
let col c = Dim.to_int (Vec6.get c Axis.W)
let chan c = Dim.to_int (Vec6.get c Axis.C)
let f32 = Payload.Fmt Payload.F32

let tensors_match shape a b =
  let ok = ref true in
  Vec6.iter shape (fun c ->
      if not (Float.equal (Tensor.read a c) (Tensor.read b c)) then ok := false);
  !ok

let%expect_test "Symbolic: pointwise expr pp" =
  let module S = Symbolic.Make () in
  let module R = Pointwise.Relu.Compute (S) in
  let module A = Pointwise.Add.Compute (S) in
  let xs = Tensor_sig.create ~name:"x" ~shape:(s1c 3) ~fmt:f32 () in
  let ys = Tensor_sig.create ~name:"y" ~shape:(s1c 3) ~fmt:f32 () in
  Format.printf "%a@." Expr.pp (R.pixel xs Symbolic.out_coord);
  [%expect {| select((x[N,T,D,H,W,C] < 0), 0, x[N,T,D,H,W,C]) |}];
  Format.printf "%a@." Expr.pp (A.pixel xs ys Symbolic.out_coord);
  [%expect {| (x[N,T,D,H,W,C] + y[N,T,D,H,W,C]) |}]

let%expect_test "Symbolic conv: expr structure + eval matches Direct" =
  let module S = Symbolic.Make () in
  let module Cd = Conv.Conv2d.Compute (Direct) in
  let module Cs = Conv.Conv2d.Compute (S) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:3 ~c:1 in
  let w_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int ((row c * 3) + col c))
  in
  let weight = Tensor.materialize w_shape (fun _ -> 1.) in
  let bias = Tensor.materialize (s1c 1) (fun _ -> 0.) in
  let xs = Tensor_sig.create ~name:"x" ~shape:x_shape ~fmt:f32 () in
  let ws = Tensor_sig.create ~name:"w" ~shape:w_shape ~fmt:f32 () in
  let bs = Tensor_sig.create ~name:"b" ~shape:(s1c 1) ~fmt:f32 () in
  let p =
    {
      Conv.Conv2d.kernel = Op_config.Hw.{ h = Dim.extent 2; w = Dim.extent 2 };
      in_channels = Dim.extent 1;
      stride =
        Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
      pad =
        Op_config.Hw.
          { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
    }
  in
  let out_shape = Conv.Conv2d.output_shape ~x_shape ~weight_shape:w_shape p in
  let e = Cs.pixel p ~x_shape ~x:xs ~weight:ws ~bias:bs Symbolic.out_coord in
  Format.printf "%a@." Expr.pp e;
  [%expect
    {| (sum(r1=0..1: sum(r2=max(0,0+-1*H)..min(2,3+0+-1*H): sum(r3=max(0,0+-1*W)..min(2,3+0+-1*W): (x[N,T,D,1*H+r2+0,1*W+r3+0,r1] * w[C,0,0,r2,r3,r1])))) + b[0,0,0,0,0,C]) |}];
  let binding (s : Tensor_sig.t) =
    if s.id = xs.id then x else if s.id = ws.id then weight else bias
  in
  let outs = ref [] and ok = ref true in
  Vec6.iter out_shape (fun c ->
      let d = Cd.pixel p ~x_shape ~x ~weight ~bias (Schedule.coord_index c) in
      let s = Expr.eval ~binding ~coord:(Schedule.coord_index c) e in
      outs := s :: !outs;
      if not (Float.equal d s) then ok := false);
  Format.printf "eval=%s direct==symbolic=%b@."
    (String.concat "," (List.map (Printf.sprintf "%g") (List.rev !outs)))
    !ok;
  [%expect {| eval=8,12,20,24 direct==symbolic=true |}]

let%expect_test
    "Symbolic max_pool2d: eval matches Direct (negative inputs, padded)" =
  let module S = Symbolic.Make () in
  let module Pd = Pool.MaxPool2d.Compute (Direct) in
  let module Ps = Pool.MaxPool2d.Compute (S) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:3 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c ->
        float_of_int (-((row c * 3) + col c) - 1))
  in
  let xs = Tensor_sig.create ~name:"x" ~shape:x_shape ~fmt:f32 () in
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
  let out_shape = Pool.MaxPool2d.output_shape ~x_shape p in
  let e = Ps.pixel p ~x_shape ~x:xs Symbolic.out_coord in
  let binding (s : Tensor_sig.t) = if s.id = xs.id then x else assert false in
  let outs = ref [] and ok = ref true in
  Vec6.iter out_shape (fun c ->
      let d = Pd.pixel p ~x_shape ~x (Schedule.coord_index c) in
      let s = Expr.eval ~binding ~coord:(Schedule.coord_index c) e in
      outs := s :: !outs;
      if not (Float.equal d s) then ok := false);
  Format.printf "eval=%s direct==symbolic=%b@."
    (String.concat "," (List.map (Printf.sprintf "%g") (List.rev !outs)))
    !ok;
  [%expect
    {| eval=-1,-1,-2,-3,-1,-1,-2,-3,-4,-4,-5,-6,-7,-7,-8,-9 direct==symbolic=true |}]

let%expect_test
    "Symbolic avg_pool2d: eval matches Direct (padded, count_include_pad)" =
  let module S = Symbolic.Make () in
  let module Pd = Pool.AvgPool2d.Compute (Direct) in
  let module Ps = Pool.AvgPool2d.Compute (S) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let x = Tensor.materialize x_shape (fun _ -> 1.) in
  let xs = Tensor_sig.create ~name:"x" ~shape:x_shape ~fmt:f32 () in
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
  let out_shape = Pool.AvgPool2d.output_shape ~x_shape p in
  let e = Ps.pixel p ~x_shape ~x:xs Symbolic.out_coord in
  let binding (s : Tensor_sig.t) = if s.id = xs.id then x else assert false in
  let outs = ref [] and ok = ref true in
  Vec6.iter out_shape (fun c ->
      let d = Pd.pixel p ~x_shape ~x (Schedule.coord_index c) in
      let s = Expr.eval ~binding ~coord:(Schedule.coord_index c) e in
      outs := s :: !outs;
      if not (Float.equal d s) then ok := false);
  Format.printf "eval=%s direct==symbolic=%b@."
    (String.concat "," (List.map (Printf.sprintf "%g") (List.rev !outs)))
    !ok;
  [%expect
    {| eval=0.25,0.5,0.25,0.5,1,0.5,0.25,0.5,0.25 direct==symbolic=true |}]

let%expect_test "Symbolic linear: eval matches Direct" =
  let module S = Symbolic.Make () in
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
  let xs = Tensor_sig.create ~name:"x" ~shape:x_shape ~fmt:f32 () in
  let ws = Tensor_sig.create ~name:"w" ~shape:weight_shape ~fmt:f32 () in
  let bs = Tensor_sig.create ~name:"b" ~shape:bias_shape ~fmt:f32 () in
  let p = { Linear.Linear.in_features = Dim.extent 3 } in
  let out_shape = Linear.Linear.output_shape ~x_shape ~weight_shape in
  let e = Ls.pixel p ~x:xs ~weight:ws ~bias:bs Symbolic.out_coord in
  let binding (s : Tensor_sig.t) =
    if s.id = xs.id then x else if s.id = ws.id then weight else bias
  in
  let outs = ref [] and ok = ref true in
  Vec6.iter out_shape (fun c ->
      let d = Ld.pixel p ~x ~weight ~bias (Schedule.coord_index c) in
      let s = Expr.eval ~binding ~coord:(Schedule.coord_index c) e in
      outs := s :: !outs;
      if not (Float.equal d s) then ok := false);
  Format.printf "eval=%s direct==symbolic=%b@."
    (String.concat "," (List.map (Printf.sprintf "%g") (List.rev !outs)))
    !ok;
  [%expect {| eval=11,105 direct==symbolic=true |}]

(* Grounding: the [pixel] expression is built exactly once per op below, then
   [Schedule.ground] is called once per concrete input, against several
   genuinely different inputs — the point being that the expression itself is
   reusable data, not a one-shot evaluation. *)

let%expect_test "Symbolic ground: relu over several different inputs" =
  let module S = Symbolic.Make () in
  let module R = Pointwise.Relu.Compute (S) in
  let module Rd = Pointwise.Relu.Compute (Direct) in
  let x_shape = s1c 4 in
  let xs = Tensor_sig.create ~name:"x" ~shape:x_shape ~fmt:f32 () in
  let e = R.pixel xs Symbolic.out_coord in
  let inputs =
    [ [| -2.; -0.5; 1.; 3. |]; [| 0.; 0.; 0.; 0. |]; [| 5.; -5.; 2.; -2. |] ]
  in
  List.iter
    (fun vals ->
      let x = Tensor.materialize x_shape (fun c -> vals.(chan c)) in
      let binding (s : Tensor_sig.t) =
        if s.id = xs.id then x else assert false
      in
      let direct =
        Schedule.evaluate (Pointwise.Relu.output_shape x_shape) (Rd.pixel x)
      in
      let grounded = Schedule.ground x_shape ~binding e in
      Format.printf "grounded=%a match=%b@." Tensor.pp grounded
        (tensors_match x_shape direct grounded))
    inputs;
  [%expect
    {|
    grounded=tensor f32 [N=1 T=1 D=1 H=1 W=1 C=4] {0, 0, 1, 3} match=true
    grounded=tensor f32 [N=1 T=1 D=1 H=1 W=1 C=4] {0, 0, 0, 0} match=true
    grounded=tensor f32 [N=1 T=1 D=1 H=1 W=1 C=4] {5, 0, 2, 0} match=true
    |}]

let%expect_test "Symbolic ground: add over several different input pairs" =
  let module S = Symbolic.Make () in
  let module A = Pointwise.Add.Compute (S) in
  let module Ad = Pointwise.Add.Compute (Direct) in
  let x_shape = s1c 3 and y_shape = s1c 3 in
  let xs = Tensor_sig.create ~name:"x" ~shape:x_shape ~fmt:f32 () in
  let ys = Tensor_sig.create ~name:"y" ~shape:y_shape ~fmt:f32 () in
  let e = A.pixel xs ys Symbolic.out_coord in
  let out_shape = Pointwise.Add.output_shape x_shape y_shape in
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
      let binding (s : Tensor_sig.t) =
        if s.id = xs.id then x else if s.id = ys.id then y else assert false
      in
      let direct = Schedule.evaluate out_shape (Ad.pixel x y) in
      let grounded = Schedule.ground out_shape ~binding e in
      Format.printf "grounded=%a match=%b@." Tensor.pp grounded
        (tensors_match out_shape direct grounded))
    cases;
  [%expect
    {|
    grounded=tensor f32 [N=1 T=1 D=1 H=1 W=1 C=3] {11, 12, 13} match=true
    grounded=tensor f32 [N=1 T=1 D=1 H=1 W=1 C=3] {-1, -2, -3} match=true
    grounded=tensor f32 [N=1 T=1 D=1 H=1 W=1 C=3] {0, 0, 0} match=true
    |}]

let%expect_test
    "Symbolic ground: conv2d over several different (x, weight, bias)" =
  let module S = Symbolic.Make () in
  let module Cs = Conv.Conv2d.Compute (S) in
  let module Cd = Conv.Conv2d.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:3 ~c:1 in
  let weight_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let xs = Tensor_sig.create ~name:"x" ~shape:x_shape ~fmt:f32 () in
  let ws = Tensor_sig.create ~name:"w" ~shape:weight_shape ~fmt:f32 () in
  let bs = Tensor_sig.create ~name:"b" ~shape:(s1c 1) ~fmt:f32 () in
  let p =
    {
      Conv.Conv2d.kernel = Op_config.Hw.{ h = Dim.extent 2; w = Dim.extent 2 };
      in_channels = Dim.extent 1;
      stride =
        Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
      pad =
        Op_config.Hw.
          { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
    }
  in
  let out_shape = Conv.Conv2d.output_shape ~x_shape ~weight_shape p in
  let e = Cs.pixel p ~x_shape ~x:xs ~weight:ws ~bias:bs Symbolic.out_coord in
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
      let binding (s : Tensor_sig.t) =
        if s.id = xs.id then x else if s.id = ws.id then weight else bias
      in
      let direct =
        Schedule.evaluate out_shape (Cd.pixel p ~x_shape ~x ~weight ~bias)
      in
      let grounded = Schedule.ground out_shape ~binding e in
      Format.printf "grounded=%a match=%b@." Tensor.pp grounded
        (tensors_match out_shape direct grounded))
    cases;
  [%expect
    {|
    grounded=tensor f32 [N=1 T=1 D=1 H=2 W=2 C=1] {8, 12, 20, 24} match=true
    grounded=tensor f32 [N=1 T=1 D=1 H=2 W=2 C=1] {-15, -23, -39, -47} match=true
    grounded=tensor f32 [N=1 T=1 D=1 H=2 W=2 C=1] {2, 2, 2, 2} match=true
    |}]

let%expect_test "Symbolic ground: max_pool2d over several different inputs" =
  let module S = Symbolic.Make () in
  let module Ps = Pool.MaxPool2d.Compute (S) in
  let module Pd = Pool.MaxPool2d.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:3 ~c:1 in
  let xs = Tensor_sig.create ~name:"x" ~shape:x_shape ~fmt:f32 () in
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
  let out_shape = Pool.MaxPool2d.output_shape ~x_shape p in
  let e = Ps.pixel p ~x_shape ~x:xs Symbolic.out_coord in
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
      let binding (s : Tensor_sig.t) =
        if s.id = xs.id then x else assert false
      in
      let direct = Schedule.evaluate out_shape (Pd.pixel p ~x_shape ~x) in
      let grounded = Schedule.ground out_shape ~binding e in
      Format.printf "grounded=%a match=%b@." Tensor.pp grounded
        (tensors_match out_shape direct grounded))
    inputs;
  [%expect
    {|
    grounded=tensor f32 [N=1 T=1 D=1 H=4 W=4 C=1] {0, 1, 2, 2, 3, 4, 5, 5, ...} match=true
    grounded=tensor f32 [N=1 T=1 D=1 H=4 W=4 C=1] {-1, -1, -2, -3, -1, -1, -2, -3, ...} match=true
    grounded=tensor f32 [N=1 T=1 D=1 H=4 W=4 C=1] {9, 9, 0, 0, 9, 9, 9, 0, ...} match=true
    |}]

let%expect_test "Symbolic ground: avg_pool2d over several different inputs" =
  let module S = Symbolic.Make () in
  let module Ps = Pool.AvgPool2d.Compute (S) in
  let module Pd = Pool.AvgPool2d.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:3 ~c:1 in
  let xs = Tensor_sig.create ~name:"x" ~shape:x_shape ~fmt:f32 () in
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
  let out_shape = Pool.AvgPool2d.output_shape ~x_shape p in
  let e = Ps.pixel p ~x_shape ~x:xs Symbolic.out_coord in
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
      let binding (s : Tensor_sig.t) =
        if s.id = xs.id then x else assert false
      in
      let direct = Schedule.evaluate out_shape (Pd.pixel p ~x_shape ~x) in
      let grounded = Schedule.ground out_shape ~binding e in
      Format.printf "grounded=%a match=%b@." Tensor.pp grounded
        (tensors_match out_shape direct grounded))
    inputs;
  [%expect
    {|
    grounded=tensor f32 [N=1 T=1 D=1 H=4 W=4 C=1] {0.25, 0.5, 0.5, 0.25, 0.5, 1, 1, 0.5, ...} match=true
    grounded=tensor f32 [N=1 T=1 D=1 H=4 W=4 C=1] {0, 0.25, 0.75, 0.5, 0.75, 2, 3, 1.75, ...} match=true
    grounded=tensor f32 [N=1 T=1 D=1 H=4 W=4 C=1] {2.25, 2.25, 0, 0, 2.25, 4.5, 2.25, 0, ...} match=true
    |}]

let%expect_test
    "Symbolic ground: linear over several different (x, weight, bias)" =
  let module S = Symbolic.Make () in
  let module Ls = Linear.Linear.Compute (S) in
  let module Ld = Linear.Linear.Compute (Direct) in
  let x_shape = s1c 3 in
  let weight_shape = Vec6.shape ~n:2 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3 in
  let bias_shape = s1c 2 in
  let xs = Tensor_sig.create ~name:"x" ~shape:x_shape ~fmt:f32 () in
  let ws = Tensor_sig.create ~name:"w" ~shape:weight_shape ~fmt:f32 () in
  let bs = Tensor_sig.create ~name:"b" ~shape:bias_shape ~fmt:f32 () in
  let p = { Linear.Linear.in_features = Dim.extent 3 } in
  let out_shape = Linear.Linear.output_shape ~x_shape ~weight_shape in
  let e = Ls.pixel p ~x:xs ~weight:ws ~bias:bs Symbolic.out_coord in
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
      let binding (s : Tensor_sig.t) =
        if s.id = xs.id then x else if s.id = ws.id then weight else bias
      in
      let direct = Schedule.evaluate out_shape (Ld.pixel p ~x ~weight ~bias) in
      let grounded = Schedule.ground out_shape ~binding e in
      Format.printf "grounded=%a match=%b@." Tensor.pp grounded
        (tensors_match out_shape direct grounded))
    cases;
  [%expect
    {|
    grounded=tensor f32 [N=1 T=1 D=1 H=1 W=1 C=2] {11, 105} match=true
    grounded=tensor f32 [N=1 T=1 D=1 H=1 W=1 C=2] {-1, -5} match=true
    grounded=tensor f32 [N=1 T=1 D=1 H=1 W=1 C=2] {7, 5} match=true
    |}]

let%expect_test "Symbolic ground: mean over spatial (H,W), several inputs" =
  let module S = Symbolic.Make () in
  let module Ms = Reduce.Mean.Compute (S) in
  let module Md = Reduce.Mean.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let xs = Tensor_sig.create ~name:"x" ~shape:x_shape ~fmt:f32 () in
  let p = { Reduce.Mean.dims = [ Axis.H; Axis.W ]; keepdim = true } in
  let out_shape = Reduce.Mean.output_shape ~x_shape p in
  let e = Ms.pixel p ~x_shape ~x:xs Symbolic.out_coord in
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
      let binding (s : Tensor_sig.t) =
        if s.id = xs.id then x else assert false
      in
      let direct = Schedule.evaluate out_shape (Md.pixel p ~x_shape ~x) in
      let grounded = Schedule.ground out_shape ~binding e in
      Format.printf "grounded=%a match=%b@." Tensor.pp grounded
        (tensors_match out_shape direct grounded))
    inputs;
  [%expect
    {|
    grounded=tensor f32 [N=1 T=1 D=1 H=1 W=1 C=1] {2.5} match=true
    grounded=tensor f32 [N=1 T=1 D=1 H=1 W=1 C=1] {5} match=true
    grounded=tensor f32 [N=1 T=1 D=1 H=1 W=1 C=1] {4} match=true |}]

let%expect_test "Symbolic rms_norm over C: expr structure + eval matches Direct"
    =
  let module S = Symbolic.Make () in
  let module Rd = Norm.RmsNorm.Compute (Direct) in
  let module Rs = Norm.RmsNorm.Compute (S) in
  let x_shape = s1c 2 in
  let x = Tensor.materialize x_shape (fun c -> [| 1.; 7. |].(chan c)) in
  let weight = Tensor.materialize x_shape (fun _ -> 1.) in
  let xs = Tensor_sig.create ~name:"x" ~shape:x_shape ~fmt:f32 () in
  let ws = Tensor_sig.create ~name:"w" ~shape:x_shape ~fmt:f32 () in
  let p = { Norm.RmsNorm.dims = [ Axis.C ]; eps = 0. } in
  let out_shape = Norm.RmsNorm.output_shape ~x_shape in
  let e = Rs.pixel p ~x_shape ~x:xs ~weight:ws Symbolic.out_coord in
  Format.printf "%a@." Expr.pp e;
  [%expect
    {| ((x[N,T,D,H,W,C] * (1 / sqrt(((sum(r1=0..2: (x[N,T,D,H,W,r1] * x[N,T,D,H,W,r1])) / 2) + 0)))) * w[0,0,0,0,0,C]) |}];
  let binding (s : Tensor_sig.t) = if s.id = xs.id then x else weight in
  let outs = ref [] and ok = ref true in
  Vec6.iter out_shape (fun c ->
      let d = Rd.pixel p ~x_shape ~x ~weight (Schedule.coord_index c) in
      let s = Expr.eval ~binding ~coord:(Schedule.coord_index c) e in
      outs := s :: !outs;
      if not (Float.equal d s) then ok := false);
  (* mean(1²,7²)=25, sqrt=5, x/5 -> {0.2, 1.4} *)
  Format.printf "eval=%s direct==symbolic=%b@."
    (String.concat "," (List.map (Printf.sprintf "%g") (List.rev !outs)))
    !ok;
  [%expect {| eval=0.2,1.4 direct==symbolic=true |}]

let%expect_test
    "Symbolic ground: bmm over several different (input, mat2) pairs" =
  let module S = Symbolic.Make () in
  let module Bs = Matmul.Bmm.Compute (S) in
  let module Bd = Matmul.Bmm.Compute (Direct) in
  (* B=2, 2×3 × 3×2 → 2×2 output per batch item *)
  let input_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:3 in
  let mat2_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:2 in
  let ins = Tensor_sig.create ~name:"input" ~shape:input_shape ~fmt:f32 () in
  let m2s = Tensor_sig.create ~name:"mat2" ~shape:mat2_shape ~fmt:f32 () in
  let out_shape = Matmul.Bmm.output_shape ~input_shape ~mat2_shape in
  let e = Bs.pixel ~input_shape ~input:ins ~mat2:m2s Symbolic.out_coord in
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
      let binding (s : Tensor_sig.t) =
        if s.id = ins.id then input
        else if s.id = m2s.id then mat2
        else assert false
      in
      let direct =
        Schedule.evaluate out_shape (Bd.pixel ~input_shape ~input ~mat2)
      in
      let grounded = Schedule.ground out_shape ~binding e in
      Format.printf "grounded=%a match=%b@." Tensor.pp grounded
        (tensors_match out_shape direct grounded))
    cases;
  [%expect
    {|
    grounded=tensor f32 [N=1 T=1 D=1 H=2 W=2 C=2] {1, 1, 2, 2, 11, 11, 12, 12} match=true
    grounded=tensor f32 [N=1 T=1 D=1 H=2 W=2 C=2] {0, 0, 0, 0, 0, 0, 0, 0} match=true
    grounded=tensor f32 [N=1 T=1 D=1 H=2 W=2 C=2] {3, 3, 3, 3, 3, 3, 3, 3} match=true
    |}]
