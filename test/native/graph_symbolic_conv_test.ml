(* Split out of what was graph_symbolic_test.ml see graph_symbolic_fixtures.ml. *)

open Graph_ir
open Graph_symbolic_fixtures

let p_to_nchw = Axis.[ (N, N); (T, T); (D, D); (H, C); (W, H); (C, W) ]

let conv_padding_params =
  {
    Conv.Conv2d_padding.stride =
      { h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
    padding = Conv.Conv2d_padding.Same;
    dilation = { h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
    groups = Op_config.Pos.of_int 1;
  }

let convolution_params =
  {
    Conv.Convolution.stride =
      { h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
    padding = { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
    dilation = { h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
    transposed = false;
    output_padding =
      { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
    groups = Op_config.Pos.of_int 1;
  }

let convolution_transposed_params =
  { convolution_params with Conv.Convolution.transposed = true }

let%expect_test
    "Symbolic graph: conv decomposition stage DAG + ground matches Direct" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"conv_decomp" ~outputs:(fun (_, _, _, y) -> [ y ])
          @@
          let* x = input ~shape:(s 1 1 1 2 3 3) ~name:"x_nchw" () in
          let* w = input ~shape:(s 1 1 1 2 2 2) ~name:"w" () in
          let* b = input ~shape:(s1c 1) ~name:"b" () in
          let* xh = permute ~name:"x_nhwc" p_to_nhwc x in
          let* yh =
            conv2d ~name:"y_nhwc" conv_params ~x:xh ~weight:w ~bias:b ()
          in
          let* y = permute ~name:"y_nchw" p_to_nchw yh in
          return (x, w, b, y))
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let x =
      Tensor.materialize (s 1 1 1 2 3 3) (fun c ->
          let chan = Dim.to_int (Vec6.get c Axis.H) in
          let sph = Dim.to_int (Vec6.get c Axis.W) in
          let spw = Dim.to_int (Vec6.get c Axis.C) in
          float_of_int ((chan * 100) + (sph * 10) + spw))
    in
    let w = Tensor.materialize (s 1 1 1 2 2 2) (fun _ -> 1.) in
    let b = Tensor.materialize (s1c 1) (fun _ -> 0.) in
    let inputs = List.combine g.Graph.inputs [ x; w; b ] in
    let bind id = List.assoc id inputs in
    let grounded =
      Err.or_raise ~pp_error:Stage_program.pp_error
        (Stage_program.ground prog ~bind)
    in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  [%expect
    {|
    inputs: t0, t1, t2
    t3 = t0[N,T,D,C,H,W]
    t4 = (sum(r1=0..2: sum(r2=max(0,-1*H)..min(2,3+-1+-1*H+1): sum(r3=max(0,-1*W)..min(2,3+-1+-1*W+1): (t3[N,T,D,H+r2,W+r3,r1] * t1[C,0,0,r2,r3,r1])))) + t2[0,0,0,0,0,C])
    t5 = t4[N,T,D,W,C,H]
    outputs: t5 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground y_nchw")) result;
  [%expect
    {|
    ground y_nchw = tensor f32 [W=2 C=2] {444, 452, 524, 532}
    ground matches direct: true |}]

let%expect_test "Symbolic graph: conv2d_padding same ground matches Direct" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"conv_padding" ~outputs:(fun y -> [ y ])
          @@
          let* x = input ~shape:(s 1 1 1 3 3 1) ~name:"x" () in
          let* w = input ~shape:(s 1 1 1 2 2 1) ~name:"w" () in
          conv2d_padding ~name:"y" conv_padding_params ~x ~weight:w ())
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let x =
      Tensor.materialize (s 1 1 1 3 3 1) (fun c ->
          let h = Dim.to_int (Vec6.get c Axis.H) in
          let w = Dim.to_int (Vec6.get c Axis.W) in
          float_of_int ((h * 3) + w))
    in
    let w = Tensor.materialize (s 1 1 1 2 2 1) (fun _ -> 1.) in
    let inputs = List.combine g.Graph.inputs [ x; w ] in
    let bind id = List.assoc id inputs in
    let grounded =
      Err.or_raise ~pp_error:Stage_program.pp_error
        (Stage_program.ground prog ~bind)
    in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  [%expect
    {|
    inputs: t0, t1
    t2 = (sum(r1=0..1: sum(r2=max(0,-1*H)..min(2,3+-1+-1*H+1): sum(r3=max(0,-1*W)..min(2,3+-1+-1*W+1): (t0[N,T,D,H+r2,W+r3,r1] * t1[C,0,0,r2,r3,r1])))) + t3[0,0,0,0,0,C])
    outputs: t2 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [H=3 W=3 C=1] {8, 12, 7, 20, 24, 13, 13, 15, ...}
    ground matches direct: true |}]

(* [Conv1d] stages exactly like [Conv2d] with H's window folded away by the
   pinned unit kernel/stride/pad -- there is no separate Symbolic staging
   rule to pin, only that [Conv1d]'s own dispatch reaches the shared
   [Conv2d.Compute] functor. *)
let%expect_test "Symbolic graph: conv1d ground matches Direct" =
  let params =
    {
      Conv.Conv1d.w = conv_axis ~kernel:2 ~stride:1 ~pad:0;
      in_channels = Dim.extent 1;
      groups = Op_config.Pos.of_int 1;
    }
  in
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"conv1d" ~outputs:(fun y -> [ y ])
          @@
          let* x = input ~shape:(s 1 1 1 1 4 1) ~name:"x" () in
          let* w = input ~shape:(s 1 1 1 1 2 1) ~name:"w" () in
          conv1d ~name:"y" params ~x ~weight:w ())
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let x =
      Tensor.materialize (s 1 1 1 1 4 1) (fun c ->
          float_of_int (Dim.to_int (Vec6.get c Axis.W)))
    in
    let w = Tensor.materialize (s 1 1 1 1 2 1) (fun _ -> 1.) in
    let inputs = List.combine g.Graph.inputs [ x; w ] in
    let bind id = List.assoc id inputs in
    let grounded =
      Err.or_raise ~pp_error:Stage_program.pp_error
        (Stage_program.ground prog ~bind)
    in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  [%expect
    {|
    inputs: t0, t1
    t2 = (sum(r1=0..1: sum(r2=max(0,-1*H)..min(1,1+-1+-1*H+1): sum(r3=max(0,-1*W)..min(2,4+-1+-1*W+1): (t0[N,T,D,H+r2,W+r3,r1] * t1[C,0,0,r2,r3,r1])))) + t3[0,0,0,0,0,C])
    outputs: t2 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [W=3 C=1] {1, 3, 5}
    ground matches direct: true |}]

(* [Conv3d] stages with a genuinely THIRD nested [sum] (over D's own window),
   unlike [Conv1d]'s delegation above -- this is the one Symbolic staging
   check that actually exercises [Conv3d]'s own [Compute], not [Conv2d]'s. *)
let%expect_test "Symbolic graph: conv3d ground matches Direct" =
  let params =
    {
      Conv.Conv3d.d = conv_axis ~kernel:2 ~stride:1 ~pad:0;
      h = conv_axis ~kernel:2 ~stride:1 ~pad:0;
      w = conv_axis ~kernel:2 ~stride:1 ~pad:0;
      in_channels = Dim.extent 1;
      groups = Op_config.Pos.of_int 1;
    }
  in
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"conv3d" ~outputs:(fun y -> [ y ])
          @@
          let* x = input ~shape:(s 1 1 2 2 2 1) ~name:"x" () in
          let* w = input ~shape:(s 1 1 2 2 2 1) ~name:"w" () in
          conv3d ~name:"y" params ~x ~weight:w ())
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let x =
      Tensor.materialize (s 1 1 2 2 2 1) (fun c ->
          let d = Dim.to_int (Vec6.get c Axis.D)
          and h = Dim.to_int (Vec6.get c Axis.H)
          and w = Dim.to_int (Vec6.get c Axis.W) in
          float_of_int ((d * 4) + (h * 2) + w))
    in
    let w = Tensor.materialize (s 1 1 2 2 2 1) (fun _ -> 1.) in
    let inputs = List.combine g.Graph.inputs [ x; w ] in
    let bind id = List.assoc id inputs in
    let grounded =
      Err.or_raise ~pp_error:Stage_program.pp_error
        (Stage_program.ground prog ~bind)
    in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  [%expect
    {|
    inputs: t0, t1
    t2 = (sum(r1=0..1: sum(r2=max(0,-1*D)..min(2,2+-1+-1*D+1): sum(r3=max(0,-1*H)..min(2,2+-1+-1*H+1): sum(r4=max(0,-1*W)..min(2,2+-1+-1*W+1): (t0[N,T,D+r2,H+r3,W+r4,r1] * t1[C,0,r2,r3,r4,r1]))))) + t3[0,0,0,0,0,C])
    outputs: t2 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [C=1] {28}
    ground matches direct: true |}]

let%expect_test "Symbolic graph: convolution ground matches Direct" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"convolution" ~outputs:(fun y -> [ y ])
          @@
          let* x = input ~shape:(s 1 1 1 3 3 1) ~name:"x" () in
          let* w = input ~shape:(s 1 1 1 2 2 1) ~name:"w" () in
          convolution ~name:"y" convolution_params ~x ~weight:w ())
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let x =
      Tensor.materialize (s 1 1 1 3 3 1) (fun c ->
          let h = Dim.to_int (Vec6.get c Axis.H) in
          let w = Dim.to_int (Vec6.get c Axis.W) in
          float_of_int ((h * 3) + w))
    in
    let w = Tensor.materialize (s 1 1 1 2 2 1) (fun _ -> 1.) in
    let inputs = List.combine g.Graph.inputs [ x; w ] in
    let bind id = List.assoc id inputs in
    let grounded =
      Err.or_raise ~pp_error:Stage_program.pp_error
        (Stage_program.ground prog ~bind)
    in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  [%expect
    {|
    inputs: t0, t1
    t2 = (sum(r1=0..1: sum(r2=max(0,-1*H)..min(2,3+-1+-1*H+1): sum(r3=max(0,-1*W)..min(2,3+-1+-1*W+1): (t0[N,T,D,H+r2,W+r3,r1] * t1[C,0,0,r2,r3,r1])))) + t3[0,0,0,0,0,C])
    outputs: t2 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [H=2 W=2 C=1] {8, 12, 20, 24}
    ground matches direct: true |}]

let%expect_test "Symbolic graph: transposed convolution ground matches Direct" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"convolution_transposed" ~outputs:(fun y -> [ y ])
          @@
          let* x = input ~shape:(s 1 1 1 2 2 1) ~name:"x" () in
          let* w = input ~shape:(s 1 1 1 2 2 1) ~name:"w" () in
          convolution ~name:"y" convolution_transposed_params ~x ~weight:w ())
    in
    let prog = Eval_symbolic.run g in
    let x =
      Tensor.materialize (s 1 1 1 2 2 1) (fun c ->
          let h = Dim.to_int (Vec6.get c Axis.H) in
          let w = Dim.to_int (Vec6.get c Axis.W) in
          float_of_int ((h * 2) + w + 1))
    in
    let w = Tensor.materialize (s 1 1 1 2 2 1) (fun _ -> 1.) in
    let inputs = List.combine g.Graph.inputs [ x; w ] in
    let bind id = List.assoc id inputs in
    let grounded =
      Err.or_raise ~pp_error:Stage_program.pp_error
        (Stage_program.ground prog ~bind)
    in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [H=3 W=3 C=1] {1, 3, 2, 4, 10, 6, 3, 7, ...}
    ground matches direct: true |}]

(* [Symbolic] is stateless: each stage body is an [Expr.Builder] computation run
   on its own. These pin what that does and does not guarantee, because the
   distinction is the whole reason the adapter no longer holds a supply ref. *)
