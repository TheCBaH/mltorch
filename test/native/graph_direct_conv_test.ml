(* Split out of what was graph_test.ml see graph_direct_fixtures.ml. *)

open Graph_ir
open Graph_direct_fixtures

let conv_axis ~kernel ~stride ~pad : Conv.Conv2d.axis_window =
  {
    kernel = Dim.extent kernel;
    stride = Op_config.Pos.of_int stride;
    pad_before = Op_config.Nonneg.of_int pad;
    pad_after = Op_config.Nonneg.of_int pad;
    dilation = Op_config.Pos.of_int 1;
  }

let conv_params =
  {
    Conv.Conv2d.h = conv_axis ~kernel:2 ~stride:1 ~pad:0;
    w = conv_axis ~kernel:2 ~stride:1 ~pad:0;
    in_channels = Dim.extent 2;
    groups = Op_config.Pos.of_int 1;
  }

let p_to_nhwc = Axis.[ (N, N); (T, T); (D, D); (H, W); (W, C); (C, H) ]
let p_to_nchw = Axis.[ (N, N); (T, T); (D, D); (H, C); (W, H); (C, W) ]

let pp_conv_decomp ppf (x_nhwc, y_nhwc, y_nchw, matches) =
  Format.fprintf ppf "%a@.%a@.%a@.y_nhwc matches single conv: %b"
    (pp_named_tensor "x_nhwc") x_nhwc (pp_named_tensor "y_nhwc") y_nhwc
    (pp_named_tensor "y_nchw") y_nchw matches

let%expect_test
    "Direct graph: conv NCHW -> permute -> NHWC conv -> permute -> NCHW" =
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
    let x =
      Tensor.materialize (s 1 1 1 2 3 3) (fun c ->
          let chan = Dim.to_int (Vec6.get c Axis.H) in
          let sph = Dim.to_int (Vec6.get c Axis.W) in
          let spw = Dim.to_int (Vec6.get c Axis.C) in
          float_of_int ((chan * 100) + (sph * 10) + spw))
    in
    let w = Tensor.materialize (s 1 1 1 2 2 2) (fun _ -> 1.) in
    let b = Tensor.materialize (s1c 1) (fun _ -> 0.) in
    let* env =
      lift_eval
        (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x; w; b ]))
    in
    let* x_nhwc = tensor_of_name g env "x_nhwc" in
    let* y_nhwc = tensor_of_name g env "y_nhwc" in
    let* y_nchw = tensor_of_name g env "y_nchw" in
    let module Cv = Conv.Conv2d.Compute (Direct) in
    let (Tensor.Tensor r) = x_nhwc in
    let* ref_shape =
      lift_shape
        (Conv.Conv2d.output_shape ~x_shape:r.shape ~weight_shape:(s 1 1 1 2 2 2)
           conv_params)
    in
    let ref_y =
      Schedule.evaluate ref_shape
        (Cv.pixel conv_params ~x_shape:r.shape ~weight_shape:(s 1 1 1 2 2 2)
           ~x:x_nhwc ~weight:w ~bias:b)
    in
    Err.return (x_nhwc, y_nhwc, y_nchw, Tensor.equal_bits ref_y y_nhwc)
  in
  Format.printf "%a@." (pp_result pp_conv_decomp) result;
  [%expect
    {|
    x_nhwc = tensor f32 [H=3 W=3 C=2] {0, 100, 1, 101, 2, 102, 10, 110, ...}
    y_nhwc = tensor f32 [H=2 W=2 C=1] {444, 452, 524, 532}
    y_nchw = tensor f32 [W=2 C=2] {444, 452, 524, 532}
    y_nhwc matches single conv: true |}]

(* Conv with no bias: omit [?bias] so the IR carries [bias = None]; the evaluator
   fills a zeros bias. The result must equal the same conv with an explicit zero
   bias. *)

let pp_bias_compare ppf (y_no_bias, matches) =
  Format.fprintf ppf "y (no bias) = %a@.matches explicit zero bias: %b"
    Tensor.pp y_no_bias matches

let%expect_test "Direct graph: conv with optional bias omitted (None -> zeros)"
    =
  let build_one ~with_bias =
    Graph_builder.(
      build ~name:"conv" ~outputs:(fun y -> [ y ])
      @@
      let* x = input ~shape:(s 1 1 1 3 3 2) ~name:"x" () in
      let* w = input ~shape:(s 1 1 1 2 2 2) ~name:"w" () in
      if with_bias then
        let* b = input ~shape:(s1c 1) ~name:"b" () in
        conv2d ~name:"y" conv_params ~x ~weight:w ~bias:b ()
      else conv2d ~name:"y" conv_params ~x ~weight:w ())
  in
  let result =
    let open Err.Syntax in
    let x =
      Tensor.materialize (s 1 1 1 3 3 2) (fun c ->
          float_of_int (Dim.to_int (Vec6.get c Axis.H) + chan c))
    in
    let w = Tensor.materialize (s 1 1 1 2 2 2) (fun _ -> 1.) in
    let b = Tensor.materialize (s1c 1) (fun _ -> 0.) in
    let* g_no = lift_build (build_one ~with_bias:false) in
    let* g_yes = lift_build (build_one ~with_bias:true) in
    let* env_no =
      lift_eval
        (Eval_direct.run g_no ~inputs:(List.combine g_no.Graph.inputs [ x; w ]))
    in
    let* env_yes =
      lift_eval
        (Eval_direct.run g_yes
           ~inputs:(List.combine g_yes.Graph.inputs [ x; w; b ]))
    in
    let* y_no = tensor_of_name g_no env_no "y" in
    let* y_yes = tensor_of_name g_yes env_yes "y" in
    Err.return (y_no, Tensor.equal_bits y_no y_yes)
  in
  Format.printf "%a@." (pp_result pp_bias_compare) result;
  [%expect
    {|
    y (no bias) = tensor f32 [H=2 W=2 C=1] {8, 8, 16, 16}
    matches explicit zero bias: true |}]

let%expect_test "Direct graph: transposed convolution" =
  let params =
    {
      Conv.Convolution.stride =
        { h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
      padding = { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
      dilation = { h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
      transposed = true;
      output_padding =
        { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
      groups = Op_config.Pos.of_int 1;
    }
  in
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"transposed" ~outputs:(fun y -> [ y ])
          @@
          let* x = input ~shape:(s 1 1 1 2 2 1) ~name:"x" () in
          let* w = input ~shape:(s 1 1 1 2 2 1) ~name:"w" () in
          convolution ~name:"y" params ~x ~weight:w ())
    in
    let x =
      Tensor.materialize (s 1 1 1 2 2 1) (fun c ->
          float_of_int
            ((Dim.to_int (Vec6.get c Axis.H) * 2)
            + Dim.to_int (Vec6.get c Axis.W)
            + 1))
    in
    let w = Tensor.materialize (s 1 1 1 2 2 1) (fun _ -> 1.) in
    let* env =
      lift_eval
        (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x; w ]))
    in
    tensor_of_name g env "y"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "y")) result;
  [%expect {| y = tensor f32 [H=3 W=3 C=1] {1, 3, 2, 4, 10, 6, 3, 7, ...} |}]

(* A structural group computes add -> relu in the parent graph's global SSA
   namespace. Its result must match the same computation built flat. *)

let pp_nested_compare ppf (nested_out, matches) =
  Format.fprintf ppf "nested_out = %a@.matches flat: %b" Tensor.pp nested_out
    matches

let%expect_test "Direct graph: grouped evaluation" =
  let result =
    let open Err.Syntax in
    let* nested =
      lift_build
        Graph_builder.(
          build ~name:"outer" ~outputs:(fun outs -> outs)
          @@
          let* x = input ~shape:(s1c 4) ~name:"x" () in
          let* y = input ~shape:(s1c 4) ~name:"y" () in
          group ~label:"add_relu"
            (let* t = add ~name:"sum" x y in
             let* r = relu ~name:"r" t in
             return [ r ]))
    in
    let* flat =
      lift_build
        Graph_builder.(
          build ~name:"flat" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s1c 4) ~name:"x" () in
          let* y = input ~shape:(s1c 4) ~name:"y" () in
          let* t = add x y in
          relu ~name:"out" t)
    in
    let x =
      Tensor.materialize (s1c 4) (fun c -> [| 1.; -5.; 2.; -8. |].(chan c))
    in
    let y =
      Tensor.materialize (s1c 4) (fun c -> [| -3.; 1.; 2.; 3. |].(chan c))
    in
    let* env_n =
      lift_eval
        (Eval_direct.run nested
           ~inputs:(List.combine nested.Graph.inputs [ x; y ]))
    in
    let* env_f =
      lift_eval
        (Eval_direct.run flat ~inputs:(List.combine flat.Graph.inputs [ x; y ]))
    in
    let* out_n = tensor_of_name nested env_n "nested_out" in
    let* out_f = tensor_of_name flat env_f "out" in
    Err.return (out_n, Tensor.equal_bits out_n out_f)
  in
  Format.printf "%a@." (pp_result pp_nested_compare) result;
  [%expect
    {|
    nested_out = tensor f32 [C=4] {0, 0, 4, 0}
    matches flat: true |}]

(* Tensor_sig.id must equal the Tensor_id key in the graph's tensor map.
   This verifies the state-monad id generator produces consistent ids. *)
