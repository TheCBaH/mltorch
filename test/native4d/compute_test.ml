(* Native4D evaluation. Two properties, checked against different oracles.
   The shape/gather family (permute4, pad4, unbind, select4, split_with_sizes,
   index_tensor4) lives in compute_shape_test.ml, split out under the tracked
   file-size ceiling; this file and that one share both properties below and
   compute_fixtures.ml's helpers.

   1. DIRECT values, against numbers worked out by hand. Not against Native:
      both sides instantiate the same [Compute (S)] functor, so agreement would
      show the parameter translation is self-consistent and say nothing about
      whether it is right.

   2. DIRECT versus grounded SYMBOLIC, bitwise. That one IS a self-comparison —
      the two modes share [Eval_op4] — and is worth having for what it does
      cover: staging, scheduling, the shape machinery, and the [Symbolic]
      instance's binding of constants. It is the stage-3 acceptance criterion.

   What each catches, measured by breaking [Eval_op4] and watching, rather than
   assumed:

   - A WRONG GROUP COUNT is not caught here at all. Setting
     [Depthwise_conv2d]'s groups to 1 raises out of Native's own convolution
     validation — "weight C extent 1 must equal in_channels/groups 2" — because
     a depthwise weight has one input channel per group. The compute these arms
     delegate to guards its own parameters, which is an argument for delegating.

   - A WRONG PERMUTATION MAP is caught only by property 1 (see
     compute_shape_test.ml's own "permute4 swaps H and W" test). Making [perm6]
     the identity leaves the per-op table entirely green, since both modes
     share the translation.

   So the two properties are not redundant, and neither subsumes the other. *)

open Native4d
open Compute_fixtures

(* ---- direct, against hand values ------------------------------------------ *)

let%expect_test "direct4: pointwise" =
  let shape = s4 ~n:1 ~h:1 ~w:1 ~c:4 in
  let g =
    build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape () in
       let* y = input ~shape () in
       let* s = add x y in
       relu s)
  in
  (* -1.5 makes the sum straddle zero, so relu clamps exactly half. An offset
     that clamped all four or none would be indistinguishable from a relu that
     was not applied at all. *)
  let a = Tensor.materialize (Shape4.to_vec6 shape) (fun _ -> -1.5) in
  let b = seq shape in
  let ids = g.Graph.Graph.inputs in
  Format.printf "relu(-1.5 + [0 1 2 3]), expect [0 0 0.5 1.5]: %a@." pp_values
    (values (single g ~inputs:[ (List.nth ids 0, a); (List.nth ids 1, b) ] ()));
  [%expect {| relu(-1.5 + [0 1 2 3]), expect [0 0 0.5 1.5]: [0 0 0.5 1.5] |}]

(* H fans from 1 to 3 and C fans from 1 to 2, W stays fixed at its own
   extent: a wrong per-axis broadcast coordinate would repeat the wrong
   values across H or C instead of W's own two. *)
let%expect_test "direct4: expand4 fans H and C, keeps W" =
  let x_shape = s4 ~n:1 ~h:1 ~w:2 ~c:1 in
  let target = s4 ~n:1 ~h:3 ~w:2 ~c:2 in
  let g =
    build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape:x_shape () in
       expand4 target x)
  in
  let x =
    Tensor.materialize (Shape4.to_vec6 x_shape) (fun c ->
        if Dim.to_int (Vec6.get c Axis.W) = 0 then 3. else 7.)
  in
  let id = List.hd g.Graph.Graph.inputs in
  Format.printf "%a@." pp_values (values (single g ~inputs:[ (id, x) ] ()));
  [%expect {| [3 3 7 7 3 3 7 7 3 3 7 7] |}]

let%expect_test "direct4: unbind preserves int64 cells" =
  let shape = s4 ~n:1 ~h:1 ~w:2 ~c:2 in
  let i64 = Payload.Fmt Payload.I64 in
  let g =
    Builder.build ~dtype:i64 ~outputs:Fun.id
      (let open Builder in
       let* x = input ~shape () in
       unbind Axis4.C x)
    |> Err.or_raise ~pp_error:Builder.pp_error
  in
  let data =
    Bigarray.(
      Array1.of_array int64 c_layout
        [| Int64.min_int; -1L; 9_007_199_254_740_993L; Int64.max_int |])
  in
  let x =
    Tensor.Tensor
      {
        Tensor.shape = Shape4.to_vec6 shape;
        payload = { Payload.fmt = Payload.I64; quant = Payload.No_quant; data };
      }
  in
  let env = run_direct g ~inputs:[ (List.hd g.Graph.Graph.inputs, x) ] in
  List.iter
    (fun id -> Format.printf "%a@." Tensor.pp (Tensor_id.Map.find id env))
    g.Graph.Graph.outputs;
  [%expect
    {|
    tensor i64 [C=2] {-9223372036854775808, 9007199254740993}
    tensor i64 [C=2] {-1, 9223372036854775807} |}]

let%expect_test "direct4: adaptive_avg_pool2d keeps ATen bins" =
  let shape = s4 ~n:1 ~h:5 ~w:5 ~c:1 in
  let params =
    {
      Pool.AdaptiveAvgPool2d.output_size =
        Op_config.Hw.{ h = Op_config.Pos.of_int 3; w = Op_config.Pos.of_int 3 };
    }
  in
  let g =
    build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape () in
       adaptive_avg_pool2d params x)
  in
  let x = seq shape in
  let id = List.hd g.Graph.Graph.inputs in
  Format.printf "%a@." pp_values (values (single g ~inputs:[ (id, x) ] ()));
  [%expect {| [3 4.5 6 10.5 12 13.5 18 19.5 21] |}]

let chan c = Dim.to_int (Vec6.get c Axis.C)
let axis_int axis c = Dim.to_int (Vec6.get c axis)

let activation_single f ~values:vs =
  let shape = s4 ~n:1 ~h:1 ~w:1 ~c:(List.length vs) in
  let g =
    build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape () in
       f x)
  in
  let x =
    Tensor.materialize (Shape4.to_vec6 shape) (fun c -> List.nth vs (chan c))
  in
  let ids = g.Graph.Graph.inputs in
  values (single g ~inputs:[ (List.nth ids 0, x) ] ())

(* Hand values, not Native as oracle: [Compute (Direct)] is the same functor
   whichever dialect calls it, so agreement between Native and Native4D here
   would only show the parameter translation is self-consistent, not that the
   formula is right. -6, -0.5, 0.5, 6 are worked out by hand in op5-impl. *)
let%expect_test "direct4: silu" =
  let vs = activation_single Builder.silu ~values:[ -6.; -0.5; 0.5; 6. ] in
  Format.printf "%a@." pp_values vs;
  [%expect {| [-0.0148357 -0.18877 0.31123 5.98516] |}]

let%expect_test "direct4: hardsigmoid" =
  let vs =
    activation_single Builder.hardsigmoid ~values:[ -6.; -0.5; 0.5; 6. ]
  in
  Format.printf "%a@." pp_values vs;
  [%expect {| [0 0.416667 0.583333 1] |}]

let%expect_test "direct4: hardswish" =
  let vs = activation_single Builder.hardswish ~values:[ -6.; -0.5; 0.5; 6. ] in
  Format.printf "%a@." pp_values vs;
  [%expect {| [-0 -0.208333 0.291667 6] |}]

let%expect_test "direct4: leaky_relu" =
  let vs =
    activation_single
      (Builder.leaky_relu { Pointwise.Leaky_relu.negative_slope = 0.2 })
      ~values:[ -2.; -0.5; 0.; 3. ]
  in
  Format.printf "%a@." pp_values vs;
  [%expect {| [-0.4 -0.1 0 3] |}]

let%expect_test "direct4: zeros4 preserves requested F64 dtype" =
  let g =
    build
      ~outputs:(fun y -> [ y ])
      (Builder.zeros4
         {
           Ops4.Zeros4.shape = s4 ~n:1 ~h:1 ~w:2 ~c:2;
           fmt = Payload.Fmt Payload.F64;
         })
  in
  Format.printf "%a@." Tensor.pp (single g ~inputs:[] ());
  [%expect {| tensor f64 [W=2 C=2] {0, 0, 0, 0} |}]

let%expect_test "direct4: arange4 preserves Float start and endpoint" =
  let g =
    build
      ~outputs:(fun y -> [ y ])
      (Builder.arange4
         {
           Ops4.Arange4.start = 0.5;
           stop = 4.;
           step = 1.;
           fmt = Payload.Fmt Payload.F32;
         })
  in
  Format.printf "%a@." Tensor.pp (single g ~inputs:[] ());
  [%expect {| tensor f32 [C=4] {0.5, 1.5, 2.5, 3.5} |}]

let%expect_test "direct4: batch_norm_no_stats keeps activation and statistics" =
  let x_shape = s4 ~n:1 ~h:2 ~w:1 ~c:2 in
  let c_shape = s4 ~n:1 ~h:1 ~w:1 ~c:2 in
  let g =
    build ~outputs:Fun.id
      (let open Builder in
       let* x = input ~shape:x_shape () in
       let* weight = input ~shape:c_shape () in
       let* bias = input ~shape:c_shape () in
       batch_norm_no_stats
         { Ops4.Batch_norm_no_stats.channel = Axis4.C; eps = 0. }
         ~x ~weight ~bias ())
  in
  let x =
    Tensor.materialize (Shape4.to_vec6 x_shape) (fun c ->
        [| [| 1.; 3. |]; [| 5.; 7. |] |].(chan c).(Dim.to_int
                                                     (Vec6.get c Axis.H)))
  in
  let vec2 a b =
    Tensor.materialize (Shape4.to_vec6 c_shape) (fun c -> [| a; b |].(chan c))
  in
  let env =
    run_direct g
      ~inputs:
        (List.combine g.Graph.Graph.inputs [ x; vec2 2. 10.; vec2 1. (-1.) ])
  in
  List.iter
    (fun id ->
      Format.printf "%a@." pp_values (values (Tensor_id.Map.find id env)))
    g.Graph.Graph.outputs;
  [%expect {|
    [-1 -11 3 9]
    [2 6]
    [1 1] |}]

let%expect_test "symbolic4: batch_norm_no_stats reaches the kernel adapter" =
  let x_shape = s4 ~n:1 ~h:2 ~w:1 ~c:2 in
  let c_shape = s4 ~n:1 ~h:1 ~w:1 ~c:2 in
  let g =
    build ~outputs:Fun.id
      (let open Builder in
       let* x = input ~shape:x_shape () in
       let* weight = input ~shape:c_shape () in
       let* bias = input ~shape:c_shape () in
       batch_norm_no_stats
         { Ops4.Batch_norm_no_stats.channel = Axis4.C; eps = 1e-5 }
         ~x ~weight ~bias ())
  in
  (match Kernel_adapt.of_stage_program (Eval_symbolic4.run g) with
  | Ok _ -> print_endline "kernel accepted"
  | Error e -> Format.printf "%a@." Kernel_adapt.pp_error (Err.Error.kind e));
  [%expect {| kernel accepted |}]

(* Two DIFFERENT per-head dot products (H=2): hard-coding H=0 on either
   operand, [Bmm]'s own restriction, would print [17 53] as [17 17]. *)
let%expect_test "direct4: batched_matmul contracts per head" =
  let a_shape = s4 ~n:1 ~h:2 ~w:1 ~c:2 and b_shape = s4 ~n:1 ~h:2 ~w:2 ~c:1 in
  let g =
    build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* a = input ~shape:a_shape () in
       let* b = input ~shape:b_shape () in
       batched_matmul a b)
  in
  let mk shape off base =
    Tensor.materialize (Shape4.to_vec6 shape) (fun c ->
        float_of_int ((axis_int Axis.H c * 2) + off c + base))
  in
  let a = mk a_shape chan 1 in
  let b = mk b_shape (axis_int Axis.W) 5 in
  let ids = g.Graph.Graph.inputs in
  Format.printf "expect [17 53]: %a@." pp_values
    (values (single g ~inputs:[ (List.nth ids 0, a); (List.nth ids 1, b) ] ()));
  [%expect {| expect [17 53]: [17 53] |}]

(* Depthwise is the arm where a wrong group count is invisible without a hand
   value: with in_channels = 2 and a 1x1 weight, output channel i is
   x[i] * w[i], whereas groups=1 would sum both channels into each output. *)
let%expect_test "direct4: depthwise convolution is per channel" =
  let x_shape = s4 ~n:1 ~h:1 ~w:1 ~c:2 in
  let w_shape = s4 ~n:2 ~h:1 ~w:1 ~c:1 in
  let params : Ops4.Conv_params.t =
    {
      h = Fixtures4.axis_window ~kernel:1;
      w = Fixtures4.axis_window ~kernel:1;
      in_channels = Dim.extent 2;
    }
  in
  let g =
    build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape:x_shape () in
       let* w = constant ~shape:w_shape () in
       depthwise_conv2d params ~x ~weight:w ())
  in
  let x =
    Tensor.materialize (Shape4.to_vec6 x_shape) (fun c ->
        if Dim.to_int (Vec6.get c Axis.C) = 0 then 3. else 5.)
  in
  let w =
    Tensor.materialize (Shape4.to_vec6 w_shape) (fun c ->
        if Dim.to_int (Vec6.get c Axis.N) = 0 then 10. else 100.)
  in
  let ids = g.Graph.Graph.inputs in
  Format.printf "expect [30 500]: %a@." pp_values
    (values
       (single g
          ~inputs:[ (List.nth ids 0, x) ]
          ~constants:[ (List.nth ids 1, w) ]
          ()));
  [%expect {| expect [30 500]: [30 500] |}]

(* The general form, between [Conv2d] (all channels shared) and
   [Depthwise_conv2d] (one channel per group) above: 2 groups of 2 input
   channels each, so a hand value catches a wrong per-group channel slice —
   output channel 0 must read input channels [0;1], never [2;3]. *)
let%expect_test "direct4: grouped convolution reads its own group's channels" =
  let x_shape = s4 ~n:1 ~h:1 ~w:1 ~c:4 in
  let w_shape = s4 ~n:2 ~h:1 ~w:1 ~c:2 in
  let params : Ops4.Grouped_conv_params.t =
    {
      h = Fixtures4.axis_window ~kernel:1;
      w = Fixtures4.axis_window ~kernel:1;
      in_channels = Dim.extent 4;
      groups = Op_config.Pos.of_int 2;
    }
  in
  let g =
    build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape:x_shape () in
       let* w = constant ~shape:w_shape () in
       grouped_conv2d params ~x ~weight:w ())
  in
  let x =
    Tensor.materialize (Shape4.to_vec6 x_shape) (fun c ->
        match Dim.to_int (Vec6.get c Axis.C) with
        | 0 -> 3.
        | 1 -> 5.
        | 2 -> 7.
        | _ -> 11.)
  in
  let w =
    Tensor.materialize (Shape4.to_vec6 w_shape) (fun c ->
        match
          (Dim.to_int (Vec6.get c Axis.N), Dim.to_int (Vec6.get c Axis.C))
        with
        | 0, 0 -> 10.
        | 0, _ -> 1.
        | _, 0 -> 1.
        | _ -> 10.)
  in
  let ids = g.Graph.Graph.inputs in
  Format.printf "expect [35 117]: %a@." pp_values
    (values
       (single g
          ~inputs:[ (List.nth ids 0, x) ]
          ~constants:[ (List.nth ids 1, w) ]
          ()));
  [%expect {| expect [35 117]: [35 117] |}]

(* Two groups of two channels each, mean/var computed only WITHIN a group:
   group0 = {3;5} (mean 4, std 1) and group1 = {7;13} (mean 10, std 3) give
   the SAME normalized pattern [-1;1] despite very different absolute
   scales -- a wrong window (one global mean/var across all four channels)
   would not. *)
let%expect_test "direct4: group_norm4 reduces within its own channel group" =
  let x_shape = s4 ~n:1 ~h:1 ~w:1 ~c:4 in
  let c_shape = s4 ~n:1 ~h:1 ~w:1 ~c:4 in
  let params : Ops4.Group_norm4.params =
    { channel = Axis4.C; groups = Op_config.Pos.of_int 2; eps = 1e-8 }
  in
  let g =
    build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape:x_shape () in
       let* w = constant ~shape:c_shape () in
       let* b = constant ~shape:c_shape () in
       group_norm4 params ~x ~weight:w ~bias:b ())
  in
  let x =
    Tensor.materialize (Shape4.to_vec6 x_shape) (fun c ->
        match Dim.to_int (Vec6.get c Axis.C) with
        | 0 -> 3.
        | 1 -> 5.
        | 2 -> 7.
        | _ -> 13.)
  in
  let ones = Tensor.materialize (Shape4.to_vec6 c_shape) (fun _ -> 1.) in
  let zeros = Tensor.materialize (Shape4.to_vec6 c_shape) (fun _ -> 0.) in
  let ids = g.Graph.Graph.inputs in
  Format.printf "expect [-1 1 -1 1]: %a@." pp_values
    (values
       (single g
          ~inputs:[ (List.nth ids 0, x) ]
          ~constants:[ (List.nth ids 1, ones); (List.nth ids 2, zeros) ]
          ()));
  [%expect {| expect [-1 1 -1 1]: [-1 1 -1 1] |}]

let%expect_test "direct4: mean over H and W keeps the axes" =
  let shape = s4 ~n:1 ~h:2 ~w:2 ~c:1 in
  let g =
    build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape () in
       mean_keepdims [ Axis4.H; Axis4.W ] x)
  in
  let out = single g ~inputs:[ (List.hd g.Graph.Graph.inputs, seq shape) ] () in
  let (Tensor.Tensor tt) = out in
  Format.printf "shape %a, mean of 0..3 = %a@." Vec6.pp_shape tt.Tensor.shape
    pp_values (values out);
  [%expect {| shape [C=1], mean of 0..3 = [1.5] |}]

(* Hand values, not Native-as-oracle: both sides would instantiate the same
   [Reduce.Softmax.Legacy_pixel] oracle, so agreement would prove the adapter and
   the staging rather than the arithmetic.

   [0, ln 3] over a two-element W axis gives exp values [1, 3], summing to 4:
   softmax = [1/4, 3/4], exactly representable so the comparison is not about
   float printing. *)
let%expect_test "direct4: softmax4 over W" =
  let shape = s4 ~n:1 ~h:1 ~w:2 ~c:1 in
  let g =
    build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape () in
       softmax4 { Ops4.Softmax4.axis = Axis4.W } x)
  in
  let x =
    Tensor.materialize (Shape4.to_vec6 shape) (fun c ->
        if Dim.to_int (Vec6.get c Axis.W) = 0 then 0. else Float.log 3.)
  in
  let env = run_direct g ~inputs:[ (List.hd g.Graph.Graph.inputs, x) ] in
  Format.printf "%a@." Tensor.pp
    (Tensor_id.Map.find (List.hd g.Graph.Graph.outputs) env);
  [%expect {| tensor f32 [W=2 C=1] {0.25, 0.75} |}]

(* [E=1] and [scale=Default] make [sf=1], reducing the score to [q*k]: [q=1]
   over [softmax4]'s own [0, ln 3] key setup above reproduces its exact
   [1/4, 3/4] weights, so weighting values (10, 2) by them keeps the answer
   (2.5 + 1.5) exact too. No mask -- exercises [Sdpa_mask]'s synthetic fill. *)
let%expect_test "direct4: sdpa attends over the key axis" =
  let q_shape = s4 ~n:1 ~h:1 ~w:1 ~c:1 and kv_shape = s4 ~n:1 ~h:1 ~w:2 ~c:1 in
  let g =
    build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* query = input ~shape:q_shape () in
       let* key = input ~shape:kv_shape () in
       let* value = input ~shape:kv_shape () in
       sdpa
         { Attention.Sdpa.scale = Attention.Sdpa.Scale.Default }
         ~query ~key ~value ())
  in
  let query = Tensor.materialize (Shape4.to_vec6 q_shape) (fun _ -> 1.) in
  let step w0 w1 c = if axis_int Axis.W c = 0 then w0 else w1 in
  let key =
    Tensor.materialize (Shape4.to_vec6 kv_shape) (step 0. (Float.log 3.))
  in
  let value = Tensor.materialize (Shape4.to_vec6 kv_shape) (step 10. 2.) in
  let ids = g.Graph.Graph.inputs in
  let inputs =
    [ (List.nth ids 0, query); (List.nth ids 1, key); (List.nth ids 2, value) ]
  in
  Format.printf "expect [4]: %a@." pp_values (values (single g ~inputs ()));
  [%expect {| expect [4]: [4] |}]

let%expect_test "direct4: authored Regions materialize once per key" =
  let shape = s4 ~n:1 ~h:1 ~w:2 ~c:3 in
  let build_region f =
    build
      ~outputs:(fun output -> [ output ])
      (let open Builder in
       let* x = input ~shape () in
       f x)
  in
  let rms =
    build_region (fun x ->
        Builder.rms_norm { Ops4.Rms_norm.dims = [ Axis4.C ]; eps = 1e-5 } ~x ())
  in
  let layer =
    build_region (fun x ->
        Builder.layer_norm4
          { Ops4.Layer_norm.dims = [ Axis4.C ]; eps = 1e-5 }
          ~x ())
  in
  let softmax =
    build_region (fun x -> Builder.softmax4 { Ops4.Softmax4.axis = Axis4.C } x)
  in
  let input = seq shape in
  let count name graph =
    let output = List.hd graph.Graph.Graph.outputs in
    let counters = Region_execution.counters () in
    ignore
      (Eval_direct4.run
         ~region_counters:(Tensor_id.Map.singleton output counters)
         graph
         ~inputs:[ (List.hd graph.Graph.Graph.inputs, input) ]
      |> Err.or_raise ~pp_error:Eval_direct4.pp_error);
    Format.printf "%s keys=%d locals=%d emitters=%d loads=%d reductions=%d@."
      name counters.keys counters.locals counters.emitters counters.loads
      counters.reductions
  in
  List.iter
    (fun (name, graph) -> count name graph)
    [ ("rms", rms); ("layer", layer); ("softmax", softmax) ];
  [%expect
    {|
    rms keys=2 locals=4 emitters=6 loads=24 reductions=6
    layer keys=2 locals=8 emitters=6 loads=36 reductions=12
    softmax keys=2 locals=4 emitters=6 loads=18 reductions=12 |}]

let%expect_test "symbolic4: authored Regions carry into the Kernel unchanged" =
  let shape = s4 ~n:1 ~h:1 ~w:2 ~c:3 in
  let build_region f =
    build
      ~outputs:(fun output -> [ output ])
      (let open Builder in
       let* x = input ~shape () in
       f x)
  in
  let cases =
    [
      ( "rms",
        build_region (fun x ->
            Builder.rms_norm
              { Ops4.Rms_norm.dims = [ Axis4.C ]; eps = 1e-5 }
              ~x ()) );
      ( "layer",
        build_region (fun x ->
            Builder.layer_norm4
              { Ops4.Layer_norm.dims = [ Axis4.C ]; eps = 1e-5 }
              ~x ()) );
      ( "softmax",
        build_region (fun x ->
            Builder.softmax4 { Ops4.Softmax4.axis = Axis4.C } x) );
    ]
  in
  List.iter
    (fun (name, graph) ->
      let symbolic = Eval_symbolic4.run graph in
      let stage = List.hd symbolic.Stage_program.stages in
      let carried = stage.Stage_program.Stage.computation in
      let kernel =
        Kernel_adapt.of_stage_program symbolic
        |> Err.or_raise ~pp_error:Kernel_adapt.pp_error
      in
      let received = (List.hd kernel.Kernel.values).Kernel.Value.computation in
      Format.printf "%s region=%b same_object=%b@." name
        (Region_program.pixel_expression carried = None)
        (carried == received))
    cases;
  [%expect
    {|
    rms region=true same_object=true
    layer region=true same_object=true
    softmax region=true same_object=true |}]

let%expect_test "symbolic4: authored Region traces retain unit T and D" =
  let shape = s4 ~n:2 ~h:2 ~w:1 ~c:3 in
  let build_region f =
    build
      ~outputs:(fun output -> [ output ])
      (let open Builder in
       let* x = input ~shape () in
       f x)
  in
  let trace name graph =
    let symbolic = Eval_symbolic4.run graph in
    let stage = List.hd symbolic.Stage_program.stages in
    let program = Stage_program.Stage.computation stage in
    let trace =
      Region_trace.collect program ~output_shape:stage.sg.shape
      |> Err.or_raise ~pp_error:Region_trace.pp_error
    in
    let unit coord =
      Dim.to_int (Vec6.get coord Axis.T) = 0
      && Dim.to_int (Vec6.get coord Axis.D) = 0
    in
    let singleton =
      List.for_all
        (fun entry ->
          unit entry.Region_trace.key && List.for_all unit entry.outputs)
        trace.entries
    in
    let coverage = trace.coverage in
    Format.printf "%s keys=%d total=%d td_singleton=%b@." name coverage.keys
      coverage.total singleton
  in
  let rms =
    build_region (fun x ->
        Builder.rms_norm
          { Ops4.Rms_norm.dims = [ Axis4.N; Axis4.C ]; eps = 1e-5 }
          ~x ())
  in
  let layer =
    build_region (fun x ->
        Builder.layer_norm4
          { Ops4.Layer_norm.dims = [ Axis4.H; Axis4.W; Axis4.C ]; eps = 1e-5 }
          ~x ())
  in
  trace "rms_n_c" rms;
  trace "layer_h_w_c" layer;
  List.iter
    (fun (name, axis) ->
      trace ("softmax_" ^ name)
        (build_region (fun x -> Builder.softmax4 { Ops4.Softmax4.axis } x)))
    [ ("n", Axis4.N); ("h", Axis4.H); ("w", Axis4.W); ("c", Axis4.C) ];
  [%expect
    {|
    rms_n_c keys=2 total=12 td_singleton=true
    layer_h_w_c keys=2 total=12 td_singleton=true
    softmax_n keys=6 total=12 td_singleton=true
    softmax_h keys=6 total=12 td_singleton=true
    softmax_w keys=12 total=12 td_singleton=true
    softmax_c keys=4 total=12 td_singleton=true |}]

let%expect_test "symbolic4: norm Region matrix covers Axis4 and affine states" =
  let shape = s4 ~n:2 ~h:2 ~w:1 ~c:3 in
  let check graph =
    let symbolic = Eval_symbolic4.run graph in
    let stage = List.hd symbolic.Stage_program.stages in
    let trace =
      Region_trace.collect
        (Stage_program.Stage.computation stage)
        ~output_shape:stage.sg.shape
      |> Err.or_raise ~pp_error:Region_trace.pp_error
    in
    List.for_all
      (fun entry ->
        let unit coord =
          Dim.to_int (Vec6.get coord Axis.T) = 0
          && Dim.to_int (Vec6.get coord Axis.D) = 0
        in
        unit entry.Region_trace.key && List.for_all unit entry.outputs)
      trace.entries
  in
  let rms dims =
    build
      ~outputs:(fun output -> [ output ])
      (let open Builder in
       let* x = input ~shape () in
       rms_norm { Ops4.Rms_norm.dims; eps = 1e-5 } ~x ())
  in
  let layer dims ~weight ~bias =
    let affine_shape =
      Norm.normalized_shape ~x_shape:(Shape4.to_vec6 shape)
        ~dims:(List.map Axis4.to_axis dims)
      |> Shape4.of_vec6
      |> Err.or_raise ~pp_error:Shape4.pp_error
    in
    build
      ~outputs:(fun output -> [ output ])
      (let open Builder in
       let* x = input ~shape () in
       match (weight, bias) with
       | false, false -> layer_norm4 { Ops4.Layer_norm.dims; eps = 1e-5 } ~x ()
       | true, false ->
           let* w = input ~shape:affine_shape () in
           layer_norm4 { Ops4.Layer_norm.dims; eps = 1e-5 } ~x ~weight:w ()
       | false, true ->
           let* b = input ~shape:affine_shape () in
           layer_norm4 { Ops4.Layer_norm.dims; eps = 1e-5 } ~x ~bias:b ()
       | true, true ->
           let* w = input ~shape:affine_shape () in
           let* b = input ~shape:affine_shape () in
           layer_norm4
             { Ops4.Layer_norm.dims; eps = 1e-5 }
             ~x ~weight:w ~bias:b ())
  in
  let dims =
    [
      [ Axis4.N ];
      [ Axis4.H ];
      [ Axis4.W ];
      [ Axis4.C ];
      [ Axis4.N; Axis4.C ];
      [ Axis4.H; Axis4.W; Axis4.C ];
    ]
  in
  let rms_ok = List.for_all (fun dims -> check (rms dims)) dims in
  let layer_ok =
    List.for_all
      (fun dims ->
        List.for_all
          (fun (weight, bias) -> check (layer dims ~weight ~bias))
          [ (false, false); (true, false); (false, true); (true, true) ])
      dims
  in
  Format.printf
    "rms_cases=%d rms_td_singleton=%b layer_cases=%d layer_td_singleton=%b@."
    (List.length dims) rms_ok
    (List.length dims * 4)
    layer_ok;
  [%expect
    {| rms_cases=6 rms_td_singleton=true layer_cases=24 layer_td_singleton=true |}]

(* ---- direct versus grounded symbolic --------------------------------------- *)

(* The stage-3 acceptance criterion, over one graph per op. Comparison is
   BITWISE — [Core.Float_bits.exact] — for the reason map_verify.mli gives about
   [Float.equal] conflating -0./+0. and every NaN, which are exactly the
   distinctions a max-pool turns on. *)
let same_bits a b =
  let va = values a and vb = values b in
  List.length va = List.length vb
  && List.for_all2 Core.Float_bits.equal_exact va vb

let agree name g ~inputs ~constants =
  let direct = run_direct g ~constants ~inputs in
  let program = Eval_symbolic4.run g in
  let bind id =
    match List.assoc_opt id inputs with
    | Some t -> t
    | None -> (
        match List.assoc_opt id constants with
        | Some t -> t
        | None -> invalid_arg "unbound")
  in
  let grounded = Stage_program.ground program ~bind in
  (* EVERY output, not [List.hd]. For the eighteen single-output ops that is the
     same thing; for [Unbind] it is the difference between comparing the whole
     op and comparing its first slice, and a per-ordinal bug is exactly what a
     multi-output op can have. *)
  let verdict out =
    let d = Tensor_id.Map.find out direct in
    match Tensor_id.Map.find_opt out grounded with
    | None -> "symbolic produced no stage"
    | Some s -> if same_bits d s then "direct = symbolic" else "MISMATCH"
  in
  match g.Graph.Graph.outputs with
  | [ out ] -> Format.printf "%-22s %s@." name (verdict out)
  | outputs ->
      List.iteri
        (fun i out -> Format.printf "%-22s out%d %s@." name i (verdict out))
        outputs

(* Same coverage discipline as op_json_test.ml: an op with no fixture is an op
   whose two evaluation paths have never been compared. *)
let%expect_test "direct4 = symbolic4: every op has a fixture" =
  Format.printf "fixtures: %d, registry: %d@."
    (List.length (Fixtures4.per_op ()))
    (List.length Op.op_registry);
  [%expect {| fixtures: 59, registry: 59 |}]

let%expect_test "direct4 = symbolic4, bitwise, per op" =
  List.iter
    (fun (name, g, inputs, constants) -> agree name g ~inputs ~constants)
    (Fixtures4.per_op ());
  [%expect
    {|
    add                    direct = symbolic
    sub                    direct = symbolic
    mul                    direct = symbolic
    div                    direct = symbolic
    add_scalar             direct = symbolic
    div_scalar             direct = symbolic
    expand4                direct = symbolic
    mul_scalar             direct = symbolic
    pow                    direct = symbolic
    rsub_scalar            direct = symbolic
    clamp                  direct = symbolic
    hardtanh               direct = symbolic
    leaky_relu             direct = symbolic
    zeros4                 direct = symbolic
    arange4                direct = symbolic
    eye4                   direct = symbolic
    batch_norm_no_stats    out0 direct = symbolic
    batch_norm_no_stats    out1 direct = symbolic
    batch_norm_no_stats    out2 direct = symbolic
    batched_matmul         direct = symbolic
    relu                   direct = symbolic
    repeat4                direct = symbolic
    repeat_interleave4     direct = symbolic
    gelu                   direct = symbolic
    sigmoid                direct = symbolic
    silu                   direct = symbolic
    hardsigmoid            direct = symbolic
    hardswish              direct = symbolic
    sqrt                   direct = symbolic
    to_copy                direct = symbolic
    max_pool2d             direct = symbolic
    adaptive_avg_pool2d    direct = symbolic
    adaptive_max_pool2d    direct = symbolic
    avg_pool2d             direct = symbolic
    mean_keepdims          direct = symbolic
    max_keepdims           direct = symbolic
    sum_keepdims           direct = symbolic
    pad4                   direct = symbolic
    slice4                 direct = symbolic
    softmax4               direct = symbolic
    cumsum4                direct = symbolic
    select4                direct = symbolic
    select_scatter4        direct = symbolic
    concat4                direct = symbolic
    stack4                 direct = symbolic
    permute4               direct = symbolic
    reshape4               direct = symbolic
    rms_norm               direct = symbolic
    layer_norm             direct = symbolic
    sdpa                   direct = symbolic
    group_norm4            direct = symbolic
    conv2d                 direct = symbolic
    depthwise_conv2d       direct = symbolic
    grouped_conv2d         direct = symbolic
    transposed_conv2d      direct = symbolic
    unbind                 out0 direct = symbolic
    unbind                 out1 direct = symbolic
    split_with_sizes4      out0 direct = symbolic
    split_with_sizes4      out1 direct = symbolic
    upsample_bilinear2d    direct = symbolic
    upsample_nearest2d     direct = symbolic
    vector_norm_keepdims   direct = symbolic
    index_tensor4          direct = symbolic |}]
