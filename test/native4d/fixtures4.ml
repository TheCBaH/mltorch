(* Native4D graphs, one per op, with bindings for every graph input.

   The per-op list is what makes the stage-3 acceptance criterion a real
   criterion rather than a spot check: one graph per op, and the count is
   asserted against the op registry the same way op_json_test.ml asserts its
   samples. An op whose evaluation nobody exercised is an op whose Direct and
   Symbolic paths have never been compared.

   Every fixture exposes ALL of its node's outputs, not just the first. That
   costs nothing for the single-output ops and is the entire point for [Unbind],
   whose consumers compare per ordinal. *)

open Native4d

let s4 ~n ~h ~w ~c = Shape4.of_ints ~n ~h ~w ~c

let axis_window ~kernel : Conv.Conv2d.axis_window =
  {
    kernel = Dim.extent kernel;
    stride = Op_config.Pos.of_int 1;
    pad_before = Op_config.Nonneg.of_int 0;
    pad_after = Op_config.Nonneg.of_int 0;
    dilation = Op_config.Pos.of_int 1;
  }

let hw n : 'a Op_config.Hw.t = { h = n; w = n }

let pool_params : Pool.MaxPool2d.params =
  {
    ceil_mode = false;
    kernel = hw (Dim.extent 2);
    stride = hw (Op_config.Pos.of_int 2);
    pad = hw (Op_config.Nonneg.of_int 0);
  }

let avg_params : Pool.AvgPool2d.params =
  {
    ceil_mode = false;
    count_include_pad = true;
    kernel = hw (Dim.extent 2);
    stride = hw (Op_config.Pos.of_int 2);
    pad = hw (Op_config.Nonneg.of_int 0);
  }

let adaptive_params : Pool.AdaptiveAvgPool2d.params =
  { output_size = hw (Op_config.Pos.of_int 3) }

(* [align_corners=true], the other value the JSON-round-trip fixture in
   op_json_test.ml already exercises with [false] -- between them, both
   [Bilinear_axis.endpoints] branches get a Direct-vs-Symbolic comparison. *)
let upsample_params : Resize.Bilinear2d.params =
  { output_size = hw (Op_config.Pos.of_int 7); align_corners = true }

let nearest_params : Resize.Nearest2d.params =
  { output_size = hw (Op_config.Pos.of_int 7) }

let conv_params ~in_channels ~kernel : Ops4.Conv_params.t =
  {
    h = axis_window ~kernel;
    w = axis_window ~kernel;
    in_channels = Dim.extent in_channels;
  }

let transposed_params : Ops4.Transposed_conv2d.params =
  {
    stride = hw (Op_config.Pos.of_int 1);
    padding = hw (Op_config.Nonneg.of_int 0);
    dilation = hw (Op_config.Pos.of_int 1);
    output_padding = hw (Op_config.Nonneg.of_int 0);
  }

let build ~outputs m =
  Builder.build ~outputs m
  |> Err.or_raise ~pp_error:(fun ppf e ->
      Fmt.pf ppf "fixture4: %a" Builder.pp_error e)

(* Distinct, non-degenerate values: a fill of zeros would make a wrong group
   count or a wrong axis map invisible. Kept small and exactly representable so
   the bitwise comparison is about the computation, not about float printing. *)
let seq shape =
  let i = ref 0. in
  Tensor.materialize (Shape4.to_vec6 shape) (fun _ ->
      i := !i +. 1.;
      !i /. 4.)

(* Every graph input gets a payload, whichever kind it is: [Stage_program.ground]
   binds by id and does not care, and Direct wants inputs and constants split. *)
let bindings (g : Graph.graph) shapes =
  List.map2 (fun id shape -> (id, seq shape)) g.Graph.Graph.inputs shapes

let split (g : Graph.graph) bound =
  List.partition
    (fun (id, _) -> Graph.input_kind g id = Graph_ir.Input.Input)
    bound

(* [unary] and [binary] cover the shape-preserving ops; the rest are spelled out
   because their operand shapes differ. *)
let unary ~shape f =
  let g =
    build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape () in
       f x)
  in
  (g, [ shape ])

let binary ~shape f =
  let g =
    build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* a = input ~shape () in
       let* b = input ~shape () in
       f a b)
  in
  (g, [ shape; shape ])

let nhwc = s4 ~n:1 ~h:4 ~w:4 ~c:2
let flat = s4 ~n:1 ~h:1 ~w:1 ~c:4

(* Each entry: name, graph, input bindings, constant bindings. *)
let per_op () =
  let cases =
    [
      ("add", binary ~shape:nhwc Builder.add);
      ("sub", binary ~shape:nhwc Builder.sub);
      ("mul", binary ~shape:nhwc Builder.mul);
      ("div", binary ~shape:nhwc Builder.div);
      ("add_scalar", unary ~shape:nhwc (Builder.add_scalar 0.5));
      ("div_scalar", unary ~shape:nhwc (Builder.div_scalar 2.));
      ("mul_scalar", unary ~shape:nhwc (Builder.mul_scalar 2.));
      ("pow", unary ~shape:nhwc (Builder.pow 2.));
      ( "clamp",
        unary ~shape:nhwc
          (Builder.clamp { Pointwise.Clamp.min = Some 0.5; max = Some 2. }) );
      ( "hardtanh",
        unary ~shape:nhwc
          (Builder.hardtanh { Pointwise.Hardtanh.min_val = 0.; max_val = 1. })
      );
      ( "leaky_relu",
        unary ~shape:nhwc
          (Builder.leaky_relu { Pointwise.Leaky_relu.negative_slope = 0.2 }) );
      ( "zeros4",
        ( build
            ~outputs:(fun y -> [ y ])
            (Builder.zeros4
               {
                 Ops4.Zeros4.shape = s4 ~n:1 ~h:2 ~w:2 ~c:2;
                 fmt = Payload.Fmt Payload.F32;
               }),
          [] ) );
      ( "arange4",
        ( build
            ~outputs:(fun y -> [ y ])
            (Builder.arange4
               {
                 Ops4.Arange4.start = 0.5;
                 stop = 4.;
                 step = 1.;
                 fmt = Payload.Fmt Payload.F32;
               }),
          [] ) );
      ("relu", unary ~shape:nhwc Builder.relu);
      ("gelu", unary ~shape:nhwc (Builder.gelu Pointwise.Gelu.Exact));
      ("sigmoid", unary ~shape:nhwc Builder.sigmoid);
      ("silu", unary ~shape:nhwc Builder.silu);
      ("hardsigmoid", unary ~shape:nhwc Builder.hardsigmoid);
      ("hardswish", unary ~shape:nhwc Builder.hardswish);
      ("sqrt", unary ~shape:nhwc Builder.sqrt);
      ("max_pool2d", unary ~shape:nhwc (Builder.max_pool2d pool_params));
      ( "adaptive_avg_pool2d",
        unary ~shape:nhwc (Builder.adaptive_avg_pool2d adaptive_params) );
      ("avg_pool2d", unary ~shape:nhwc (Builder.avg_pool2d avg_params));
      ( "mean_keepdims",
        unary ~shape:nhwc (Builder.mean_keepdims [ Axis4.H; Axis4.W ]) );
      ( "max_keepdims",
        unary ~shape:nhwc (Builder.max_keepdims [ Axis4.H; Axis4.W ]) );
      ( "sum_keepdims",
        unary ~shape:nhwc (Builder.sum_keepdims [ Axis4.H; Axis4.W ]) );
      (* Both signs on two different axes, so the fixture covers padding and
         cropping in one graph. Constant rather than reflect because that is the
         mode whose pixel map goes through [select] — the arm the [Symbolic]
         instance stages rather than evaluates, and so the one this
         direct-versus-symbolic comparison is actually about. Reflect's mirror
         is pinned by hand values above. *)
      ( "pad4",
        unary ~shape:nhwc
          (Builder.pad4
             {
               Ops4.Pad4.pads =
                 [
                   (Axis4.H, { Pad.Pad.before = 1; after = -1 });
                   (Axis4.W, { Pad.Pad.before = 0; after = 2 });
                 ];
               mode = Pad.Pad.Constant 0.25;
             }) );
      (* A non-unit step on a NON-square input, so a fixture that read the wrong
         axis differs in shape as well as in values. *)
      ( "slice4",
        unary ~shape:nhwc
          (Builder.slice4
             {
               Ops4.Slice4.axis = Axis4.W;
               start = 1;
               stop = 4;
               step = Op_config.Pos.of_int 2;
             }) );
      (* Three operands of DIFFERENT extents along the joined axis (W), so a
         fixture that only ever concatenated equal-sized pieces could not
         catch a wrong per-operand offset. *)
      ( "concat4",
        let a_shape = s4 ~n:1 ~h:2 ~w:1 ~c:2 in
        let b_shape = s4 ~n:1 ~h:2 ~w:2 ~c:2 in
        let c_shape = s4 ~n:1 ~h:2 ~w:3 ~c:2 in
        let g =
          build
            ~outputs:(fun o -> [ o ])
            (let open Builder in
             let* a = input ~shape:a_shape () in
             let* b = input ~shape:b_shape () in
             let* c = input ~shape:c_shape () in
             concat4 { Ops4.Concat4.axis = Axis4.W } [ a; b; c ])
        in
        (g, [ a_shape; b_shape; c_shape ]) );
      ( "permute4",
        unary ~shape:nhwc
          (Builder.permute4
             (Ops4.Permute4.of_fn (function H -> W | W -> H | a -> a))) );
      ("reshape4", unary ~shape:nhwc (Builder.reshape4 (s4 ~n:1 ~h:2 ~w:2 ~c:8)));
      ( "rms_norm",
        unary ~shape:nhwc (fun x ->
            Builder.rms_norm
              { Ops4.Rms_norm.dims = [ Axis4.C ]; eps = 1e-5 }
              ~x ()) );
      (* Both affine operands PRESENT, unlike the rms_norm fixture above: the
         Direct-vs-Symbolic sweep runs the same functor on both sides, so what
         it can see is the operand wiring and the reduction structure, and an
         absent operand is filled with a constant that exercises neither. *)
      ( "layer_norm",
        let g =
          build
            ~outputs:(fun o -> [ o ])
            (let open Builder in
             let* x = input ~shape:nhwc () in
             let* w = constant ~shape:(s4 ~n:1 ~h:1 ~w:1 ~c:2) () in
             let* b = constant ~shape:(s4 ~n:1 ~h:1 ~w:1 ~c:2) () in
             layer_norm4
               { Ops4.Layer_norm.dims = [ Axis4.C ]; eps = 1e-5 }
               ~x ~weight:w ~bias:b ())
        in
        (g, [ nhwc; s4 ~n:1 ~h:1 ~w:1 ~c:2; s4 ~n:1 ~h:1 ~w:1 ~c:2 ]) );
      ( "conv2d",
        let g =
          build
            ~outputs:(fun o -> [ o ])
            (let open Builder in
             let* x = input ~shape:nhwc () in
             let* w = constant ~shape:(s4 ~n:3 ~h:1 ~w:1 ~c:2) () in
             conv2d (conv_params ~in_channels:2 ~kernel:1) ~x ~weight:w ())
        in
        (g, [ nhwc; s4 ~n:3 ~h:1 ~w:1 ~c:2 ]) );
      ( "depthwise_conv2d",
        let g =
          build
            ~outputs:(fun o -> [ o ])
            (let open Builder in
             let* x = input ~shape:nhwc () in
             let* w = constant ~shape:(s4 ~n:2 ~h:1 ~w:1 ~c:1) () in
             depthwise_conv2d
               (conv_params ~in_channels:2 ~kernel:1)
               ~x ~weight:w ())
        in
        (g, [ nhwc; s4 ~n:2 ~h:1 ~w:1 ~c:1 ]) );
      ( "transposed_conv2d",
        let g =
          build
            ~outputs:(fun o -> [ o ])
            (let open Builder in
             let* x = input ~shape:flat () in
             (* transposed weight is [Cin,1,1,Kh,Kw,Cout] *)
             let* w = constant ~shape:(s4 ~n:4 ~h:1 ~w:1 ~c:3) () in
             transposed_conv2d transposed_params ~x ~weight:w ())
        in
        (g, [ flat; s4 ~n:4 ~h:1 ~w:1 ~c:3 ]) );
      (* The one multi-output fixture, and the reason the harness compares every
         output rather than [List.hd]. Unbinding C on a [1,4,4,2] input stays
         inside the dialect: dropping the innermost axis shifts every axis
         outside it one place inward, and N/T/D are unit either way. Two slices,
         so a bug that only gets ordinal 0 right cannot pass. *)
      ( "unbind",
        let g =
          build ~outputs:Fun.id
            (let open Builder in
             let* x = input ~shape:nhwc () in
             unbind Axis4.C x)
        in
        (g, [ nhwc ]) );
      ( "upsample_bilinear2d",
        unary ~shape:nhwc (Builder.upsample_bilinear2d upsample_params) );
      ( "upsample_nearest2d",
        unary ~shape:nhwc (Builder.upsample_nearest2d nearest_params) );
      ( "vector_norm_keepdims",
        unary ~shape:nhwc (Builder.vector_norm_keepdims [ Axis4.H; Axis4.W ]) );
    ]
  in
  List.map
    (fun (name, (g, shapes)) ->
      let inputs, constants = split g (bindings g shapes) in
      (name, g, inputs, constants))
    cases
