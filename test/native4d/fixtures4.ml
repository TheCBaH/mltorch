(* Native4D graphs, one per op, with bindings for every graph input.

   The per-op list is what makes the stage-3 acceptance criterion a real
   criterion rather than a spot check: nineteen ops, nineteen graphs, and the
   count is asserted against the op registry the same way op_json_test.ml
   asserts its samples. An op whose evaluation nobody exercised is an op whose
   Direct and Symbolic paths have never been compared. *)

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
    kernel = hw (Dim.extent 2);
    stride = hw (Op_config.Pos.of_int 2);
    pad = hw (Op_config.Nonneg.of_int 0);
  }

let avg_params : Pool.AvgPool2d.params =
  {
    kernel = hw (Dim.extent 2);
    stride = hw (Op_config.Pos.of_int 2);
    pad = hw (Op_config.Nonneg.of_int 0);
  }

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
  |> Core.or_raise (fun ppf e -> Fmt.pf ppf "fixture4: %a" Builder.pp_error e)

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
      ( "clamp",
        unary ~shape:nhwc
          (Builder.clamp { Pointwise.Clamp.min = Some 0.5; max = Some 2. }) );
      ( "hardtanh",
        unary ~shape:nhwc
          (Builder.hardtanh { Pointwise.Hardtanh.min_val = 0.; max_val = 1. })
      );
      ("relu", unary ~shape:nhwc Builder.relu);
      ("sqrt", unary ~shape:nhwc Builder.sqrt);
      ("max_pool2d", unary ~shape:nhwc (Builder.max_pool2d pool_params));
      ("avg_pool2d", unary ~shape:nhwc (Builder.avg_pool2d avg_params));
      ( "mean_keepdims",
        unary ~shape:nhwc (Builder.mean_keepdims [ Axis4.H; Axis4.W ]) );
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
    ]
  in
  List.map
    (fun (name, (g, shapes)) ->
      let inputs, constants = split g (bindings g shapes) in
      (name, g, inputs, constants))
    cases
