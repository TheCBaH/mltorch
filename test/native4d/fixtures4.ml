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

(* Same non-trivial 4-in/3-out ratio as [adaptive_params] (uneven bin widths:
   some bins 1 wide, some 2), so the max-reduce nest and its index derivation
   get a real multi-element window rather than a degenerate 1:1 identity. *)
let adaptive_max_params : Pool.AdaptiveMaxPool2d.params =
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

let grouped_conv_params ~in_channels ~groups ~kernel :
    Ops4.Grouped_conv_params.t =
  {
    h = axis_window ~kernel;
    w = axis_window ~kernel;
    in_channels = Dim.extent in_channels;
    groups = Op_config.Pos.of_int groups;
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

(* [IndexTensor4]'s own entry, built by hand rather than through [unary]/
   [binary]: [index] must be genuinely I64-formatted for
   [Tensor.read_i64_at6] to accept it at all ([Direct.load_index]'s own
   contract), which [bindings]' uniform F32 [seq] cannot supply. Gathering
   along W (self extent 4) with three in-range positions, so a fixture
   reading the wrong element would still differ. *)
let index_tensor4_case () =
  let self_shape = nhwc in
  let index_shape = s4 ~n:1 ~h:1 ~w:1 ~c:3 in
  let g =
    build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* self = input ~shape:self_shape () in
       let* index =
         input ~shape:index_shape ~fmt:(Payload.Fmt Payload.I64) ()
       in
       index_tensor4 { Ops4.IndexTensor4.axis = Axis4.W } ~self ~index)
  in
  let self_id, index_id =
    match g.Graph.Graph.inputs with
    | [ a; b ] -> (a, b)
    | _ -> invalid_arg "index_tensor4_case: expected two inputs"
  in
  let index_values =
    Tensor.materialize_i64 (Shape4.to_vec6 index_shape) (fun c ->
        Int64.of_int (Dim.to_int c.Vec6.c))
  in
  ( "index_tensor4",
    g,
    [ (self_id, seq self_shape); (index_id, index_values) ],
    [] )

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
      (* Two DIFFERENT fanned axes (H and C), so a fixture that only ever
         broadcast one axis could not catch a wrong per-axis broadcast
         coordinate. *)
      ( "expand4",
        let x_shape = s4 ~n:1 ~h:1 ~w:4 ~c:1 in
        let target = s4 ~n:1 ~h:3 ~w:4 ~c:5 in
        let g =
          build
            ~outputs:(fun o -> [ o ])
            (let open Builder in
             let* x = input ~shape:x_shape () in
             expand4 target x)
        in
        (g, [ x_shape ]) );
      ("mul_scalar", unary ~shape:nhwc (Builder.mul_scalar 2.));
      ("pow", unary ~shape:nhwc (Builder.pow 2.));
      ( "rsub_scalar",
        unary ~shape:nhwc
          (Builder.rsub_scalar { Pointwise.Rsub_scalar.other = 1.; alpha = 2. })
      );
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
      ( "eye4",
        ( build
            ~outputs:(fun y -> [ y ])
            (Builder.eye4
               {
                 Ops4.Eye4.shape = s4 ~n:1 ~h:1 ~w:2 ~c:3;
                 fmt = Payload.Fmt Payload.F32;
               }),
          [] ) );
      ( "batch_norm_no_stats",
        ( build ~outputs:Fun.id
            (let open Builder in
             let* x = input ~shape:nhwc () in
             let c_shape = s4 ~n:1 ~h:1 ~w:1 ~c:2 in
             let* weight = input ~shape:c_shape () in
             let* bias = input ~shape:c_shape () in
             batch_norm_no_stats
               { Ops4.Batch_norm_no_stats.channel = Axis4.C; eps = 1e-5 }
               ~x ~weight ~bias ()),
          [ nhwc; s4 ~n:1 ~h:1 ~w:1 ~c:2; s4 ~n:1 ~h:1 ~w:1 ~c:2 ] ) );
      (* [H=2] on both operands (the corpus's real batch axis,
         `.ai/matmul_softmax_design.md` §5) and a contraction extent (3)
         distinct from either row/column count, so a fixture reading the
         wrong axis as the contraction would still differ. *)
      ( "batched_matmul",
        let a_shape = s4 ~n:1 ~h:2 ~w:2 ~c:3 in
        let b_shape = s4 ~n:1 ~h:2 ~w:3 ~c:4 in
        let g =
          build
            ~outputs:(fun o -> [ o ])
            (let open Builder in
             let* a = input ~shape:a_shape () in
             let* b = input ~shape:b_shape () in
             batched_matmul a b)
        in
        (g, [ a_shape; b_shape ]) );
      ("relu", unary ~shape:nhwc Builder.relu);
      ("repeat4", unary ~shape:nhwc (Builder.repeat4 (s4 ~n:1 ~h:2 ~w:1 ~c:3)));
      ( "repeat_interleave4",
        unary ~shape:nhwc
          (Builder.repeat_interleave4 Axis4.W (Op_config.Pos.of_int 3)) );
      ("gelu", unary ~shape:nhwc (Builder.gelu Pointwise.Gelu.Exact));
      ("sigmoid", unary ~shape:nhwc Builder.sigmoid);
      ("silu", unary ~shape:nhwc Builder.silu);
      ("hardsigmoid", unary ~shape:nhwc Builder.hardsigmoid);
      ("hardswish", unary ~shape:nhwc Builder.hardswish);
      ("sqrt", unary ~shape:nhwc Builder.sqrt);
      ("to_copy", unary ~shape:nhwc (Builder.to_copy Pointwise.To_copy.Long));
      ("max_pool2d", unary ~shape:nhwc (Builder.max_pool2d pool_params));
      ( "adaptive_avg_pool2d",
        unary ~shape:nhwc (Builder.adaptive_avg_pool2d adaptive_params) );
      ( "adaptive_max_pool2d",
        unary ~shape:nhwc (Builder.adaptive_max_pool2d adaptive_max_params) );
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
      (* Softmax reduces over W without changing shape -- unlike [slice4]
         above, the fixture only needs a non-trivial extent (4) on the
         reduced axis so the max/sum reduction is over more than one
         element. *)
      ( "softmax4",
        unary ~shape:nhwc (Builder.softmax4 { Ops4.Softmax4.axis = Axis4.W }) );
      (* Same shape-preserving reduction shape as [softmax4] just above, and
         the same requirement on the fixture: a non-unit extent (4) on the
         walked axis, so the running sum spans more than one element. *)
      ( "cumsum4",
        unary ~shape:nhwc
          (Builder.cumsum4 { Ops4_cumsum.Cumsum4.axis = Axis4.W }) );
      (* N=1, the same precondition [unbind]'s own fixture comment gives:
         dropping W leaves N/T/D unit either way, so the result stays
         four-axis. An index that is neither 0 nor the axis's last valid one,
         so a fixture reading the wrong element would still differ. *)
      ( "select4",
        unary ~shape:nhwc
          (Builder.select4 { Ops4.Select4.axis = Axis4.W; index = 2 }) );
      (* [self] and [src] are DIFFERENT shapes (unlike [binary]'s pairs):
         [src] is the shape [Select4] itself would produce at this
         axis/index -- dropping W repacks the surviving N/H onto T/H,
         right-aligned, so [src]'s own W holds what was [self]'s H (4) and
         its H is the vacated unit slot -- so a fixture checking the wrong
         operand's shape would fail to build at all. *)
      ( "select_scatter4",
        let self_shape = nhwc in
        let src_shape = s4 ~n:1 ~h:1 ~w:4 ~c:2 in
        let g =
          build
            ~outputs:(fun o -> [ o ])
            (let open Builder in
             let* self = input ~shape:self_shape () in
             let* src = input ~shape:src_shape () in
             select_scatter4
               { Ops4.Select_scatter4.axis = Axis4.W; index = 1 }
               ~self ~src)
        in
        (g, [ self_shape; src_shape ]) );
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
      (* Two operands of the SAME shape (unlike [concat4] above, [Stack]
         cannot differ along the joined axis -- it names a brand-new one), so
         a fixture reading the wrong operand by index would still differ.
         H=1, the same precondition [stack_h_batch1] documents: inserting at
         H, W, or C alike shifts H's own extent onto D, so H=1 keeps the
         result four-axis. *)
      ( "stack4",
        let x_shape = s4 ~n:1 ~h:1 ~w:2 ~c:3 in
        let g =
          build
            ~outputs:(fun o -> [ o ])
            (let open Builder in
             let* a = input ~shape:x_shape () in
             let* b = input ~shape:x_shape () in
             stack4 { Ops4.Stack4.axis = Axis4.W } [ a; b ])
        in
        (g, [ x_shape; x_shape ]) );
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
      (* [Wq=2 <> Wk=3], so a fixture that only ever squared the sequence
         length could not catch a query/key extent confused in [score_shape].
         The mask broadcasts on H and Wq (its own H and W are both 1) and is
         exact on Wk (its C matches key's W) -- the same broadcast
         [Region_context.broadcast_coord] exists for, so a fixture with a
         full-shaped mask could not exercise it. *)
      ( "sdpa",
        let q_shape = s4 ~n:1 ~h:2 ~w:2 ~c:3 in
        let k_shape = s4 ~n:1 ~h:2 ~w:3 ~c:3 in
        let v_shape = s4 ~n:1 ~h:2 ~w:3 ~c:3 in
        let mask_shape = s4 ~n:1 ~h:1 ~w:1 ~c:3 in
        let g =
          build
            ~outputs:(fun o -> [ o ])
            (let open Builder in
             let* query = input ~shape:q_shape () in
             let* key = input ~shape:k_shape () in
             let* value = input ~shape:v_shape () in
             let* mask = input ~shape:mask_shape () in
             sdpa
               { Attention.Sdpa.scale = Attention.Sdpa.Scale.Default }
               ~query ~key ~value ~mask ())
        in
        (g, [ q_shape; k_shape; v_shape; mask_shape ]) );
      (* [groups=2] on [C=4]: two groups of two channels each, so a wrong
         group window (reducing over every channel, or one channel alone)
         would be visible against [layer_norm]/[rms_norm]'s whole-channel
         reductions above. *)
      ( "group_norm4",
        let x_shape = s4 ~n:1 ~h:4 ~w:4 ~c:4 in
        let c_shape = s4 ~n:1 ~h:1 ~w:1 ~c:4 in
        let g =
          build
            ~outputs:(fun o -> [ o ])
            (let open Builder in
             let* x = input ~shape:x_shape () in
             let* w = constant ~shape:c_shape () in
             let* b = constant ~shape:c_shape () in
             group_norm4
               {
                 Ops4.Group_norm4.channel = Axis4.C;
                 groups = Op_config.Pos.of_int 2;
                 eps = 1e-5;
               }
               ~x ~weight:w ~bias:b ())
        in
        (g, [ x_shape; c_shape; c_shape ]) );
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
      (* [in_channels=4], [groups=2]: 2 input channels per group, so distinct
         from both [conv2d] (groups=1) and [depthwise_conv2d] (1 per group)
         above -- the general form neither of those constructors can carry. *)
      ( "grouped_conv2d",
        let x_shape = s4 ~n:1 ~h:4 ~w:4 ~c:4 in
        let g =
          build
            ~outputs:(fun o -> [ o ])
            (let open Builder in
             let* x = input ~shape:x_shape () in
             let* w = constant ~shape:(s4 ~n:4 ~h:1 ~w:1 ~c:2) () in
             grouped_conv2d
               (grouped_conv_params ~in_channels:4 ~groups:2 ~kernel:1)
               ~x ~weight:w ())
        in
        (g, [ x_shape; s4 ~n:4 ~h:1 ~w:1 ~c:2 ]) );
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
      (* One of the two multi-output fixtures, and the reason the harness
         compares every output rather than [List.hd]. Unbinding C on a
         [1,4,4,2] input stays inside the dialect: dropping the innermost
         axis shifts every axis outside it one place inward, and N/T/D are
         unit either way. Two slices, so a bug that only gets ordinal 0
         right cannot pass. *)
      ( "unbind",
        let g =
          build ~outputs:Fun.id
            (let open Builder in
             let* x = input ~shape:nhwc () in
             unbind Axis4.C x)
        in
        (g, [ nhwc ]) );
      (* The other multi-output fixture: unequal sizes on a KEPT axis (W stays
         rank-4, unlike [unbind]'s dropped C), so a fixture built only from
         equal pieces could not catch a wrong per-piece offset. *)
      ( "split_with_sizes4",
        let g =
          build ~outputs:Fun.id
            (let open Builder in
             let* x = input ~shape:nhwc () in
             split_with_sizes4 Axis4.W [ 1; 3 ] x)
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
  @ [ index_tensor4_case () ]
