(* The Native4D legalization engine: the Native->Native4D parameter-translation
   helpers and [lower_node], the per-source-node dispatch. Split out of
   lower.ml under the tracked file-size ceiling; lower.ml keeps the public
   surface ([t], [convert], [evaluate], ...) and calls into [lower_node] here
   for the per-node work. See lower.mli for the module this feeds.

   The walk accumulator ([acc]) and its low-level operations ([resolve],
   [fresh_tensor], [fresh_constant], [emit]) live in lower_engine_acc.ml. *)

open Lower_engine_acc
open Graph_ir
open Err.Syntax

(* ---- parameter translation ------------------------------------------------ *)

let conv_params (p : Conv.Conv2d.params) : Ops4.Conv_params.t =
  { h = p.h; w = p.w; in_channels = p.in_channels }

(* Grouping becomes a constructor. §7.2/§8: one group is [Conv2D], one input
   channel per group is [DepthwiseConv2D], and every other count is
   [GroupedConv2D] — the general form, which needs [groups] itself since
   neither of the other two constructors carries it. *)
let forward_conv ~node:_ ~params ~x ~weight ~bias ~weight_shape =
  let groups = (params.Conv.Conv2d.groups :> int) in
  let payload =
    { Ops4.Conv_payload.params = conv_params params; x; weight; bias }
  in
  if groups = 1 then Err.return (Op.Conv2d payload)
  else if Dim.to_int (Vec6.get weight_shape Axis.C) = 1 then
    Err.return (Op.Depthwise_conv2d payload)
  else
    let grouped_params =
      {
        Ops4.Grouped_conv_params.h = params.Conv.Conv2d.h;
        w = params.Conv.Conv2d.w;
        in_channels = params.Conv.Conv2d.in_channels;
        groups = params.Conv.Conv2d.groups;
      }
    in
    Err.return
      (Op.Grouped_conv2d
         { Ops4.Grouped_conv_payload.params = grouped_params; x; weight; bias })

let perm4_of_native ~node (perm : Permute.Permute.perm) =
  Err.List.map
    (fun out ->
      let in_axis = Permute.Permute.lookup perm (Axis4.to_axis out) in
      match Axis4.of_axis in_axis with
      | Some a -> Err.return (out, a)
      | None -> Err.fail (`Axis_outside_dialect (node, in_axis)))
    Axis4.all

let dims4 ~node dims =
  Err.List.map
    (fun axis ->
      Axis4.of_axis axis |> Err.of_option (`Axis_outside_dialect (node, axis)))
    dims

let shape4 ~id shape =
  match Shape4.of_vec6 shape with
  | Ok s -> Err.return s
  | Error _ -> Err.fail (`Non_four_dimensional_tensor (id, shape))

let unit_conv_params ~in_channels : Ops4.Conv_params.t =
  {
    h = Conv.Conv2d.unit_window;
    w = Conv.Conv2d.unit_window;
    in_channels = Dim.extent in_channels;
  }

(* ---- one source node ------------------------------------------------------ *)

(* Every arm resolves its operands through [subst] first, which is how clone
   removal reaches consumers: the clone contributes no node and instead records
   "my output means my input", so every later reference rewires. *)
let lower_node ~view acc (n : node) =
  let node = n.Node.id in
  let op_of = resolve acc in
  let sig_of id =
    match Graph_view.sig_of view id with
    | Some sg -> Err.return sg.Tensor_sig.shape
    | None ->
        Err.fail
          (`Non_four_dimensional_tensor
             (id, Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1))
  in
  (* An INVARIANT, not a diagnostic. [Graph_view] has already checked every
     node's arity against shape inference, and [Domain.check] has already run,
     so a single-output arm meeting anything else means those two disagree with
     this match — our bug, not a model we cannot represent.

     It must not be `Unsupported_op`: [Me_classify.native4d] maps that to
     [Unavailable Outside_dialect_domain], which tells a user to change their
     model to work around a lowering defect. Same treatment as
     [Native_interp.add_env] gives its own module invariants.

     Lazy, unlike the [List.hd] it replaces: that was computed before the match
     and so would have raised on a zero-output [Discard] node before reaching
     the arm that rejects one. *)
  let single () =
    match n.Node.outputs with
    | [ o ] -> o
    | outputs ->
        invalid_arg
          (Format.asprintf
             "Native4d.Lower: %a is a single-output op but declares %d outputs"
             Node_id.pp node (List.length outputs))
  in
  let simple op = Err.return (emit acc ~from:node op [ single () ]) in
  match n.Node.op with
  (* §7.1 direct counterparts. The payload records are Native's own, reused
     unchanged — they name no axis and carry no shape. *)
  | Add { Pointwise.Bin.a; b } ->
      simple (Op.Add { Pointwise.Bin.a = op_of a; b = op_of b })
  | Addcmul { Pointwise.Addcmul.self; tensor1; tensor2; value } ->
      simple
        (Op.Addcmul
           {
             Pointwise.Addcmul.self = op_of self;
             tensor1 = op_of tensor1;
             tensor2 = op_of tensor2;
             value;
           })
  | Sub { Pointwise.Bin.a; b } ->
      simple (Op.Sub { Pointwise.Bin.a = op_of a; b = op_of b })
  | Mul { Pointwise.Bin.a; b } ->
      simple (Op.Mul { Pointwise.Bin.a = op_of a; b = op_of b })
  | Div { Pointwise.Bin.a; b } ->
      simple (Op.Div { Pointwise.Bin.a = op_of a; b = op_of b })
  | Add_scalar { Pointwise.Scalar_bin.x; scalar } ->
      simple (Op.Add_scalar { Pointwise.Scalar_bin.x = op_of x; scalar })
  | Div_scalar { Pointwise.Scalar_bin.x; scalar } ->
      simple (Op.Div_scalar { Pointwise.Scalar_bin.x = op_of x; scalar })
  | Floor_div_scalar { Pointwise.Scalar_bin.x; scalar } ->
      simple (Op.Floor_div_scalar { Pointwise.Scalar_bin.x = op_of x; scalar })
  | Mul_scalar { Pointwise.Scalar_bin.x; scalar } ->
      simple (Op.Mul_scalar { Pointwise.Scalar_bin.x = op_of x; scalar })
  | Pow { Pointwise.Scalar_bin.x; scalar } ->
      simple (Op.Pow { Pointwise.Scalar_bin.x = op_of x; scalar })
  | Rpow_scalar { Pointwise.Scalar_bin.x; scalar } ->
      simple (Op.Rpow_scalar { Pointwise.Scalar_bin.x = op_of x; scalar })
  | Rsub_scalar { Pointwise.Rsub_scalar.params; x } ->
      simple (Op.Rsub_scalar { Pointwise.Rsub_scalar.params; x = op_of x })
  | Bitwise_not { Pointwise.Bitwise_not.x } ->
      simple (Op.Bitwise_not { Pointwise.Bitwise_not.x = op_of x })
  | Clamp { Pointwise.Clamp.params; x } ->
      simple (Op.Clamp { Pointwise.Clamp.params; x = op_of x })
  | Col2im { Im2col.Col2im.params; x } ->
      simple (Op.Col2im { Im2col.Col2im.params; x = op_of x })
  | Cos { Pointwise.Cos.x } -> simple (Op.Cos { Pointwise.Cos.x = op_of x })
  | Gelu { Pointwise.Gelu.x; approximate } ->
      simple (Op.Gelu { Pointwise.Gelu.x = op_of x; approximate })
  | Hardsigmoid { Pointwise.Hardsigmoid.x } ->
      simple (Op.Hardsigmoid { Pointwise.Hardsigmoid.x = op_of x })
  | Hardswish { Pointwise.Hardswish.x } ->
      simple (Op.Hardswish { Pointwise.Hardswish.x = op_of x })
  | Hardtanh { Pointwise.Hardtanh.params; x } ->
      simple (Op.Hardtanh { Pointwise.Hardtanh.params; x = op_of x })
  | Im2col { Im2col.Im2col.params; x } ->
      simple (Op.Im2col { Im2col.Im2col.params; x = op_of x })
  | Leaky_relu { Pointwise.Leaky_relu.params; x } ->
      simple (Op.Leaky_relu { Pointwise.Leaky_relu.params; x = op_of x })
  | Arange { Factory.Arange.params } ->
      let* source = sig_of (single ()) in
      let+ _ = shape4 ~id:(single ()) source in
      emit acc ~from:node
        (Op.Arange4
           {
             Ops4.Arange4.params =
               {
                 start = params.start;
                 stop = params.stop;
                 step = params.step;
                 fmt = params.fmt;
               };
           })
        [ single () ]
  | Zeros { Factory.Zeros.params } ->
      let* shape = sig_of (single ()) in
      let+ shape = shape4 ~id:(single ()) shape in
      emit acc ~from:node
        (Op.Zeros4 { Ops4.Zeros4.params = { shape; fmt = params.fmt } })
        [ single () ]
  | Eye { Factory.Eye.params } ->
      let* shape = sig_of (single ()) in
      let+ shape = shape4 ~id:(single ()) shape in
      emit acc ~from:node
        (Op.Eye4 { Ops4.Eye4.params = { shape; fmt = params.fmt } })
        [ single () ]
  | Relu { Pointwise.Relu.x } -> simple (Op.Relu { Pointwise.Relu.x = op_of x })
  (* A direct counterpart, the same shape [Expand]'s own arm has: [repeats]
     converts through [shape4] exactly like [Expand]'s [size] does, and
     needs no post-hoc output re-check the way [Select]/[Stack] do, because
     [Repeat] neither drops nor inserts an axis -- every axis keeps its own
     identity (repeat.ml's own doc comment), so a T/D-unit [repeats] and a
     T/D-unit input [x] compose to a T/D-unit output automatically. *)
  | Repeat { Repeat.Repeat.params; x } ->
      let* repeats = shape4 ~id:(single ()) params.Repeat.Repeat.repeats in
      simple (Op.Repeat4 { Ops4.Repeat4.params = { repeats }; x = op_of x })
  (* A direct counterpart with only the axis KEY converted, the same reason
     [Select]'s own arm converts its axis -- so [Domain]'s
     [Axis_outside_dialect] can still name the rejected Native axis. No
     post-hoc output re-check needed, the same reasoning [Repeat4]'s arm
     above gives: [RepeatInterleave] multiplies its named axis's extent in
     place rather than dropping or inserting one, so nothing shifts into
     T/D. *)
  | RepeatInterleave { Repeat.RepeatInterleave.params; x } ->
      let* axis4 = dims4 ~node [ params.Repeat.RepeatInterleave.axis ] in
      simple
        (Op.RepeatInterleave4
           {
             Ops4.RepeatInterleave4.params =
               {
                 axis = List.hd axis4;
                 repeats = params.Repeat.RepeatInterleave.repeats;
               };
             x = op_of x;
           })
  | Sigmoid { Pointwise.Sigmoid.x } ->
      simple (Op.Sigmoid { Pointwise.Sigmoid.x = op_of x })
  | Silu { Pointwise.Silu.x } -> simple (Op.Silu { Pointwise.Silu.x = op_of x })
  | Sin { Pointwise.Sin.x } -> simple (Op.Sin { Pointwise.Sin.x = op_of x })
  | Sqrt { Pointwise.Sqrt.x } -> simple (Op.Sqrt { Pointwise.Sqrt.x = op_of x })
  | To_copy { Pointwise.To_copy.target; x } ->
      simple (Op.To_copy { Pointwise.To_copy.target; x = op_of x })
  | Avg_pool2d { Pool.AvgPool2d.params; x } ->
      simple (Op.Avg_pool2d { Pool.AvgPool2d.params; x = op_of x })
  | Adaptive_avg_pool2d { Pool.AdaptiveAvgPool2d.params; x } ->
      simple
        (Op.Adaptive_avg_pool2d { Pool.AdaptiveAvgPool2d.params; x = op_of x })
  | Adaptive_max_pool2d { Pool.AdaptiveMaxPool2d.params; x } ->
      simple
        (Op.Adaptive_max_pool2d { Pool.AdaptiveMaxPool2d.params; x = op_of x })
  (* Two outputs (value, indices), the same fixed arity
     [Batch_norm_no_stats] asserts above -- not [simple], which is
     single-output only. [params] names no axis and carries no shape, so it
     crosses unchanged like [Max_pool2d]'s own params do. *)
  | Adaptive_max_pool2d_with_indices
      { Pool.AdaptiveMaxPool2dWithIndices.params; x } ->
      let outputs =
        match n.Node.outputs with
        | [ _; _ ] as outputs -> outputs
        | outputs ->
            invalid_arg
              (Format.asprintf
                 "Native4d.Lower: %a is a two-output op but declares %d outputs"
                 Node_id.pp node (List.length outputs))
      in
      Err.return
        (emit acc ~from:node
           (Op.Adaptive_max_pool2d_with_indices
              { Pool.AdaptiveMaxPool2dWithIndices.params; x = op_of x })
           outputs)
  | Max_pool2d { Pool.MaxPool2d.params; x } ->
      simple (Op.Max_pool2d { Pool.MaxPool2d.params; x = op_of x })
  (* Same two-output shape as [Adaptive_max_pool2d_with_indices] above. *)
  | Max_pool2d_with_indices { Pool.MaxPool2dWithIndices.params; x } ->
      let outputs =
        match n.Node.outputs with
        | [ _; _ ] as outputs -> outputs
        | outputs ->
            invalid_arg
              (Format.asprintf
                 "Native4d.Lower: %a is a two-output op but declares %d outputs"
                 Node_id.pp node (List.length outputs))
      in
      Err.return
        (emit acc ~from:node
           (Op.Max_pool2d_with_indices
              { Pool.MaxPool2dWithIndices.params; x = op_of x })
           outputs)
  (* Output count tracks the operand count, not a fixed arity like
     [Max_pool2d_with_indices] above -- [Unbind]'s shape, not that one's.
     [used_axes ~rank] is exactly what [Domain.check_node]'s [check_dims]
     call already refused any T/D entry from, so [dims4] here cannot fail,
     for the same reason [Unbind]'s doesn't. Every output is checked against
     [Shape4] here too, for [Unbind]'s own reason: a dead output slice is
     invisible to [Domain.check_shapes], which only inspects live tensors. *)
  | Meshgrid { Meshgrid.Meshgrid.tensors } ->
      let* (_ : Axis4.t list) =
        dims4 ~node (Aten_shape.used_axes ~rank:(List.length tensors))
      in
      let+ () =
        Err.List.iter
          (fun o ->
            let* shape = sig_of o in
            let+ (_ : Shape4.t) = shape4 ~id:o shape in
            ())
          n.Node.outputs
      in
      emit acc ~from:node
        (Op.Meshgrid { Meshgrid.Meshgrid.tensors = List.map op_of tensors })
        n.Node.outputs
  (* Direct counterpart, like [Max_pool2d] above: [Resize.Bicubic2d.params]
     names no axis and carries no shape, so it crosses unchanged. *)
  | Upsample_bicubic2d { Resize.Bicubic2d.params; x } ->
      simple (Op.Upsample_bicubic2d { Resize.Bicubic2d.params; x = op_of x })
  (* Direct counterpart, like [Max_pool2d] above: [Resize.Bilinear2d.params]
     names no axis and carries no shape, so it crosses unchanged. *)
  | Upsample_bilinear2d { Resize.Bilinear2d.params; x } ->
      simple (Op.Upsample_bilinear2d { Resize.Bilinear2d.params; x = op_of x })
  (* Same reasoning as [Upsample_bilinear2d] just above: [Resize.Nearest2d.params]
     names no axis and carries no shape either. *)
  | Upsample_nearest2d { Resize.Nearest2d.params; x } ->
      simple (Op.Upsample_nearest2d { Resize.Nearest2d.params; x = op_of x })
  (* The axes were gated on the NATIVE [Axis.t] by [Domain.check_dims] before
     this walk started, so [dims4] here only converts what is already known to
     be inside the dialect -- which is what lets the diagnostic name the
     rejected axis instead of reporting "conversion failed". *)
  | Layer_norm { Norm.LayerNorm.params; x; weight; bias } ->
      let+ dims = dims4 ~node params.Norm.LayerNorm.dims in
      emit acc ~from:node
        (Op.Layer_norm
           {
             Ops4.Layer_norm.params = { dims; eps = params.Norm.LayerNorm.eps };
             x = op_of x;
             weight = Option.map op_of weight;
             bias = Option.map op_of bias;
           })
        [ single () ]
  | Rms_norm { Norm.RmsNorm.params; x; weight } ->
      let+ dims = dims4 ~node params.Norm.RmsNorm.dims in
      emit acc ~from:node
        (Op.Rms_norm
           {
             Ops4.Rms_norm.params = { dims; eps = params.Norm.RmsNorm.eps };
             x = op_of x;
             weight = Option.map op_of weight;
           })
        [ single () ]
  | Batch_norm_no_stats { Norm.BatchNormNoStats.params; x; weight; bias } ->
      let outputs =
        match n.Node.outputs with
        | [ _; _; _ ] as outputs -> outputs
        | outputs ->
            invalid_arg
              (Format.asprintf
                 "Native4d.Lower: %a is a three-output op but declares %d \
                  outputs"
                 Node_id.pp node (List.length outputs))
      in
      let* channel =
        Axis4.of_axis params.Norm.BatchNormNoStats.channel
        |> Err.of_option
             (`Axis_outside_dialect (node, params.Norm.BatchNormNoStats.channel))
      in
      Err.return
        (emit acc ~from:node
           (Op.Batch_norm_no_stats
              {
                Ops4.Batch_norm_no_stats.params =
                  { channel; eps = params.Norm.BatchNormNoStats.eps };
                x = op_of x;
                weight = Option.map op_of weight;
                bias = Option.map op_of bias;
              })
           outputs)
  (* Same channel-conversion shape as [Batch_norm_no_stats] just above --
     [Domain.check_node] has already gated it to C -- but single-output like
     [Layer_norm]/[Rms_norm], and [groups] crosses unchanged: it names no
     axis, so there is nothing for this arm to convert. *)
  | Group_norm { Norm.GroupNorm.params; x; weight; bias } ->
      let* channel =
        Axis4.of_axis params.Norm.GroupNorm.channel
        |> Err.of_option
             (`Axis_outside_dialect (node, params.Norm.GroupNorm.channel))
      in
      simple
        (Op.Group_norm4
           {
             Ops4.Group_norm4.params =
               {
                 channel;
                 groups = params.Norm.GroupNorm.groups;
                 eps = params.Norm.GroupNorm.eps;
               };
             x = op_of x;
             weight = Option.map op_of weight;
             bias = Option.map op_of bias;
           })
  (* §7.1: [Clone] is removed and its output tied to its input. No node, no
     fresh id — just a substitution, and a pair cluster recording that the two
     edges are the same value. *)
  | Clone { Pointwise.Clone.x } ->
      Err.return
        {
          acc with
          subst = Tensor_id.Map.add (single ()) (op_of x) acc.subst;
          (* Drop the signature too. Left in place the edge would exist in the
             destination universe, and naming it only as a cluster SOURCE would
             then be [Unpaired_src] — an id present in both graphs has to be
             named on both sides. *)
          tensors = Tensor_id.Map.remove (single ()) acc.tensors;
        }
  | Permute { Permute.Permute.perm; x } ->
      let+ perm = perm4_of_native ~node perm in
      emit acc ~from:node
        (Op.Permute4 { Ops4.Permute4.perm; x = op_of x })
        [ single () ]
  | Reshape { Reshape.Reshape.params; x } ->
      let* shape = shape4 ~id:(single ()) params.Reshape.Reshape.shape in
      simple (Op.Reshape4 { Ops4.Reshape4.params = { shape }; x = op_of x })
  (* Same shape as [Reshape] just above: the target is typed [Shape4.t], so a
     broadcast that fans an axis onto T or D is refused HERE, by [shape4],
     rather than by a domain-check arm -- [Domain.check_node] admits every
     [Expand] unconditionally, for the reason its own comment gives. *)
  | Expand { Pointwise.Expand.params; x } ->
      let* size = shape4 ~id:(single ()) params.Pointwise.Expand.size in
      simple (Op.Expand4 { Ops4.Expand4.params = { size }; x = op_of x })
  (* A direct counterpart with only the axis KEYS converted: the signed amounts
     and the mode cross unchanged, and [Eval_op4] runs the very same
     [Pad.Pad.Compute] functor over f32 values on both sides, so no value
     changes and the claim is [Identical].

     [Domain.check_node] has already refused T and D, so this conversion cannot
     fail — it is still written as a conversion rather than asserted away, for
     the reason the [Unbind] arm below gives: the domain check and this match
     disagreeing is a bug worth reporting, not worth raising on.

     The pairs are rebuilt in ONE traversal rather than mapping the axes and
     re-pairing them, so there is no second list whose length could drift from
     the entries it is paired with. *)
  (* A direct counterpart with only the axis converted: the bounds are already
     canonical against the same extent on both sides, and [Eval_op4] runs the
     very same [Split.Slice.Compute], so no value changes and the claim is
     [Identical]. T and D have already been refused by [Domain.check_node], for
     the reason the [Unbind] arm below gives. *)
  | Slice { Split.Slice.params; x } ->
      let* axis = dims4 ~node [ params.axis ] in
      simple
        (Op.Slice4
           {
             Ops4.Slice4.params =
               {
                 axis = List.hd axis;
                 start = params.start;
                 stop = params.stop;
                 step = params.step;
               };
             x = op_of x;
           })
  (* A direct counterpart with only the axis converted: every operand's shape
     already agrees off the joined axis (Native's own shape rule), and
     [Eval_op4] runs the very same [Concat.Concat.Compute], so no value
     changes and the claim is [Identical]. T and D have already been refused
     by [Domain.check_node], for the reason the [Unbind] arm gives. *)
  | Concat { Concat.Concat.params; xs } ->
      let* axis = dims4 ~node [ params.axis ] in
      simple
        (Op.Concat4
           {
             Ops4.Concat4.params = { axis = List.hd axis };
             xs = List.map op_of xs;
           })
  | Pad { Pad.Pad.params; x } ->
      let* pads =
        Err.List.map
          (fun (axis, entry) ->
            let+ axis4 =
              Axis4.of_axis axis
              |> Err.of_option (`Axis_outside_dialect (node, axis))
            in
            (axis4, entry))
          params.Pad.Pad.pads
      in
      simple
        (Op.Pad4
           {
             Ops4.Pad4.params = { pads; mode = params.Pad.Pad.mode };
             x = op_of x;
           })
  (* §7.3. The Native weight is already [Out,1,1,1,1,In] — literally a 1x1
     convolution weight — so this is a params-only rewrite with no data
     movement, and the spatial singleton loops add no arithmetic, leaving the
     input-channel reduction order unchanged. Identical. *)
  | Linear { Linear.Linear.params; x; weight; bias } ->
      simple
        (Op.Conv2d
           {
             Ops4.Conv_payload.params =
               unit_conv_params
                 ~in_channels:(Dim.to_int params.Linear.Linear.in_features);
             x = op_of x;
             weight = op_of weight;
             bias = Option.map op_of bias;
           })
  (* [Conv1d]'s own H window is always [Conv2d.unit_window] by construction
     (conv_conv1d.ml), so translating through [Conv.Conv1d.to_conv2d_params]
     and reusing [forward_conv] unchanged is the SAME "map onto an existing
     op after translating parameters" legalization [Linear] gets above --
     Native4D gains no new op or payload for it, just another source of a
     [Conv2D]/[DepthwiseConv2D]/[GroupedConv2D] the dialect already has. *)
  | Conv1d { Conv.Conv1d.params; x; weight; bias } ->
      let* weight_shape = sig_of weight in
      let* op =
        forward_conv ~node
          ~params:(Conv.Conv1d.to_conv2d_params params)
          ~x:(op_of x) ~weight:(op_of weight) ~bias:(Option.map op_of bias)
          ~weight_shape
      in
      simple op
  | Conv2d { Conv.Conv2d.params; x; weight; bias } ->
      let* weight_shape = sig_of weight in
      let* op =
        forward_conv ~node ~params ~x:(op_of x) ~weight:(op_of weight)
          ~bias:(Option.map op_of bias) ~weight_shape
      in
      simple op
  | Conv2d_padding { Conv.Conv2d_padding.params; x; weight; bias } ->
      let* weight_shape = sig_of weight in
      (* "same"/"valid" resolve to explicit windows first, through Native's own
         translation — restating it here would be a second definition free to
         drift from the one the compute uses. *)
      let* params =
        Err.map_error
          (fun _ -> `Unsupported_op (node, n.Node.op))
          (Conv.Conv2d_padding.to_conv2d_params ~weight_shape params)
      in
      let* op =
        forward_conv ~node ~params ~x:(op_of x) ~weight:(op_of weight)
          ~bias:(Option.map op_of bias) ~weight_shape
      in
      simple op
  | Convolution { Conv.Convolution.params; x; weight; bias } ->
      let* weight_shape = sig_of weight in
      if params.Conv.Convolution.transposed then
        let groups = (params.Conv.Convolution.groups :> int) in
        if groups <> 1 then
          Err.fail (`Unsupported_grouped_transposed_conv (node, groups))
        else
          simple
            (Op.Transposed_conv2d
               {
                 Ops4.Transposed_conv2d.params =
                   {
                     stride = params.Conv.Convolution.stride;
                     padding = params.Conv.Convolution.padding;
                     dilation = params.Conv.Convolution.dilation;
                     output_padding = params.Conv.Convolution.output_padding;
                   };
                 x = op_of x;
                 weight = op_of weight;
                 bias = Option.map op_of bias;
               })
      else
        let* params =
          Err.map_error
            (fun _ -> `Unsupported_op (node, n.Node.op))
            (Conv.Convolution.to_conv2d_params ~weight_shape params)
        in
        let* op =
          forward_conv ~node ~params ~x:(op_of x) ~weight:(op_of weight)
            ~bias:(Option.map op_of bias) ~weight_shape
        in
        simple op
  (* The four retained reductions carry [keepdim] directly.  Their Native
     output shape still re-enters through [Graph_shape4.four], so a dropped
     axis that would put real extent on T/D is refused without a Reshape4. *)
  | Mean { Reduce.Mean.params; x } ->
      let* dims = dims4 ~node params.Reduce.Mean.dims in
      simple
        (Op.Mean_keepdims
           {
             Ops4.Mean_keepdims.params =
               { dims; keepdim = params.Reduce.Mean.keepdim };
             x = op_of x;
           })
  | Amax { Reduce.Amax.params; x } ->
      let* dims = dims4 ~node params.Reduce.Amax.dims in
      simple
        (Op.Max_keepdims
           {
             Ops4.Max_keepdims.params =
               { dims; keepdim = params.Reduce.Amax.keepdim };
             x = op_of x;
           })
  | Sum { Reduce.Sum.params; x } ->
      let* dims = dims4 ~node params.Reduce.Sum.dims in
      simple
        (Op.Sum_keepdims
           {
             Ops4.Sum_keepdims.params =
               { dims; keepdim = params.Reduce.Sum.keepdim };
             x = op_of x;
           })
  | Vector_norm { Reduce.Vector_norm.params; x } ->
      let* dims = dims4 ~node params.Reduce.Vector_norm.dims in
      simple
        (Op.Vector_norm_keepdims
           {
             Ops4.Vector_norm_keepdims.params =
               { dims; keepdim = params.Reduce.Vector_norm.keepdim };
             x = op_of x;
           })
  (* §7.4. [Bmm]'s shape is exactly [Batched_matmul]'s at N=T=D=1: both
     read [input]/[mat2] at the same coordinates once [mat2]'s N/T/D/H are
     read off the OUTPUT (as [Batched_matmul.Compute] does) rather than
     hard-coded to index 0 (as [Bmm.Compute] does) -- at extent 1 those agree
     bit-for-bit, so the claim stays [Identical] at any batch [H], not just
     [H = 1]. This retires the previous 1x1-convolution legalization (Permute4
     + Conv2d, sound only at batch 1 and only for an f32-exact [mat2] format,
     since it MATERIALIZED [mat2] through the permute where both ops here read
     it directly): a direct counterpart, one node, no relayout, no format
     restriction, no batch restriction beyond [Domain]'s existing D = 1.
     Native's own [Batched_matmul] payload names no axis and carries no shape
     and so crosses unchanged; [Bmm]'s payload has the same two fields and
     crosses the same way. *)
  | Bmm { Matmul.Bmm.input; mat2 }
  | Batched_matmul { Matmul.Batched_matmul.input; mat2 } ->
      simple
        (Op.Batched_matmul
           { Matmul.Batched_matmul.input = op_of input; mat2 = op_of mat2 })
  | Batch_norm
      { Norm.BatchNorm.params; x; weight; bias; running_mean; running_var } ->
      let* channel =
        Axis4.of_axis params.Norm.BatchNorm.channel
        |> Err.of_option
             (`Axis_outside_dialect (node, params.Norm.BatchNorm.channel))
      in
      simple
        (Op.Batch_norm
           {
             Ops4.Batch_norm.params =
               { channel; eps = params.Norm.BatchNorm.eps };
             x = op_of x;
             weight = Option.map op_of weight;
             bias = Option.map op_of bias;
             running_mean = op_of running_mean;
             running_var = op_of running_var;
           })
  (* Like [Meshgrid] above, a multi-output node whose count tracks the
     operand/rank rather than a fixed arity. The COMPLETE ordered output list
     is carried over unchanged, which is the whole of the correspondence
     work:
     under the id policy an edge whose value is preserved keeps its source id
     and so appears in no cluster, making every slice implicitly [Identical].
     Reordering or dropping one would be silent here — [Graph_map]'s output
     check is positional over the GRAPH's signature, not over a node's, so a
     swap inside this list is caught by [Map_verify] comparing the per-ordinal
     stages, and nowhere earlier.

     The axis converts here rather than in [Domain.check_node], so the domain's
     [Axis_outside_dialect] diagnostic can still name the rejected Native axis;
     by this point T/D have already been refused and [dims4] cannot fail. *)
  | Unbind { Split.Unbind.params; x } ->
      let* axis4 = dims4 ~node [ params.axis ] in
      (* Every slice, checked HERE. [Domain.check_shapes] only inspects tensors
         that are live, so an unbind whose slices are all dead reaches this arm
         unvalidated; without this it would fail later inside
         [Snapshot4.create] as [`View _], which [Me_classify.native4d] calls
         Fatal — reporting a graph outside the dialect as our own defect, and
         naming no node. [shape4] gives the accurate row and the tensor id. *)
      let+ () =
        Err.List.iter
          (fun o ->
            let* shape = sig_of o in
            let+ (_ : Shape4.t) = shape4 ~id:o shape in
            ())
          n.Node.outputs
      in
      emit acc ~from:node
        (Op.Unbind
           { Ops4.Unbind.params = { axis = List.hd axis4 }; x = op_of x })
        n.Node.outputs
  (* [Unbind]'s rank-preserving sibling: the axis converts here for the same
     reason (so [Domain]'s [Axis_outside_dialect] can still name the rejected
     Native axis), and every output is checked here for the same reason (a
     dead slice would otherwise reach [Snapshot4.create] unvalidated). [sizes]
     crosses unchanged -- Native has already bounded its length and proved it
     sums to the axis extent, so there is nothing left for this arm to
     recheck. *)
  | Split_with_sizes { Split.Split_with_sizes.params; x } ->
      let* axis4 = dims4 ~node [ params.axis ] in
      let+ () =
        Err.List.iter
          (fun o ->
            let* shape = sig_of o in
            let+ (_ : Shape4.t) = shape4 ~id:o shape in
            ())
          n.Node.outputs
      in
      emit acc ~from:node
        (Op.Split_with_sizes4
           {
             Ops4.Split_with_sizes4.params =
               {
                 axis = List.hd axis4;
                 sizes = params.Split.Split_with_sizes.sizes;
               };
             x = op_of x;
           })
        n.Node.outputs
  (* The axis converts here for the same reason [Unbind]'s does (so
     [Domain]'s [Axis_outside_dialect] can still name the rejected Native
     axis); the single output is checked here for the same reason [Unbind]'s
     are -- [Select] drops its axis, so the packed result re-enters the
     dialect only when [Shape4.of_vec6] accepts it, and a dead output would
     otherwise reach [Snapshot4.create] unvalidated. [index] crosses
     unchanged -- Native has already resolved it against the axis extent, so
     there is nothing left for this arm to recheck. *)
  | Select { Split.Select.params; x } ->
      let* axis4 = dims4 ~node [ params.axis ] in
      let* () =
        let o = single () in
        let* shape = sig_of o in
        let+ (_ : Shape4.t) = shape4 ~id:o shape in
        ()
      in
      simple
        (Op.Select4
           {
             Ops4.Select4.params =
               { axis = List.hd axis4; index = params.Split.Select.index };
             x = op_of x;
           })
  (* The axis converts here for the same reason [Select]'s does. No post-hoc
     output re-check, unlike [Select]'s: this op's output shape is [self]'s
     OWN shape unchanged (no drop, no repack -- [Split.Select_scatter]'s own
     [output_shape] returns [self_shape] verbatim), so if [self] is already
     four-axis the output automatically is too, whichever axis this op
     names. *)
  | Select_scatter { Split.Select_scatter.params; self; src } ->
      let* axis4 = dims4 ~node [ params.axis ] in
      simple
        (Op.Select_scatter4
           {
             Ops4.Select_scatter4.params =
               {
                 axis = List.hd axis4;
                 index = params.Split.Select_scatter.index;
               };
             self = op_of self;
             src = op_of src;
           })
  (* Output stays four-axis only for [index_rank = 1]; the generalization
     is unevidenced here, so rejected rather than re-derived. *)
  | Index_tensor { Index_tensor.Index_tensor.params; self; index } ->
      if params.Index_tensor.Index_tensor.index_rank <> 1 then
        Err.fail (`Unsupported_op (node, n.Node.op))
      else
        let* axis4 = dims4 ~node [ params.Index_tensor.Index_tensor.axis ] in
        simple
          (Op.IndexTensor4
             {
               Ops4.IndexTensor4.params = { axis = List.hd axis4 };
               self = op_of self;
               index = op_of index;
             })
  (* [Concat]'s variadic-operand handling above, plus [Select]'s post-hoc
     output check: [Stack] INSERTS an axis rather than keeping every one the
     way [Concat] does, so -- the same reason [Select]'s arm re-validates its
     single output -- the packed result re-enters the dialect only when
     [Shape4.of_vec6] accepts it. The axis converts here for the same reason
     [Select]'s does: so [Domain]'s [Axis_outside_dialect] can still name the
     rejected Native axis. *)
  | Stack { Concat.Stack.params; xs } ->
      let* axis4 = dims4 ~node [ params.axis ] in
      let* () =
        let o = single () in
        let* shape = sig_of o in
        let+ (_ : Shape4.t) = shape4 ~id:o shape in
        ()
      in
      simple
        (Op.Stack4
           {
             Ops4.Stack4.params = { axis = List.hd axis4 };
             xs = List.map op_of xs;
           })
  (* The axis converts here for the same reason [Select]'s/[Stack]'s does. No
     post-hoc output re-check, unlike theirs: this op's output shape is [x]'s
     OWN shape unchanged ([Reduce.Softmax.output_shape] returns [x_shape]
     verbatim, the same fact [Select_scatter]'s arm above relies on), so if
     [x] is already four-axis the output automatically is too, whichever axis
     this op reduces over. *)
  | Softmax { Reduce.Softmax.params; x } ->
      let* axis4 = dims4 ~node [ params.axis ] in
      simple
        (Op.Softmax4
           { Ops4.Softmax4.params = { axis = List.hd axis4 }; x = op_of x })
  (* The axis converts here for the same reason [Softmax]'s does just above:
     [Reduce.Cumsum.output_shape] also returns [x_shape] verbatim, so no
     post-hoc output re-check is needed either. *)
  | Cumsum { Reduce.Cumsum.params; x } ->
      let* axis4 = dims4 ~node [ params.axis ] in
      simple
        (Op.Cumsum4
           {
             Ops4_cumsum.Cumsum4.params = { axis = List.hd axis4 };
             x = op_of x;
           })
  (* Direct counterpart, once [Domain.check] has proved D = 1: [Attention.Sdpa.t]
     names no axis and carries no shape, so it crosses unchanged, and
     [Region_computation4]'s [native_op] routes it back through the exact same
     [Region_program] Native uses -- no second numeric kernel. *)
  | Sdpa { Attention.Sdpa.params; query; key; value; mask } ->
      simple
        (Op.Sdpa
           {
             Attention.Sdpa.params;
             query = op_of query;
             key = op_of key;
             value = op_of value;
             mask = Option.map op_of mask;
           })
  (* Direct counterpart, unconditionally: [Lstm.Lstm.t]'s shapes hardcode
     [T=1,D=1] everywhere (state/weight/bias/input shapes never touch T or
     D; [batch_first] only ever selects between [H]/[W], both dialect axes),
     so unlike [Sdpa] there is no [Domain.check_node] precondition to have
     proved here at all -- [check_shapes] alone already guarantees every
     Lstm tensor is four-axis. [map_operands] is reused rather than
     restated, so the nested [layers] traversal (forward/reverse directions,
     each with an optional bias pair) cannot drift from [Lstm.Lstm]'s own.
     Three outputs (output, h_n, c_n), the same [Batch_norm_no_stats]
     validation shape just above -- not [simple], which is single-output
     only. *)
  | Lstm t ->
      let outputs =
        match n.Node.outputs with
        | [ _; _; _ ] as outputs -> outputs
        | outputs ->
            invalid_arg
              (Format.asprintf
                 "Native4d.Lower: %a is a three-output op but declares %d \
                  outputs"
                 Node_id.pp node (List.length outputs))
      in
      Err.return
        (emit acc ~from:node (Op.Lstm (Lstm.Lstm.map_operands op_of t)) outputs)
  (* Rejected by [Domain.check] before the walk starts; reaching them means the
     domain check and this match disagree, which is a bug in one of them.
     [Conv3d]/[Unfold] are intrinsic axis boundaries, not missing
     counterparts. [Max_dim] is a missing counterpart (see [Domain]'s own
     comment) deferred for lack of a corpus need, not an intrinsic one.
     [Adaptive_max_pool2d_with_indices]/
     [Max_pool2d_with_indices]/[Repeat]/[RepeatInterleave]/[Select_scatter]/
     [Softmax]/[Batched_matmul]/[Sdpa]/[Index_tensor]/[Lstm]/[Meshgrid] no
     longer join them: all eleven now have real conversion arms above. *)
  | Conv3d _ | Discard _ | Max_dim _ | Unfold _ ->
      Err.fail (`Unsupported_op (node, n.Node.op))
