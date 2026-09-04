(* Shape dispatch for Native4D: the S-independent twin of [Eval_op4], and the
   Native4D counterpart of [Graph_shape].

   Every arm returns a [Shape4.t], which is the acceptance criterion
   .ai/native4d_plan.md sets for this stage — an inferred output that is not
   four-axis cannot escape into the IR. The arms themselves delegate to the
   NATIVE op's [output_shape]: the shape rules are identical, since the physical
   frame is identical (design §4.1), and restating them would be a second
   definition free to drift from the one the compute actually uses.

   So each arm is: unwrap operand signatures to [Vec6.shape], call Native, wrap
   the result back through [Shape4.of_vec6]. The wrap is not ceremony — it is
   where an op whose output leaves the dialect gets caught, which is how
   correction C1 (Mean keepdim=false putting the batch extent on D) is enforced
   without a special case anywhere. *)

open Op

type error =
  [ `Missing_tensor_sig of Tensor_id.t | Shape4.error | Shape_error.t ]

let pp_error fmt : [< error ] -> unit = function
  | `Missing_tensor_sig id -> Fmt.pf fmt "missing tensor sig %a" Tensor_id.pp id
  | #Shape4.error as e -> Shape4.pp_error fmt e
  | #Shape_error.t as e -> Shape_error.pp fmt e

let widen (r : ('a, [< error ]) Err.t) : ('a, error) Err.t =
  (r :> ('a, error) Err.t)

(* Native's window/broadcast rules answer in the six-axis frame; this is the one
   place the answer re-enters the dialect. *)
let four (r : (Vec6.shape, [< error ]) Err.t) =
  let open Err.Syntax in
  let* s = widen r in
  widen (Shape4.of_vec6 s)

(* The list form of [four], for the one op whose output count is not fixed by
   the op. Every inferred shape is re-entered into the dialect, not just the
   first: an unbind whose slices leave the four-axis domain does so for all of
   them, but checking one and assuming the rest is the kind of shortcut that
   holds until an op has genuinely differing outputs. *)
let four_all (r : (Vec6.shape list, [< error ]) Err.t) =
  let open Err.Syntax in
  let* shapes = widen r in
  Err.List.map (fun s -> four (Err.return s)) shapes

(* Grouping is a constructor here, not a parameter, so the Native params these
   build are the only place a [groups] value exists at all. *)
let conv2d_params (p : Ops4.Conv_params.t) ~groups : Conv.Conv2d.params =
  {
    h = p.Ops4.Conv_params.h;
    w = p.Ops4.Conv_params.w;
    in_channels = p.Ops4.Conv_params.in_channels;
    groups = Op_config.Pos.of_int groups;
  }

(* Unlike [conv2d_params] above, [groups] here is [Grouped_conv2d]'s own field
   (Ops4.ml), not a value this module supplies per constructor. *)
let grouped_conv2d_params (p : Ops4.Grouped_conv_params.t) : Conv.Conv2d.params
    =
  {
    h = p.Ops4.Grouped_conv_params.h;
    w = p.Ops4.Grouped_conv_params.w;
    in_channels = p.Ops4.Grouped_conv_params.in_channels;
    groups = p.Ops4.Grouped_conv_params.groups;
  }

let transposed_params (p : Ops4.Transposed_conv2d.params) :
    Conv.Convolution.params =
  {
    stride = p.Ops4.Transposed_conv2d.stride;
    padding = p.Ops4.Transposed_conv2d.padding;
    dilation = p.Ops4.Transposed_conv2d.dilation;
    transposed = true;
    output_padding = p.Ops4.Transposed_conv2d.output_padding;
    groups = Op_config.Pos.of_int 1;
  }

(* A four-axis permutation completed to the six-axis bijection Native's rule
   wants: T and D map to themselves, which is exactly what "does not use T or D
   as semantic axes" means once [Axis4.t] has made it unsayable otherwise. *)
let perm6 (perm : Ops4.Permute4.perm) : Permute.Permute.perm =
  Permute.Permute.of_fn (fun axis ->
      match Axis4.of_axis axis with
      | None -> axis (* T and D are fixed *)
      | Some a4 -> Axis4.to_axis (Ops4.Permute4.lookup perm a4))

let mean_params (p : Ops4.Mean_keepdims.params) : Reduce.Mean.params =
  {
    dims = List.map Axis4.to_axis p.Ops4.Mean_keepdims.dims;
    keepdim = true (* the dialect has no other form *);
  }

let max_params (p : Ops4.Max_keepdims.params) : Reduce.Amax.params =
  {
    dims = List.map Axis4.to_axis p.Ops4.Max_keepdims.dims;
    keepdim = true (* the dialect has no other form *);
  }

let sum_params (p : Ops4.Sum_keepdims.params) : Reduce.Sum.params =
  {
    dims = List.map Axis4.to_axis p.Ops4.Sum_keepdims.dims;
    keepdim = true (* the dialect has no other form *);
  }

let vector_norm_params (p : Ops4.Vector_norm_keepdims.params) :
    Reduce.Vector_norm.params =
  {
    dims = List.map Axis4.to_axis p.Ops4.Vector_norm_keepdims.dims;
    keepdim = true (* the dialect has no other form *);
  }

(* Shared with [Eval_op4], which needs the same translation for the same op:
   one adapter, so the shape rule and the compute cannot disagree about which
   axis is being joined along. *)
let concat_params (p : Ops4.Concat4.params) : Concat.Concat.params =
  { axis = Axis4.to_axis p.Ops4.Concat4.axis }

(* Shared with [Eval_op4], which needs the same translation for the same op:
   one adapter, so the shape rule and the compute cannot disagree about which
   axis is being inserted. *)
let stack_params (p : Ops4.Stack4.params) : Concat.Stack.params =
  { axis = Axis4.to_axis p.Ops4.Stack4.axis }

(* Shared with [Eval_op4] for the same reason [unbind_params] is: one
   translation, so the shape rule and the compute cannot disagree about which
   axes carry which amounts. Only the axis type narrows -- the entries and the
   mode are Native's own values, unchanged. *)
let pad_params (p : Ops4.Pad4.params) : Pad.Pad.params =
  {
    pads = List.map (fun (a, e) -> (Axis4.to_axis a, e)) p.Ops4.Pad4.pads;
    mode = p.Ops4.Pad4.mode;
  }

(* Shared with [Eval_op4] for [pad_params]' reason. Only the axis type narrows;
   the bounds are already canonical and mean the same thing in either dialect. *)
let slice_params (p : Ops4.Slice4.params) : Split.Slice.params =
  {
    axis = Axis4.to_axis p.Ops4.Slice4.axis;
    start = p.Ops4.Slice4.start;
    stop = p.Ops4.Slice4.stop;
    step = p.Ops4.Slice4.step;
  }

(* Shared with [Eval_op4], which needs the same translation for the same op:
   one adapter, so the shape rule and the compute cannot disagree about which
   axis is being split. *)
let unbind_params (p : Ops4.Unbind.params) : Split.Unbind.params =
  { axis = Axis4.to_axis p.Ops4.Unbind.axis }

(* Shared with [Eval_op4], which needs the same translation for the same op:
   one adapter, so the shape rule and the compute cannot disagree about which
   axis is being dropped. [index] crosses unchanged. *)
let select_params (p : Ops4.Select4.params) : Split.Select.params =
  { axis = Axis4.to_axis p.Ops4.Select4.axis; index = p.Ops4.Select4.index }

(* Shared with [Eval_op4], which needs the same translation for the same op:
   one adapter, so the shape rule and the compute cannot disagree about which
   axis is multiplied. [repeats] crosses unchanged. *)
let repeat_interleave_params (p : Ops4.RepeatInterleave4.params) :
    Repeat.RepeatInterleave.params =
  {
    axis = Axis4.to_axis p.Ops4.RepeatInterleave4.axis;
    repeats = p.Ops4.RepeatInterleave4.repeats;
  }

(* Shared with [Eval_op4], which needs the same translation for the same op:
   one adapter, so the shape rule and the compute cannot disagree about which
   axis is gathered. *)
let index_tensor_params (p : Ops4.IndexTensor4.params) :
    Index_tensor.Index_tensor.params =
  { axis = Axis4.to_axis p.Ops4.IndexTensor4.axis }

(* Shared with [Eval_op4], which needs the same translation for the same op:
   one adapter, so the shape rule and the compute cannot disagree about which
   axis is written. [index] crosses unchanged. *)
let select_scatter_params (p : Ops4.Select_scatter4.params) :
    Split.Select_scatter.params =
  {
    axis = Axis4.to_axis p.Ops4.Select_scatter4.axis;
    index = p.Ops4.Select_scatter4.index;
  }

(* Shared with [Eval_op4], which needs the same translation for the same op:
   one adapter, so the shape rule and the compute cannot disagree about which
   axis softmax reduces over. *)
let softmax_params (p : Ops4.Softmax4.params) : Reduce.Softmax.params =
  { axis = Axis4.to_axis p.Ops4.Softmax4.axis }

(* Shared with [Eval_op4], which needs the same translation for the same op:
   one adapter, so the shape rule and the compute cannot disagree about which
   axis cumsum walks. *)
let cumsum_params (p : Ops4_cumsum.Cumsum4.params) : Reduce.Cumsum.params =
  { axis = Axis4.to_axis p.Ops4_cumsum.Cumsum4.axis }

let split_with_sizes_params (p : Ops4.Split_with_sizes4.params) :
    Split.Split_with_sizes.params =
  {
    axis = Axis4.to_axis p.Ops4.Split_with_sizes4.axis;
    sizes = p.Ops4.Split_with_sizes4.sizes;
  }

let layer_norm_params (p : Ops4.Layer_norm.params) : Norm.LayerNorm.params =
  {
    dims = List.map Axis4.to_axis p.Ops4.Layer_norm.dims;
    eps = p.Ops4.Layer_norm.eps;
  }

let rms_params (p : Ops4.Rms_norm.params) : Norm.RmsNorm.params =
  {
    dims = List.map Axis4.to_axis p.Ops4.Rms_norm.dims;
    eps = p.Ops4.Rms_norm.eps;
  }

let batch_norm_no_stats_params (p : Ops4.Batch_norm_no_stats.params) :
    Norm.BatchNormNoStats.params =
  { channel = Axis4.to_axis p.Ops4.Batch_norm_no_stats.channel; eps = p.eps }

let group_norm_params (p : Ops4.Group_norm4.params) : Norm.GroupNorm.params =
  {
    channel = Axis4.to_axis p.Ops4.Group_norm4.channel;
    groups = p.Ops4.Group_norm4.groups;
    eps = p.Ops4.Group_norm4.eps;
  }

(* Exhaustive with no default arm, as [Graph_shape] is. *)
let output_shape (op : Op.t)
    ~(sig_of : Tensor_ref.t -> (Tensor_sig.t, error) Err.t) :
    (Shape4.t list, error) Err.t =
  let open Err.Syntax in
  let shape r = sig_of r >>| fun sg -> sg.Tensor_sig.shape in
  let one r = r >>| fun s -> [ s ] in
  match op with
  | Add { Pointwise.Bin.a; b } ->
      let* a_shape = shape a in
      let* b_shape = shape b in
      one (four (Pointwise.Add.output_shape a_shape b_shape))
  | Add_scalar { Pointwise.Scalar_bin.x; _ } ->
      let* x_shape = shape x in
      one (four (Pointwise.Add_scalar.output_shape x_shape))
  | Adaptive_avg_pool2d { Pool.AdaptiveAvgPool2d.params; x } ->
      let* x_shape = shape x in
      one (four (Pool.AdaptiveAvgPool2d.output_shape ~x_shape params))
  | Adaptive_max_pool2d { Pool.AdaptiveMaxPool2d.params; x } ->
      let* x_shape = shape x in
      one (four (Pool.AdaptiveMaxPool2d.output_shape ~x_shape params))
  | Avg_pool2d { Pool.AvgPool2d.params; x } ->
      let* x_shape = shape x in
      one (four (Pool.AvgPool2d.output_shape ~x_shape params))
  | Batch_norm_no_stats { Ops4.Batch_norm_no_stats.params; x; weight; bias } ->
      let* x_shape = shape x in
      let p = batch_norm_no_stats_params params in
      let* () =
        Err.List.iter
          (fun r ->
            let* actual = shape r in
            widen (Norm.BatchNormNoStats.check_affine ~x_shape p ~actual))
          (Option.to_list weight @ Option.to_list bias)
      in
      let* shapes = widen (Norm.BatchNormNoStats.output_shapes ~x_shape p) in
      Err.List.map (fun shape -> widen (Shape4.of_vec6 shape)) shapes
  | Batched_matmul { Matmul.Batched_matmul.input; mat2 } ->
      let* input_shape = shape input in
      let* mat2_shape = shape mat2 in
      one (four (Matmul.Batched_matmul.output_shape ~input_shape ~mat2_shape))
  | Clamp { Pointwise.Clamp.params; x } ->
      let* x_shape = shape x in
      one (four (Pointwise.Clamp.output_shape params x_shape))
  (* Variadic OPERANDS, not variadic outputs -- unlike [Unbind]'s [four_all],
     this stays a single-shape arm, one [xs_shapes] gathered over every
     operand rather than one [x_shape]. *)
  | Concat4 { Ops4.Concat4.params; xs } ->
      let* xs_shapes = Err.List.map shape xs in
      one (four (Concat.Concat.output_shape ~xs_shapes (concat_params params)))
  | Conv2d { Ops4.Conv_payload.params; x; weight; _ } ->
      let* x_shape = shape x in
      let* weight_shape = shape weight in
      one
        (four
           (Conv.Conv2d.output_shape ~x_shape ~weight_shape
              (conv2d_params params ~groups:1)))
  | Depthwise_conv2d { Ops4.Conv_payload.params; x; weight; _ } ->
      let* x_shape = shape x in
      let* weight_shape = shape weight in
      (* One group per input channel: the grouping the constructor names. *)
      let groups = Dim.to_int params.Ops4.Conv_params.in_channels in
      one
        (four
           (Conv.Conv2d.output_shape ~x_shape ~weight_shape
              (conv2d_params params ~groups)))
  (* Cumsum rescales nothing and drops no axis, exactly [Softmax4]'s reason:
     [Reduce.Cumsum.output_shape] is the identity on [x_shape] -- delegated
     rather than restated. *)
  | Cumsum4 { Ops4_cumsum.Cumsum4.params; x } ->
      let* x_shape = shape x in
      one (four (Reduce.Cumsum.output_shape ~x_shape (cumsum_params params)))
  | Div { Pointwise.Bin.a; b } ->
      let* a_shape = shape a in
      let* b_shape = shape b in
      one (four (Pointwise.Div.output_shape a_shape b_shape))
  | Div_scalar { Pointwise.Scalar_bin.x; _ } ->
      let* x_shape = shape x in
      one (four (Pointwise.Div_scalar.output_shape x_shape))
  (* The target is already a [Shape4.t] -- an expansion cannot leave the
     dialect, the same reason [Reshape4]'s target is typed rather than
     validated for axes. Delegates to Native's own rule like every other arm,
     so a broadcast this dialect cannot represent is caught by the [four]
     wrap rather than restated here. *)
  | Expand4 { Ops4.Expand4.params; x } ->
      let* x_shape = shape x in
      one
        (four
           (Pointwise.Expand.output_shape ~x_shape
              {
                Pointwise.Expand.size = Shape4.to_vec6 params.Ops4.Expand4.size;
              }))
  | Gelu { Pointwise.Gelu.x; _ } ->
      let* x_shape = shape x in
      one (four (Pointwise.Gelu.output_shape x_shape))
  (* Same [check_affine]-then-[output_shape] shape as [Layer_norm] above, one
     [Norm.GroupNorm] function apiece so the two dialects cannot disagree
     about which extents an operand carries. *)
  | Group_norm4 { Ops4.Group_norm4.params; x; weight; bias } ->
      let* x_shape = shape x in
      let opt_shape = function
        | None -> Err.return None
        | Some r ->
            let+ s = shape r in
            Some s
      in
      let* weight = opt_shape weight in
      let* bias = opt_shape bias in
      let p = group_norm_params params in
      let* () =
        widen
          (Norm.GroupNorm.check_affine ~x_shape
             ~channel:p.Norm.GroupNorm.channel ~weight ~bias)
      in
      one (four (Norm.GroupNorm.output_shape ~x_shape p))
  | Grouped_conv2d { Ops4.Grouped_conv_payload.params; x; weight; _ } ->
      let* x_shape = shape x in
      let* weight_shape = shape weight in
      one
        (four
           (Conv.Conv2d.output_shape ~x_shape ~weight_shape
              (grouped_conv2d_params params)))
  | Hardsigmoid { Pointwise.Hardsigmoid.x } ->
      let* x_shape = shape x in
      one (four (Pointwise.Hardsigmoid.output_shape x_shape))
  | Hardswish { Pointwise.Hardswish.x } ->
      let* x_shape = shape x in
      one (four (Pointwise.Hardswish.output_shape x_shape))
  | Hardtanh { Pointwise.Hardtanh.x; _ } ->
      let* x_shape = shape x in
      one (four (Pointwise.Hardtanh.output_shape x_shape))
  (* [self]'s [axis] extent is overwritten by [index]'s own length -- no drop,
     no repack, the same reason [Select_scatter4] needs no post-hoc re-check
     -- delegated to [Index_tensor.Index_tensor.output_shape] rather than
     restated, so this arm and its [Compute] cannot disagree about which axis
     is gathered. *)
  | IndexTensor4 { Ops4.IndexTensor4.params; self; index } ->
      let* self_shape = shape self in
      let* index_shape = shape index in
      one
        (four
           (Index_tensor.Index_tensor.output_shape ~self_shape ~index_shape
              (index_tensor_params params)))
  (* The affine check that Native's own [Graph_shape] runs and this file's
     [Rms_norm] arm does NOT (see below). A JSON-decoded Native4D graph reaches
     this rule and no other, so leaving it out means an operand of the wrong
     extent builds a graph here and raises out of [Tensor.read]'s bounds check
     partway through the result. One shared [check_affine], so the two dialects
     cannot come to disagree about which extents an operand carries. *)
  | Layer_norm { Ops4.Layer_norm.params; x; weight; bias } ->
      let* x_shape = shape x in
      let opt_shape = function
        | None -> Err.return None
        | Some r ->
            let+ s = shape r in
            Some s
      in
      let* weight = opt_shape weight in
      let* bias = opt_shape bias in
      let p = layer_norm_params params in
      let* () =
        widen
          (Norm.LayerNorm.check_affine ~x_shape ~dims:p.Norm.LayerNorm.dims
             ~weight ~bias)
      in
      one (four (Norm.LayerNorm.output_shape ~x_shape p))
  | Leaky_relu { Pointwise.Leaky_relu.x; _ } ->
      let* x_shape = shape x in
      one (four (Pointwise.Leaky_relu.output_shape x_shape))
  | Max_keepdims { Ops4.Max_keepdims.params; x } ->
      let* x_shape = shape x in
      one (four (Reduce.Amax.output_shape ~x_shape (max_params params)))
  | Max_pool2d { Pool.MaxPool2d.params; x } ->
      let* x_shape = shape x in
      one (four (Pool.MaxPool2d.output_shape ~x_shape params))
  | Mean_keepdims { Ops4.Mean_keepdims.params; x } ->
      let* x_shape = shape x in
      one (four (Reduce.Mean.output_shape ~x_shape (mean_params params)))
  | Mul { Pointwise.Bin.a; b } ->
      let* a_shape = shape a in
      let* b_shape = shape b in
      one (four (Pointwise.Mul.output_shape a_shape b_shape))
  | Mul_scalar { Pointwise.Scalar_bin.x; _ } ->
      let* x_shape = shape x in
      one (four (Pointwise.Mul_scalar.output_shape x_shape))
  | Pad4 { Ops4.Pad4.params; x } ->
      let* x_shape = shape x in
      one (four (Pad.Pad.output_shape ~x_shape (pad_params params)))
  | Permute4 { Ops4.Permute4.perm; x } ->
      let* x_shape = shape x in
      one (four (Permute.Permute.output_shape ~x_shape (perm6 perm)))
  | Pow { Pointwise.Scalar_bin.x; _ } ->
      let* x_shape = shape x in
      one (four (Pointwise.Pow.output_shape x_shape))
  | Relu { Pointwise.Relu.x } ->
      let* x_shape = shape x in
      one (four (Pointwise.Relu.output_shape x_shape))
  | Repeat4 { Ops4.Repeat4.params; x } ->
      let* x_shape = shape x in
      one
        (four
           (Repeat.Repeat.output_shape ~x_shape
              {
                Repeat.Repeat.repeats =
                  Shape4.to_vec6 params.Ops4.Repeat4.repeats;
              }))
  | RepeatInterleave4 { Ops4.RepeatInterleave4.params; x } ->
      let* x_shape = shape x in
      one
        (four
           (Repeat.RepeatInterleave.output_shape ~x_shape
              (repeat_interleave_params params)))
  | Reshape4 { Ops4.Reshape4.params; x } ->
      (* The target is already a [Shape4.t]: a reshape cannot leave the
         dialect, which is why the target is typed rather than validated for
         AXES. It says nothing about element count, though, so this arm
         delegates to the Native rule like every other -- restating a shape
         rule here "would be a second definition free to drift" (module
         header above). Without this, [Builder.reshape4] and any
         JSON-decoded Native4D graph inherited op3-impl.md's F1: a target
         whose numel disagrees with the source was accepted silently. *)
      let* x_shape = shape x in
      one
        (four
           (Reshape.Reshape.output_shape ~x_shape
              {
                Reshape.Reshape.shape =
                  Shape4.to_vec6 params.Ops4.Reshape4.shape;
              }))
  (* Asymmetric with the arm above ON PURPOSE, and worth stating rather than
     quietly copying: [Rms_norm] does not re-run [check_weight] here, so a
     JSON-decoded Native4D rms_norm skips the affine validation Native's
     [Graph_shape] performs. Noted in .ai/native4d_design.md; not fixed in the
     same change that adds a different op. *)
  | Rms_norm { Ops4.Rms_norm.params; x; _ } ->
      let* x_shape = shape x in
      one (four (Norm.RmsNorm.output_shape ~x_shape (rms_params params)))
  | Rsub_scalar { Pointwise.Rsub_scalar.x; _ } ->
      let* x_shape = shape x in
      one (four (Pointwise.Rsub_scalar.output_shape x_shape))
  | Sdpa { Attention.Sdpa.params = _; query; key; value; mask } ->
      let* query_shape = shape query in
      let* key_shape = shape key in
      let* value_shape = shape value in
      let* mask_shape =
        match mask with
        | None -> Err.return None
        | Some m ->
            let+ s = shape m in
            Some s
      in
      one
        (four
           (Attention.Sdpa.output_shape ~query_shape ~key_shape ~value_shape
              ~mask_shape))
  (* [Select] drops its axis, unlike [Slice4] above: the shape rule is
     [Split.Select.output_shape], which repacks every surviving axis
     right-aligned -- delegated rather than restated, so this arm and
     [Split.Select.Compute] cannot disagree about which axis vacates. *)
  | Select4 { Ops4.Select4.params; x } ->
      let* x_shape = shape x in
      one (four (Split.Select.output_shape ~x_shape (select_params params)))
  (* Unlike [Select4], the output IS [self]'s shape -- [axis]/[index] name a
     WRITE position, not a shape transform. [src] is checked against exactly
     the shape [Split.Select.output_shape] would give at this axis/index,
     delegated rather than restated, so this arm and
     [Split.Select_scatter.Compute] cannot disagree about which position is
     written. *)
  | Select_scatter4 { Ops4.Select_scatter4.params; self; src } ->
      let* self_shape = shape self in
      let* src_shape = shape src in
      one
        (four
           (Split.Select_scatter.output_shape ~self_shape ~src_shape
              (select_scatter_params params)))
  | Sigmoid { Pointwise.Sigmoid.x } ->
      let* x_shape = shape x in
      one (four (Pointwise.Sigmoid.output_shape x_shape))
  | Silu { Pointwise.Silu.x } ->
      let* x_shape = shape x in
      one (four (Pointwise.Silu.output_shape x_shape))
  | Slice4 { Ops4.Slice4.params; x } ->
      let* x_shape = shape x in
      one (four (Split.Slice.output_shape ~x_shape (slice_params params)))
  (* Softmax rescales, it does not reduce: unlike every keep-dimensions
     arm above, [Reduce.Softmax.output_shape] is the identity on [x_shape] --
     delegated rather than restated, the same discipline every other arm
     here follows. *)
  | Softmax4 { Ops4.Softmax4.params; x } ->
      let* x_shape = shape x in
      one (four (Reduce.Softmax.output_shape ~x_shape (softmax_params params)))
  (* [Unbind]'s "carries the row through" arm, mirrored: Native has already
     bounded the count against [Kernel.Limits.Hard.outputs] and proved the
     sizes sum to the axis extent. *)
  | Split_with_sizes4 { Ops4.Split_with_sizes4.params; x } ->
      let* x_shape = shape x in
      four_all
        (Split.Split_with_sizes.output_shapes ~x_shape
           (split_with_sizes_params params))
  | Sqrt { Pointwise.Sqrt.x } ->
      let* x_shape = shape x in
      one (four (Pointwise.Sqrt.output_shape x_shape))
  (* Variadic OPERANDS, the same shape [Concat4] above has -- one [xs_shapes]
     gathered over every operand, delegated whole to [Concat.Stack], which
     folds each operand's unsqueezed shape through [Concat.Concat] itself. *)
  | Stack4 { Ops4.Stack4.params; xs } ->
      let* xs_shapes = Err.List.map shape xs in
      one (four (Concat.Stack.output_shape ~xs_shapes (stack_params params)))
  | Sub { Pointwise.Bin.a; b } ->
      let* a_shape = shape a in
      let* b_shape = shape b in
      one (four (Pointwise.Sub.output_shape a_shape b_shape))
  | Sum_keepdims { Ops4.Sum_keepdims.params; x } ->
      let* x_shape = shape x in
      one (four (Reduce.Sum.output_shape ~x_shape (sum_params params)))
  | To_copy { Pointwise.To_copy.x; _ } ->
      let* x_shape = shape x in
      one (four (Pointwise.To_copy.output_shape x_shape))
  | Transposed_conv2d { Ops4.Transposed_conv2d.params; x; weight; _ } ->
      let* x_shape = shape x in
      let* weight_shape = shape weight in
      one
        (four
           (Conv.Convolution.output_shape ~x_shape ~weight_shape
              (transposed_params params)))
  (* The one arm returning a list whose length is not the op's. Native has
     already bounded the count against [Kernel.Limits.Hard.outputs], so this
     path only carries the row through. *)
  | Unbind { Ops4.Unbind.params; x } ->
      let* x_shape = shape x in
      four_all (Split.Unbind.output_shapes ~x_shape (unbind_params params))
  | Upsample_bilinear2d { Resize.Bilinear2d.params; x } ->
      let* x_shape = shape x in
      one (four (Resize.Bilinear2d.output_shape ~x_shape params))
  | Upsample_nearest2d { Resize.Nearest2d.params; x } ->
      let* x_shape = shape x in
      one (four (Resize.Nearest2d.output_shape ~x_shape params))
  | Vector_norm_keepdims { Ops4.Vector_norm_keepdims.params; x } ->
      let* x_shape = shape x in
      one
        (four
           (Reduce.Vector_norm.output_shape ~x_shape
              (vector_norm_params params)))
  | Arange4 { Ops4.Arange4.params } ->
      let native =
        Factory.Arange.
          {
            start = params.start;
            stop = params.stop;
            step = params.step;
            fmt = params.fmt;
          }
      in
      one (four (Factory.Arange.output_shape native))
  | Zeros4 { Ops4.Zeros4.params } -> Err.return [ params.shape ]
  | Eye4 { Ops4.Eye4.params } -> Err.return [ params.shape ]
