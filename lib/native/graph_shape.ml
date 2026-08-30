(* Shape dispatch: the S-independent twin of [Eval_op]. Given an op and a way to
   look up each operand edge's signature, it returns the op's output shape(s) by
   calling each native op's own [output_shape] on the relevant input shapes. The
   builder uses it to stamp output edges; the evaluators use it to size each
   result. One of the per-op match functions touched when an op is added
   ([Eval_op], [Graph_shape], [Graph_ir.operands]/[map_operands]). *)

open Graph_ir

type error = [ `Missing_tensor_sig of Tensor_id.t | Shape_error.t ]

let pp_error ppf : [< error ] -> unit = function
  | `Missing_tensor_sig id ->
      Format.fprintf ppf "missing tensor sig t%d" (Tensor_id.to_int id)
  | #Shape_error.t as e -> Shape_error.pp ppf e

let widen (r : ('a, [< error ]) Err.t) : ('a, error) Err.t =
  (r :> ('a, error) Err.t)

(* The optional affine bias, checked HERE and not inside each op's
   [output_shape]: those take the required operands only, by design -- an output
   shape is not a function of the bias. So the check has no other home, and
   without it a short bias built a graph and raised from [Tensor.read] partway
   through evaluation. Both importers inherit the rejection. *)
let output_shape (op : op) ~(sig_of : tensor_ref -> (Tensor_sig.t, error) Err.t)
    : (Vec6.shape list, error) Err.t =
  let open Err.Syntax in
  let shape r = sig_of r >>| fun sg -> sg.Tensor_sig.shape in
  let check_bias ~shape ~expected = function
    | None -> Err.return ()
    | Some b ->
        let* actual = shape b in
        widen (Affine_bias.check ~expected ~actual)
  in
  match op with
  | Add { Pointwise.Bin.a; b } ->
      let* a_shape = shape a in
      let* b_shape = shape b in
      let+ out = widen (Pointwise.Add.output_shape a_shape b_shape) in
      [ out ]
  | Add_scalar { Pointwise.Scalar_bin.x; _ } ->
      let* x_shape = shape x in
      let+ out = widen (Pointwise.Add_scalar.output_shape x_shape) in
      [ out ]
  | Adaptive_avg_pool2d { Pool.AdaptiveAvgPool2d.params; x } ->
      let* x_shape = shape x in
      let+ out = widen (Pool.AdaptiveAvgPool2d.output_shape ~x_shape params) in
      [ out ]
  | Amax { Reduce.Amax.params; x } ->
      let* x_shape = shape x in
      let+ out = widen (Reduce.Amax.output_shape ~x_shape params) in
      [ out ]
  | Avg_pool2d { Pool.AvgPool2d.params; x } ->
      let* x_shape = shape x in
      let+ out = widen (Pool.AvgPool2d.output_shape ~x_shape params) in
      [ out ]
  | Batch_norm { Norm.BatchNorm.x; _ } ->
      let* x_shape = shape x in
      let+ out = widen (Norm.BatchNorm.output_shape ~x_shape) in
      [ out ]
  | Bmm { Matmul.Bmm.input; mat2 } ->
      let* input_shape = shape input in
      let* mat2_shape = shape mat2 in
      let+ out = widen (Matmul.Bmm.output_shape ~input_shape ~mat2_shape) in
      [ out ]
  | Clamp { Pointwise.Clamp.params; x } ->
      let* x_shape = shape x in
      let+ out = widen (Pointwise.Clamp.output_shape params x_shape) in
      [ out ]
  | Clone { Pointwise.Clone.x } ->
      let* x_shape = shape x in
      let+ out = widen (Pointwise.Clone.output_shape x_shape) in
      [ out ]
  | Concat { Concat.Concat.params; xs } ->
      let* xs_shapes = Err.List.map shape xs in
      let+ out = widen (Concat.Concat.output_shape ~xs_shapes params) in
      [ out ]
  | Conv2d { Conv.Conv2d.params; x; weight; bias } ->
      let* x_shape = shape x in
      let* weight_shape = shape weight in
      let* () =
        check_bias ~shape ~expected:(Affine_bias.shape ~weight_shape) bias
      in
      let+ out =
        widen (Conv.Conv2d.output_shape ~x_shape ~weight_shape params)
      in
      [ out ]
  | Conv2d_padding { Conv.Conv2d_padding.params; x; weight; bias } ->
      let* x_shape = shape x in
      let* weight_shape = shape weight in
      let* () =
        check_bias ~shape ~expected:(Affine_bias.shape ~weight_shape) bias
      in
      let+ out =
        widen (Conv.Conv2d_padding.output_shape ~x_shape ~weight_shape params)
      in
      [ out ]
  | Convolution { Conv.Convolution.params; x; weight; bias } ->
      let* x_shape = shape x in
      let* weight_shape = shape weight in
      (* NOT [Affine_bias.shape]: a transposed convolution's weight is
         [Cin, Cout/groups, kH, kW], so its output channel count comes from
         [weight.C * groups] and not from [weight.N]. *)
      let* () =
        check_bias ~shape
          ~expected:(Conv.Convolution.bias_shape ~weight_shape params)
          bias
      in
      let+ out =
        widen (Conv.Convolution.output_shape ~x_shape ~weight_shape params)
      in
      [ out ]
  | Discard _ -> Err.return []
  | Gelu { Pointwise.Gelu.x; _ } ->
      let* x_shape = shape x in
      let+ out = widen (Pointwise.Gelu.output_shape x_shape) in
      [ out ]
  (* Both affine operands are optional and share ONE expected layout (the
     per-channel vector [check_affine] describes), the same division of labor
     [Layer_norm]'s arm above uses. *)
  | Group_norm { Norm.GroupNorm.params; x; weight; bias } ->
      let* x_shape = shape x in
      let opt_shape = function
        | None -> Err.return None
        | Some r ->
            let+ s = shape r in
            Some s
      in
      let* weight = opt_shape weight in
      let* bias = opt_shape bias in
      let* () =
        widen
          (Norm.GroupNorm.check_affine ~x_shape
             ~channel:params.Norm.GroupNorm.channel ~weight ~bias)
      in
      let+ out = widen (Norm.GroupNorm.output_shape ~x_shape params) in
      [ out ]
  | Hardsigmoid { Pointwise.Hardsigmoid.x } ->
      let* x_shape = shape x in
      let+ out = widen (Pointwise.Hardsigmoid.output_shape x_shape) in
      [ out ]
  | Hardswish { Pointwise.Hardswish.x } ->
      let* x_shape = shape x in
      let+ out = widen (Pointwise.Hardswish.output_shape x_shape) in
      [ out ]
  | Hardtanh { Pointwise.Hardtanh.x; _ } ->
      let* x_shape = shape x in
      let+ out = widen (Pointwise.Hardtanh.output_shape x_shape) in
      [ out ]
  (* Both affine operands are optional, so both go through the op's own
     [check_affine] rather than [check_bias]: they share ONE expected layout
     (the normalized_shape), which is not the per-channel vector
     [Affine_bias.check] knows about. *)
  | Layer_norm { Norm.LayerNorm.params; x; weight; bias } ->
      let* x_shape = shape x in
      let opt_shape = function
        | None -> Err.return None
        | Some r ->
            let+ s = shape r in
            Some s
      in
      let* weight = opt_shape weight in
      let* bias = opt_shape bias in
      let* () =
        widen
          (Norm.LayerNorm.check_affine ~x_shape ~dims:params.Norm.LayerNorm.dims
             ~weight ~bias)
      in
      let+ out = widen (Norm.LayerNorm.output_shape ~x_shape params) in
      [ out ]
  | Linear { Linear.Linear.params; x; weight; bias } ->
      let* x_shape = shape x in
      let* weight_shape = shape weight in
      let* () =
        check_bias ~shape ~expected:(Affine_bias.shape ~weight_shape) bias
      in
      let+ out =
        widen (Linear.Linear.output_shape params ~x_shape ~weight_shape)
      in
      [ out ]
  | Max_pool2d { Pool.MaxPool2d.params; x } ->
      let* x_shape = shape x in
      let+ out = widen (Pool.MaxPool2d.output_shape ~x_shape params) in
      [ out ]
  | Max_pool2d_with_indices { Pool.MaxPool2dWithIndices.params; x } ->
      let* x_shape = shape x in
      let+ out =
        widen (Pool.MaxPool2dWithIndices.output_shape ~x_shape params)
      in
      (* out0 = values, out1 = indices; both share the pooled window shape. *)
      [ out; out ]
  | Mean { Reduce.Mean.params; x } ->
      let* x_shape = shape x in
      let+ out = widen (Reduce.Mean.output_shape ~x_shape params) in
      [ out ]
  | Div { Pointwise.Bin.a; b } ->
      let* a_shape = shape a in
      let* b_shape = shape b in
      let+ out = widen (Pointwise.Div.output_shape a_shape b_shape) in
      [ out ]
  | Div_scalar { Pointwise.Scalar_bin.x; _ } ->
      let* x_shape = shape x in
      let+ out = widen (Pointwise.Div_scalar.output_shape x_shape) in
      [ out ]
  | Mul { Pointwise.Bin.a; b } ->
      let* a_shape = shape a in
      let* b_shape = shape b in
      let+ out = widen (Pointwise.Mul.output_shape a_shape b_shape) in
      [ out ]
  | Mul_scalar { Pointwise.Scalar_bin.x; _ } ->
      let* x_shape = shape x in
      let+ out = widen (Pointwise.Mul_scalar.output_shape x_shape) in
      [ out ]
  | Pad { Pad.Pad.params; x } ->
      let* x_shape = shape x in
      let+ out = widen (Pad.Pad.output_shape ~x_shape params) in
      [ out ]
  | Permute { Permute.Permute.perm; x } ->
      let* x_shape = shape x in
      let+ out = widen (Permute.Permute.output_shape ~x_shape perm) in
      [ out ]
  | Pow { Pointwise.Scalar_bin.x; _ } ->
      let* x_shape = shape x in
      let+ out = widen (Pointwise.Pow.output_shape x_shape) in
      [ out ]
  | Relu { Pointwise.Relu.x } ->
      let* x_shape = shape x in
      let+ out = widen (Pointwise.Relu.output_shape x_shape) in
      [ out ]
  | Reshape { Reshape.Reshape.params; x } ->
      let* x_shape = shape x in
      let+ out = widen (Reshape.Reshape.output_shape ~x_shape params) in
      [ out ]
  | Rms_norm { Norm.RmsNorm.params; x; weight } ->
      let* x_shape = shape x in
      let* () =
        match weight with
        | None -> Err.return ()
        | Some w ->
            let* actual = shape w in
            widen
              (Norm.RmsNorm.check_weight ~x_shape ~dims:params.Norm.RmsNorm.dims
                 ~actual)
      in
      let+ out = widen (Norm.RmsNorm.output_shape ~x_shape params) in
      [ out ]
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
      let+ out =
        widen
          (Attention.Sdpa.output_shape ~query_shape ~key_shape ~value_shape
             ~mask_shape)
      in
      [ out ]
  | Select { Split.Select.params; x } ->
      let* x_shape = shape x in
      let+ out = widen (Split.Select.output_shape ~x_shape params) in
      [ out ]
  | Sigmoid { Pointwise.Sigmoid.x } ->
      let* x_shape = shape x in
      let+ out = widen (Pointwise.Sigmoid.output_shape x_shape) in
      [ out ]
  | Silu { Pointwise.Silu.x } ->
      let* x_shape = shape x in
      let+ out = widen (Pointwise.Silu.output_shape x_shape) in
      [ out ]
  | Softmax { Reduce.Softmax.params; x } ->
      let* x_shape = shape x in
      let+ out = widen (Reduce.Softmax.output_shape ~x_shape params) in
      [ out ]
  | Sqrt { Pointwise.Sqrt.x } ->
      let* x_shape = shape x in
      let+ out = widen (Pointwise.Sqrt.output_shape x_shape) in
      [ out ]
  | Stack { Concat.Stack.params; xs } ->
      let* xs_shapes = Err.List.map shape xs in
      let+ out = widen (Concat.Stack.output_shape ~xs_shapes params) in
      [ out ]
  | Sub { Pointwise.Bin.a; b } ->
      let* a_shape = shape a in
      let* b_shape = shape b in
      let+ out = widen (Pointwise.Sub.output_shape a_shape b_shape) in
      [ out ]
  | Slice { Split.Slice.params; x } ->
      let* x_shape = shape x in
      let+ out = widen (Split.Slice.output_shape ~x_shape params) in
      [ out ]
  (* The one arm whose LENGTH is not a constant of the op: [output_shapes]
     returns one shape per coordinate of the selected axis, and bounds that
     count itself before building the list. *)
  | Unbind { Split.Unbind.params; x } ->
      let* x_shape = shape x in
      widen (Split.Unbind.output_shapes ~x_shape params)
  (* Same shape as [Unbind]'s arm: the LENGTH comes from [params.sizes], not
     from the op, and [output_shapes] bounds it before the list exists. *)
  | Split_with_sizes { Split.Split_with_sizes.params; x } ->
      let* x_shape = shape x in
      widen (Split.Split_with_sizes.output_shapes ~x_shape params)
  | Upsample_bilinear2d { Resize.Bilinear2d.params; x } ->
      let* x_shape = shape x in
      let+ out = widen (Resize.Bilinear2d.output_shape ~x_shape params) in
      [ out ]
  | Vector_norm { Reduce.Vector_norm.params; x } ->
      let* x_shape = shape x in
      let+ out = widen (Reduce.Vector_norm.output_shape ~x_shape params) in
      [ out ]
