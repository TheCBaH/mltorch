(* Shape dispatch: the S-independent twin of [Eval_op]. Given an op and a way to
   look up each operand edge's signature, it returns the op's output shape(s) by
   calling each native op's own [output_shape] on the relevant input shapes. The
   builder uses it to stamp output edges; the evaluators use it to size each
   result. One of the per-op match functions touched when an op is added
   ([Eval_op], [Graph_shape], [Graph_ir.operands]/[map_operands]). *)

open Graph_ir

type error = [ Shape_error.t | `Missing_tensor_sig of Tensor_id.t ]

let pp_error ppf : [< error ] -> unit = function
  | #Shape_error.t as e -> Shape_error.pp ppf e
  | `Missing_tensor_sig id ->
      Format.fprintf ppf "missing tensor sig t%d" (Tensor_id.to_int id)

let widen (r : ('a, [< error ]) Core.result) : ('a, error) Core.result =
  (r :> ('a, error) Core.result)

let output_shape (op : op)
    ~(sig_of : tensor_ref -> (Tensor_sig.t, error) Core.result) :
    (Vec6.shape list, error) Core.result =
  let open Core.Syntax in
  let shape r = sig_of r >>| fun sg -> sg.Tensor_sig.shape in
  match op with
  | Add { Pointwise.Bin.a; b } ->
      let* a_shape = shape a in
      let* b_shape = shape b in
      let+ out = widen (Pointwise.Add.output_shape a_shape b_shape) in
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
  | Conv2d { Conv.Conv2d.params; x; weight; _ } ->
      let* x_shape = shape x in
      let* weight_shape = shape weight in
      let+ out =
        widen (Conv.Conv2d.output_shape ~x_shape ~weight_shape params)
      in
      [ out ]
  | Conv2d_padding { Conv.Conv2d_padding.params; x; weight; _ } ->
      let* x_shape = shape x in
      let* weight_shape = shape weight in
      let+ out =
        widen (Conv.Conv2d_padding.output_shape ~x_shape ~weight_shape params)
      in
      [ out ]
  | Convolution { Conv.Convolution.params; x; weight; _ } ->
      let* x_shape = shape x in
      let* weight_shape = shape weight in
      let+ out =
        widen (Conv.Convolution.output_shape ~x_shape ~weight_shape params)
      in
      [ out ]
  | Discard _ -> Core.return []
  | Linear { Linear.Linear.params; x; weight; _ } ->
      let* x_shape = shape x in
      let* weight_shape = shape weight in
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
  | Mul { Pointwise.Bin.a; b } ->
      let* a_shape = shape a in
      let* b_shape = shape b in
      let+ out = widen (Pointwise.Mul.output_shape a_shape b_shape) in
      [ out ]
  | Permute { Permute.Permute.perm; x } ->
      let* x_shape = shape x in
      let+ out = widen (Permute.Permute.output_shape ~x_shape perm) in
      [ out ]
  | Relu { Pointwise.Relu.x } ->
      let* x_shape = shape x in
      let+ out = widen (Pointwise.Relu.output_shape x_shape) in
      [ out ]
  | Reshape { Reshape.Reshape.params; _ } ->
      let+ out = widen (Reshape.Reshape.output_shape params) in
      [ out ]
  | Rms_norm { Norm.RmsNorm.x; _ } ->
      let* x_shape = shape x in
      let+ out = widen (Norm.RmsNorm.output_shape ~x_shape) in
      [ out ]
