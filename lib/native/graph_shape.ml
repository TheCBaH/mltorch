(* Shape dispatch: the S-independent twin of [Eval_op]. Given an op and a way to
   look up each operand edge's signature, it returns the op's output shape(s) by
   calling each native op's own [output_shape] on the relevant input shapes. The
   builder uses it to stamp output edges; the evaluators use it to size each
   result. One of the per-op match functions touched when an op is added
   ([Eval_op], [Graph_shape], [Graph_ir.operands]/[map_operands]). *)

open Graph_ir

let output_shape (op : op) ~(sig_of : tensor_ref -> Tensor_sig.t) :
    Vec6.shape list =
  let shape r = (sig_of r).Tensor_sig.shape in
  match op with
  | Add { a; b } -> [ Pointwise.Add.output_shape (shape a) (shape b) ]
  | Avg_pool2d { params; x } ->
      [ Pool.AvgPool2d.output_shape ~x_shape:(shape x) params ]
  | Bmm { input; mat2 } ->
      [
        Matmul.Bmm.output_shape ~input_shape:(shape input)
          ~mat2_shape:(shape mat2);
      ]
  | Conv2d { params; x; weight; _ } ->
      [
        Conv.Conv2d.output_shape ~x_shape:(shape x) ~weight_shape:(shape weight)
          params;
      ]
  | Linear { x; weight; _ } ->
      [
        Linear.Linear.output_shape ~x_shape:(shape x)
          ~weight_shape:(shape weight);
      ]
  | Max_pool2d { params; x } ->
      [ Pool.MaxPool2d.output_shape ~x_shape:(shape x) params ]
  | Mean { params; x } -> [ Reduce.Mean.output_shape ~x_shape:(shape x) params ]
  | Mul { a; b } -> [ Pointwise.Mul.output_shape (shape a) (shape b) ]
  | Permute { perm; x } ->
      [ Permute.Permute.output_shape ~x_shape:(shape x) perm ]
  | Relu { x } -> [ Pointwise.Relu.output_shape (shape x) ]
  | Rms_norm { x; _ } -> [ Norm.RmsNorm.output_shape ~x_shape:(shape x) ]
  | Subgraph { graph; _ } ->
      (* A subgraph's outputs are exactly its embedded graph's output edges. *)
      List.map
        (fun oid ->
          let s : Tensor_sig.t = Tensor_id.Map.find oid graph.Graph.tensors in
          s.shape)
        graph.Graph.outputs
