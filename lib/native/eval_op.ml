(* The single op-dispatch point, functorised over the semantics [S]. Applied once
   at [Direct] and once at a fresh [Symbolic.Make ()], so the per-op wiring lives
   in exactly one place and cannot drift between the two execution modes.

   Each arm reads its operands off the op's TYPED FIELDS and resolves each
   [tensor_ref] to a data handle ([operand]) and shape ([shape_of]); an absent
   optional operand becomes a constant-filled handle via [fill] (zeros for a
   missing bias, ones for a missing rms-norm weight). [fill] is the only
   S-specific extra capability the caller supplies — Direct fills a real tensor,
   Symbolic mints a signature it will bind to a constant at ground time. See
   .ai/native_graph_design.md. *)

open Graph_ir

(* Conv/Linear bias is laid out [1,1,1,1,1,Cout]; Cout is the weight's N extent. *)
let bias_shape ~weight_shape =
  Vec6.set
    (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1)
    Axis.C
    (Vec6.get weight_shape Axis.N)

module Make (S : Semantics.SEMANTICS) = struct
  (* [output] selects which output edge's pixel to build, for multi-output ops
     (0-based, in [Node.outputs] order). Single-output ops ignore it. *)
  let pixel (op : op) ~output ~(operand : tensor_ref -> S.input)
      ~(shape_of : tensor_ref -> Vec6.shape)
      ~(fill : float -> Vec6.shape -> S.input)
      (out : Semantics.position S.index Vec6.t) : S.t =
    match op with
    | Add { Pointwise.Bin.a; b } ->
        let module C = Pointwise.Add.Compute (S) in
        C.pixel ~a_shape:(shape_of a) ~b_shape:(shape_of b) (operand a)
          (operand b) out
    | Avg_pool2d { Pool.AvgPool2d.params; x } ->
        let module C = Pool.AvgPool2d.Compute (S) in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) out
    | Batch_norm
        { Norm.BatchNorm.params; x; weight; bias; running_mean; running_var } ->
        let module C = Norm.BatchNorm.Compute (S) in
        let weight =
          match weight with Some w -> operand w | None -> fill 1. (shape_of x)
          (* absent weight = identity scale *)
        in
        let bias =
          match bias with Some b -> operand b | None -> fill 0. (shape_of x)
          (* absent bias = identity shift *)
        in
        C.pixel params ~x:(operand x) ~weight ~bias
          ~running_mean:(operand running_mean)
          ~running_var:(operand running_var) out
    | Bmm { Matmul.Bmm.input; mat2 } ->
        let module C = Matmul.Bmm.Compute (S) in
        C.pixel ~input_shape:(shape_of input) ~input:(operand input)
          ~mat2:(operand mat2) out
    | Conv2d { Conv.Conv2d.params; x; weight; bias } ->
        let module C = Conv.Conv2d.Compute (S) in
        let bias =
          match bias with
          | Some b -> operand b
          | None -> fill 0. (bias_shape ~weight_shape:(shape_of weight))
        in
        C.pixel params ~x_shape:(shape_of x) ~weight_shape:(shape_of weight)
          ~x:(operand x) ~weight:(operand weight) ~bias out
    | Conv2d_padding { Conv.Conv2d_padding.params; x; weight; bias } ->
        let module C = Conv.Conv2d_padding.Compute (S) in
        let bias =
          match bias with
          | Some b -> operand b
          | None -> fill 0. (bias_shape ~weight_shape:(shape_of weight))
        in
        C.pixel params ~x_shape:(shape_of x) ~weight_shape:(shape_of weight)
          ~x:(operand x) ~weight:(operand weight) ~bias out
    | Convolution { Conv.Convolution.params; x; weight; bias } ->
        let module C = Conv.Convolution.Compute (S) in
        let bias =
          match bias with
          | Some b -> operand b
          | None ->
              fill 0.
                (Conv.Convolution.bias_shape ~weight_shape:(shape_of weight)
                   params)
        in
        C.pixel params ~x_shape:(shape_of x) ~weight_shape:(shape_of weight)
          ~x:(operand x) ~weight:(operand weight) ~bias out
    | Linear { Linear.Linear.params; x; weight; bias } ->
        let module C = Linear.Linear.Compute (S) in
        let bias =
          match bias with
          | Some b -> operand b
          | None -> fill 0. (bias_shape ~weight_shape:(shape_of weight))
        in
        C.pixel params ~x:(operand x) ~weight:(operand weight) ~bias out
    | Max_pool2d { Pool.MaxPool2d.params; x } ->
        let module C = Pool.MaxPool2d.Compute (S) in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) out
    | Max_pool2d_with_indices { Pool.MaxPool2dWithIndices.params; x } ->
        let module C = Pool.MaxPool2dWithIndices.Compute (S) in
        let pix = if output = 0 then C.value_pixel else C.index_pixel in
        pix params ~x_shape:(shape_of x) ~x:(operand x) out
    | Mean { Reduce.Mean.params; x } ->
        let module C = Reduce.Mean.Compute (S) in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) out
    | Mul { Pointwise.Bin.a; b } ->
        let module C = Pointwise.Mul.Compute (S) in
        C.pixel ~a_shape:(shape_of a) ~b_shape:(shape_of b) (operand a)
          (operand b) out
    | Permute { Permute.Permute.perm; x } ->
        let module C = Permute.Permute.Compute (S) in
        C.pixel perm ~x:(operand x) out
    | Relu { Pointwise.Relu.x } ->
        let module C = Pointwise.Relu.Compute (S) in
        C.pixel (operand x) out
    | Reshape { Reshape.Reshape.params; x } ->
        let module C = Reshape.Reshape.Compute (S) in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) out
    | Rms_norm { Norm.RmsNorm.params; x; weight } ->
        let module C = Norm.RmsNorm.Compute (S) in
        let weight =
          match weight with Some w -> operand w | None -> fill 1. (shape_of x)
          (* absent weight = identity scale *)
        in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) ~weight out
    | Discard _ ->
        invalid_arg
          "Eval_op.pixel: Discard produces no output, so it has no pixel"
end
