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
  let pixel (op : op) ~(operand : tensor_ref -> S.input)
      ~(shape_of : tensor_ref -> Vec6.shape)
      ~(fill : float -> Vec6.shape -> S.input)
      (out : Axis.t -> Semantics.position S.index) : S.t =
    match op with
    | Add { a; b } ->
        let module C = Pointwise.Add.Compute (S) in
        C.pixel ~a_shape:(shape_of a) ~b_shape:(shape_of b) (operand a)
          (operand b) out
    | Avg_pool2d { params; x } ->
        let module C = Pool.AvgPool2d.Compute (S) in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) out
    | Bmm { input; mat2 } ->
        let module C = Matmul.Bmm.Compute (S) in
        C.pixel ~input_shape:(shape_of input) ~input:(operand input)
          ~mat2:(operand mat2) out
    | Conv2d { params; x; weight; bias } ->
        let module C = Conv.Conv2d.Compute (S) in
        let bias =
          match bias with
          | Some b -> operand b
          | None -> fill 0. (bias_shape ~weight_shape:(shape_of weight))
        in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x)
          ~weight:(operand weight) ~bias out
    | Linear { params; x; weight; bias } ->
        let module C = Linear.Linear.Compute (S) in
        let bias =
          match bias with
          | Some b -> operand b
          | None -> fill 0. (bias_shape ~weight_shape:(shape_of weight))
        in
        C.pixel params ~x:(operand x) ~weight:(operand weight) ~bias out
    | Max_pool2d { params; x } ->
        let module C = Pool.MaxPool2d.Compute (S) in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) out
    | Mean { params; x } ->
        let module C = Reduce.Mean.Compute (S) in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) out
    | Mul { a; b } ->
        let module C = Pointwise.Mul.Compute (S) in
        C.pixel ~a_shape:(shape_of a) ~b_shape:(shape_of b) (operand a)
          (operand b) out
    | Permute { perm; x } ->
        let module C = Permute.Permute.Compute (S) in
        C.pixel perm ~x:(operand x) out
    | Relu { x } ->
        let module C = Pointwise.Relu.Compute (S) in
        C.pixel (operand x) out
    | Rms_norm { params; x; weight } ->
        let module C = Norm.RmsNorm.Compute (S) in
        let weight =
          match weight with Some w -> operand w | None -> fill 1. (shape_of x)
          (* absent weight = identity scale *)
        in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) ~weight out
    | Subgraph _ ->
        invalid_arg
          "Eval_op.pixel: Subgraph is handled by the graph traversal, not a \
           pixel"
end
